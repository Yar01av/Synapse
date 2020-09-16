from collections import deque

import gym
from tensorboardX import SummaryWriter
from torch import load, cuda, save, nn, tensor, isnan
from torch.optim import Adam
import torch.multiprocessing as mp
from action_selectors import BaseActionSelector, ProbValuePolicySelector
from agents.a2c.a2c import A2CNetwork
from agents.agent_training import AgentTraining
from steps_generators import MultiEnvCompressedStepsGenerator, CompressedTransition
import torch.nn.utils as nn_utils
import torch.nn.functional as F

from util import unpack


def env_maker(): return gym.make("CartPole-v1")


def child_process(queue, envs_per_thread, n_actions, model, n_steps, gamma):
    steps_generator = MultiEnvCompressedStepsGenerator([env_maker() for _ in range(envs_per_thread)],
                                                        ProbValuePolicySelector(n_actions, model=model),
                                                        n_steps=n_steps, gamma=gamma)

    for exp in steps_generator:
        queue.put(exp)

        if exp.done:
            new_rewards = steps_generator.pop_per_episode_rewards()
            assert len(new_rewards) == 1

            queue.put(new_rewards[0])


class A3CNetwork(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()

        self._base = nn.Sequential(
            nn.Linear(input_shape, 128),
            nn.ReLU()
        )

        self._value_head = nn.Sequential(
            nn.Linear(input_shape, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self._policy_head = nn.Sequential(
            nn.Linear(input_shape, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        #base_output = self._base(x)
        return self._policy_head(x), self._value_head(x)



class A3C(AgentTraining):
    """
    A version of A3C where different threads play with different environments but the learning is done centrally.
    """
    # TODO: more gpu

    def __init__(self, gamma=0.99, beta=0.01, lr=0.001, batch_size=8, max_training_steps=1000, desired_avg_reward=500,
                 unfolding_steps=2, envs_per_thread=1, clip_grad=0.1, n_processes=8):
        super().__init__()

        mp.set_start_method('spawn')

        self._n_processes = n_processes
        self._envs_per_thread = envs_per_thread
        self._unfolding_steps = unfolding_steps
        self._desired_avg_reward = desired_avg_reward
        self._beta = beta
        self._n_training_steps = max_training_steps
        self._batch_size = batch_size
        self._lr = lr
        self._gamma = gamma
        self._memory = list()
        self._clip_grad = clip_grad

        self._ref_env = self.get_environment()  # Reference environment should not be actually used to play episodes.
        self._model = A3CNetwork(self._ref_env.observation_space.shape[0], self._ref_env.action_space.n)
        self._model.share_memory()
        self._optimizer = Adam(params=self._model.parameters(), lr=lr, eps=1e-3)

        # Logging related
        self._plotter = SummaryWriter(comment=f"x{self.__class__.__name__}")

    def train(self, save_path):
        last_episodes_rewards = deque(maxlen=100)
        episode_idx = 0
        step_idx = 0
        # TODO try batch count
        train_queue = mp.Queue(maxsize=self._n_processes)
        processes = []

        # Spawn the processes
        for _ in range(self._n_processes):
            process = mp.Process(target=child_process, args=(train_queue,
                                                             self._envs_per_thread,
                                                             self.get_environment().action_space.n,
                                                             self._model,
                                                             self._unfolding_steps,
                                                             self._gamma))
            process.start()
            processes.append(process)

        try:
            while True:
                # Check if we should stop already
                if step_idx == self._n_training_steps-1 or \
                        (self._desired_avg_reward <= sum(last_episodes_rewards)/len(last_episodes_rewards)
                        if len(last_episodes_rewards) >= 100 else False):
                    save(self._model, save_path)
                    break

                exp = train_queue.get()

                # Check if the given experience is an actual experience or an episode reward
                if not isinstance(exp, CompressedTransition):
                    last_episodes_rewards.append(exp)
                    self._plotter.add_scalar("Total reward per episode", exp, episode_idx)
                    print(f"At step {step_idx}, \t the average over the last 100 games is "
                          f"{sum(last_episodes_rewards)/min(len(last_episodes_rewards), 100)}")
                    continue
                else:
                    episode_idx += 1
                    self._memory.append(exp)

                if len(self._memory) == self._batch_size:
                    t_states, t_actions, t_qvals = unpack(self._memory,
                                                          self._model,
                                                          self._gamma,
                                                          self._unfolding_steps,
                                                          "cpu")

                    self._optimizer.zero_grad()
                    t_logits, t_values = self._model(t_states)
                    t_log_probs = t_logits.log_softmax(dim=1)
                    t_probs = t_logits.softmax(dim=1)

                    # Compute the value loss
                    value_loss = F.mse_loss(t_values.squeeze(-1), t_qvals)

                    # Compute the policy loss
                    t_advantages = t_qvals - t_values.detach()
                    policy_loss = -(t_advantages * t_log_probs[range(self._batch_size), t_actions]).mean()

                    # Compute the entropy and record the original probabilities for later
                    entropy = -(t_probs * t_log_probs).sum(dim=1).mean()
                    old_probs = t_probs

                    (policy_loss + value_loss - self._beta * entropy).backward()
                    #nn_utils.clip_grad_norm_(self._model.parameters(), self._clip_grad)
                    self._optimizer.step()

                    # Compute KL divergence
                    new_probs = self._model(t_states)[0].softmax(dim=1)
                    kl_divergence = -((new_probs / old_probs).log() * old_probs).sum(dim=1).mean()

                    self._memory.clear()

                    # Plot
                    self._plotter.add_scalar("Combined Loss", (policy_loss+value_loss-self._beta*entropy).item(), step_idx)
                    self._plotter.add_scalar("Entropy", entropy.item(), step_idx)
                    self._plotter.add_scalar("KL Divergence", kl_divergence.item(), step_idx)

                step_idx += 1
        finally:
            for proc in processes:
                proc.terminate()
                proc.join()


    @classmethod
    def load_selector(cls, load_path) -> BaseActionSelector:
        return ProbValuePolicySelector(action_space_size=cls.get_environment().action_space.n,
                                       model=load(load_path))

    @classmethod
    def get_environment(cls):
        return env_maker()