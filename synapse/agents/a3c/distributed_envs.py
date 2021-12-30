from collections import deque
from copy import deepcopy
from pathlib import Path

import gym
import torch.multiprocessing as mp
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch import save
from torch.optim import Adam

from synapse.action_selectors.policy import PolicyActionsSelector
from synapse.agents.base import DiscreteAgentTraining
from synapse.steps_generators import MultiEnvCompressedStepsGenerator, CompressedTransition
from synapse.util import unpack, can_stop


def env_maker(): return gym.make("CartPole-v1")


def child_process(queue, envs_per_thread, model, n_steps, gamma):
    steps_generator = MultiEnvCompressedStepsGenerator([env_maker() for _ in range(envs_per_thread)],
                                                       PolicyActionsSelector(model=lambda x: model(x)[0],
                                                                             model_device="cpu"),
                                                       n_steps=n_steps, gamma=gamma)

    # Play the game and push the experiences into the queue
    for exp in steps_generator:
        queue.put(exp)

        if exp.done:
            new_rewards = steps_generator.pop_per_episode_rewards()
            assert len(new_rewards) == 1

            queue.put(new_rewards[0])


class A3C(DiscreteAgentTraining):
    """
    A version of A3C where different threads play with different environments but the learning is done centrally.
    """

    def __init__(self,
                 env,
                 model,
                 gamma=0.99,
                 beta=0.01,
                 lr=0.001,
                 batch_size=8,
                 max_training_steps=1000,
                 desired_avg_reward=500,
                 unfolding_steps=2,
                 envs_per_thread=1,
                 clip_grad=0.1,
                 n_processes=8):
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

        self._env = env
        self._model = model
        self._model.share_memory()
        self._optimizer = Adam(params=self._model.parameters(), lr=lr, eps=1e-3)

        # Logging related
        self._plotter = SummaryWriter(comment=f"x{self.__class__.__name__}", logdir=Path("../../runs"))

    def train(self, save_path):
        last_episodes_rewards = deque(maxlen=100)
        episode_idx = 0
        step_idx = 0
        # TODO try batch count
        train_queue = mp.Queue(maxsize=self._n_processes)
        processes = []

        # Spawn the processes
        for _ in range(self._n_processes):
            processes.append(self._spawn_experience_generating_process(train_queue))

        try:
            while True:
                # Check if we should stop already
                if can_stop(step_idx, self._n_training_steps, last_episodes_rewards, self._desired_avg_reward):
                    save(self._model, save_path)
                    break

                # Extract an experience from the queue
                exp = train_queue.get()

                # Check if the given experience is an actual experience or an episode reward
                # If it is a reward, log it. Otherwise, store it in the memory.
                if not isinstance(exp, CompressedTransition):
                    self._handle_reward(last_episodes_rewards, exp, episode_idx, step_idx)
                    continue
                else:
                    episode_idx += 1
                    self._memory.append(exp)

                # Train the model if there are enough experiences to do so.
                if len(self._memory) == self._batch_size:
                    self._learn(step_idx)

                step_idx += 1
        finally:
            # Close all the running processes so as to avoid zombies
            for proc in processes:
                proc.terminate()
                proc.join()

    def _handle_reward(self, last_episodes_rewards, exp, episode_idx, step_idx):
        last_episodes_rewards.append(exp)
        self._plotter.add_scalar("Total reward per episode", exp, episode_idx)
        print(f"At step {step_idx}, \t the average over the last 100 games is "
              f"{sum(last_episodes_rewards) / min(len(last_episodes_rewards), 100)}")

    def _spawn_experience_generating_process(self, train_queue):
        process = mp.Process(target=child_process, args=(train_queue,
                                                         self._envs_per_thread,
                                                         self._model,
                                                         self._unfolding_steps,
                                                         self._gamma))
        process.start()

        return process

    def _learn(self, step_idx):
        # Extract the states, actions and values as tensors from the memory
        t_states, t_actions, t_qvals = unpack(self._memory,
                                              self._model,
                                              self._gamma,
                                              self._unfolding_steps,
                                              "cpu")
        # Clear the old gradients
        self._optimizer.zero_grad()

        # Convert the network output into actual probabilities and their logs for later use
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

        # Put the complete expression for the loss together and backpropagate the gradients
        (policy_loss + value_loss - self._beta * entropy).backward()
        # nn_utils.clip_grad_norm_(self._model.parameters(), self._clip_grad)
        self._optimizer.step()

        # Compute KL divergence
        new_probs = self._model(t_states)[0].softmax(dim=1)
        kl_divergence = -((new_probs / old_probs).log() * old_probs).sum(dim=1).mean()

        # Empty the memory as the algorithm is on-policy and the policy has changed with the model
        self._memory.clear()

        # Plot
        self._plotter.add_scalar("Combined Loss", (policy_loss + value_loss - self._beta * entropy).item(), step_idx)
        self._plotter.add_scalar("Entropy", entropy.item(), step_idx)
        self._plotter.add_scalar("KL Divergence", kl_divergence.item(), step_idx)

    @property
    def model(self):
        return deepcopy(self._model)
