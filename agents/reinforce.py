from collections import deque

import gym
from torch.optim import Adam
from tensorboardX import SummaryWriter

from action_selectors import BaseActionSelector, SimplePolicySelector
from agents.agent_training import AgentTraining
from torch import load, nn, cuda, save, LongTensor, FloatTensor
from memory import CompositeMemory
from steps_generators import SimpleStepsGenerator, CompressedStepsGenerator


class REINFORCE(AgentTraining):
    def __init__(self, gamma=0.99, beta=0.01, lr=0.01, batch_size=10, max_training_steps=1000, desired_avg_reward=500):
        super().__init__()

        self._env = self.get_environment()
        self._desired_avg_reward = desired_avg_reward
        self._beta = beta
        self._n_training_steps = max_training_steps
        self._batch_size = batch_size
        self._lr = lr
        self._gamma = gamma
        cuda.set_device(0)
        self._model = nn.Sequential(nn.Linear(self._env.observation_space.shape[0], 128),
                                    nn.ReLU(),
                                    nn.Linear(128, self._env.action_space.n)).cuda()
        self._optimizer = Adam(params=self._model.parameters(), lr=lr)
        self._memory = CompositeMemory()

        # Logging related
        SummaryWriter(comment=f"x{self.__class__.__name__}").close()
        self._plotter = SummaryWriter(comment=f"x{self.__class__.__name__}")

    def train(self, save_path):
        steps_generator = CompressedStepsGenerator(self._env,
                                                   SimplePolicySelector(self._env.action_space.n, model=self._model))

        batch_count = 0
        last_episodes_rewards = deque(maxlen=100)
        episode_idx = 0

        for idx, transition in enumerate(steps_generator):
            if idx == self._n_training_steps-1 or self._desired_avg_reward <= sum(last_episodes_rewards)/100:
                save(self._model, save_path)
                break

            self._memory.states.append(transition.previous_state)
            self._memory.actions.append(transition.action)
            self._memory.rewards.append(transition.reward)

            if transition.done:
                batch_count += 1
                episode_idx += 1

                self._memory.compute_qvals(self._gamma)

                new_rewards = steps_generator.pop_per_episode_rewards()
                assert len(new_rewards) == 1
                last_episodes_rewards.extend(new_rewards)
                self._plotter.add_scalar("Total reward per episode", new_rewards[0], episode_idx)
                print(f"At step {idx}, \t the average over the last 100 games is {sum(last_episodes_rewards)/100}")

            if batch_count == self._batch_size:
                t_act = LongTensor(self._memory.actions).cuda()
                t_state = FloatTensor(self._memory.states).cuda()
                t_qval = FloatTensor(self._memory.q_vals).cuda()

                self._optimizer.zero_grad()
                t_logits = self._model(t_state)

                # Compute the policy loss
                policy_loss = -(t_qval*t_logits.log_softmax(dim=1)[range(len(t_state)), t_act]).mean()
                # Compute the entropy and record the original probabilities for later
                entropy = -(t_logits.softmax(dim=1)*t_logits.log_softmax(dim=1)).sum(dim=1).mean()
                old_probs = t_logits.softmax(dim=1)

                (policy_loss-self._beta*entropy).backward()
                self._optimizer.step()

                # Compute KL divergence
                new_probs = self._model(t_state).softmax(dim=1)
                kl_divergence = -((new_probs/old_probs).log()*old_probs).sum(dim=1).mean()

                batch_count = 0
                self._memory.reset()

                # Plot
                self._plotter.add_scalar("Entropy", entropy.item(), episode_idx)
                self._plotter.add_scalar("KL Divergence", kl_divergence.item(), episode_idx)

        self._plotter.close()

    @classmethod
    def load_selector(cls, load_path) -> BaseActionSelector:
        return SimplePolicySelector(action_space_size=cls.get_environment().action_space.n, model=load(load_path))

    @classmethod
    def get_environment(cls):
        return gym.make("LunarLander-v2")
