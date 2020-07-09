from collections import deque

import gym
from torch.optim import Adam
from tensorboardX import SummaryWriter

from action_selectors import BaseActionSelector, SimplePolicySelector
from agents.agent_training import AgentTraining
from torch import load, nn, cuda, save, LongTensor, FloatTensor
import numpy as np
from memory import CompositeMemory
from steps_generators import SimpleStepsGenerator


class REINFORCE(AgentTraining):
    def __init__(self, gamma=0.99, beta=0.01, lr=0.01, batch_size=10, max_training_steps=1000, desired_avg_reward=500):
        super().__init__()

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
        self._plotter = SummaryWriter(comment="xREINFORCE")

    def train(self, save_path):
        steps_generator = SimpleStepsGenerator(self._env,
                                               SimplePolicySelector(self._env.action_space.n, model=self._model))
        batch_count = 0
        last_episodes_rewards = deque(maxlen=100)
        reward_sum = 0
        episode_idx = 0

        for idx, transition in enumerate(steps_generator):
            if idx == self._n_training_steps-1 or self._desired_avg_reward <= sum(last_episodes_rewards)/100:
                save(self._model, save_path)
                break

            reward_sum += transition.reward
            self._memory.states.append(transition.previous_state)
            self._memory.actions.append(transition.action)
            self._memory.rewards.append(transition.reward)

            if transition.done:
                batch_count += 1
                episode_idx += 1

                self._memory.compute_qvals(self._gamma)
                last_episodes_rewards.append(reward_sum)

                self._plotter.add_scalar("Total reward per episode", reward_sum, episode_idx)
                print(f"At step {idx}, \t the average over the last 100 games is {sum(last_episodes_rewards)/100}")

                reward_sum = 0

            if batch_count == self._batch_size:
                t_act = LongTensor(self._memory.actions).cuda()
                t_state = FloatTensor(self._memory.states).cuda()
                t_qval = FloatTensor(self._memory.q_vals).cuda()

                self._optimizer.zero_grad()
                t_logits = self._model(t_state)

                # Compute the policy loss
                policy_loss = -(t_qval*t_logits.log_softmax(dim=1)[range(len(t_state)), t_act]).mean()
                # Compute the entropy
                entropy = -(t_logits.softmax(dim=1)*t_logits.log_softmax(dim=1)).sum(dim=1).mean()
                # Compute KL divergence

                (policy_loss-self._beta*entropy).backward()
                self._optimizer.step()

                batch_count = 0
                self._memory.reset()

                # Plot
                self._plotter.add_scalar("Entropy", float(entropy), episode_idx)

    @classmethod
    def load_selector(cls, load_path) -> BaseActionSelector:
        return SimplePolicySelector(action_space_size=cls.get_environment().action_space.n, model=load(load_path))
