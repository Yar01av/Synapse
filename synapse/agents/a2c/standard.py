import os
from collections import deque
from copy import deepcopy

import gym
from tensorboardX import SummaryWriter
from torch import cuda, nn, load, save
from torch.optim import Adam
from synapse.action_selectors.base import ActionSelector
from synapse.action_selectors.policy import PolicyActionSelector, PolicyActionsSelector
from ..base import DiscreteAgentTraining
import torch.nn.utils as nn_utils
from synapse.steps_generators import MultiEnvCompressedStepsGenerator
import torch.nn.functional as F

from synapse.util import unpack, can_stop


class A2C(DiscreteAgentTraining):
    def __init__(self,
                 environment: gym.Env,
                 model,
                 device="cuda",
                 gamma=0.99,
                 beta=0.01,
                 lr=0.001,
                 batch_size=8,
                 max_training_steps=1000,
                 desired_avg_reward=500,
                 unfolding_steps=2,
                 n_envs=1,
                 clip_grad=1e12):
        super().__init__()

        self._device = device
        self._model = model
        self._environment = environment
        self._n_envs = n_envs
        self._unfolding_steps = unfolding_steps
        self._desired_avg_reward = desired_avg_reward
        self._beta = beta
        self._n_training_steps = max_training_steps
        self._batch_size = batch_size
        self._lr = lr
        self._gamma = gamma
        cuda.set_device(0)
        self._memory = list()
        self._clip_grad = clip_grad

        self._optimizer = Adam(params=self._model.parameters(), lr=lr, eps=1e-3)
        envs = [deepcopy(self._environment) for i in range(self._n_envs)]
        def first_head_model(obs): return self._model(obs)[0]
        self._steps_generator = MultiEnvCompressedStepsGenerator(envs,
                                                                 PolicyActionsSelector(model=first_head_model),
                                                                 n_steps=self._unfolding_steps,
                                                                 gamma=self._gamma)

        # Logging related
        self._plotter = SummaryWriter(comment=f"x{self.__class__.__name__}")

    def train(self, save_path):
        last_episodes_rewards = deque(maxlen=100)
        episode_idx = 0

        for idx, transition in enumerate(self._steps_generator):
            if can_stop(idx, self._n_training_steps, last_episodes_rewards, self._desired_avg_reward):
                save(self._model, save_path)
                break

            self._memory.append(transition)

            if transition.done:
                episode_idx += 1

                self._log_new_rewards(episode_idx, idx, last_episodes_rewards, self._steps_generator)

            if len(self._memory) == self._batch_size:
                self._learn(idx)

        self._plotter.close()

    def _log_new_rewards(self, episode_idx, idx, last_episodes_rewards, steps_generator):
        new_rewards = steps_generator.pop_per_episode_rewards()
        assert len(new_rewards) == 1
        last_episodes_rewards.extend(new_rewards)

        self._plotter.add_scalar("Total reward per episode", new_rewards[0], episode_idx)
        print(f"At step {idx}, \t the average over the last 100 games is "
              f"{sum(last_episodes_rewards) / min(len(last_episodes_rewards), 100)}")

    def _learn(self, idx):
        t_states, t_actions, t_qvals = unpack(self._memory,
                                              self._model,
                                              self._gamma,
                                              self._unfolding_steps,
                                              self._device)
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
        nn_utils.clip_grad_norm_(self._model.parameters(), self._clip_grad)
        self._optimizer.step()

        # Compute KL divergence
        new_probs = self._model(t_states)[0].softmax(dim=1)
        kl_divergence = -((new_probs / old_probs).log() * old_probs).sum(dim=1).mean()
        self._memory.clear()

        # Plot
        self._plotter.add_scalar("Combined Loss", (policy_loss + value_loss - self._beta * entropy).item(), idx)
        self._plotter.add_scalar("Entropy", entropy.item(), idx)
        self._plotter.add_scalar("KL Divergence", kl_divergence.item(), idx)

    @property
    def model(self):
        return deepcopy(self._model)
