from collections import deque
from copy import deepcopy
from pathlib import Path

import gym
from torch.optim import Adam
from tensorboardX import SummaryWriter

from synapse.action_selectors.policy import PolicyActionSelector
from ..base import DiscreteAgentTraining
from torch import load, nn, cuda, save, LongTensor, FloatTensor
from synapse.memory import CompositeMemory
from synapse.steps_generators import CompressedStepsGenerator
from synapse.util import can_stop


class REINFORCE(DiscreteAgentTraining):
    def __init__(self,
                 env: gym.Env,
                 model,
                 gamma=0.99,
                 beta=0.01,
                 lr=0.01,
                 batch_size=10,
                 max_training_steps=1000,
                 desired_avg_reward=500,
                 device="cuda"):
        super().__init__()

        self._device = device
        self._model = model
        self._env = env
        self._desired_avg_reward = desired_avg_reward
        self._beta = beta
        self._n_training_steps = max_training_steps
        self._batch_size = batch_size
        self._lr = lr
        self._gamma = gamma
        self._optimizer = Adam(params=self._model.parameters(), lr=lr)
        self._memory = CompositeMemory()
        self._steps_generator = CompressedStepsGenerator(self._env, PolicyActionSelector(model=self._model))

        # Logging related
        SummaryWriter(comment=f"x{self.__class__.__name__}").close()
        self._plotter = SummaryWriter(comment=f"x{self.__class__.__name__}", logdir=Path("../../runs"))

    def train(self, save_path):
        batch_count = 0
        last_episodes_rewards = deque(maxlen=100)
        episode_idx = 0

        for idx, transition in enumerate(self._steps_generator):
            if can_stop(idx, self._n_training_steps, last_episodes_rewards, self._desired_avg_reward):
                save(self._model, save_path)
                break

            self._remember(transition)

            if transition.done:
                batch_count += 1
                episode_idx += 1

                self._handle_finished_episode(episode_idx, idx, last_episodes_rewards, self._steps_generator)

            if batch_count == self._batch_size:
                self._learn(episode_idx)

                batch_count = 0

        self._plotter.close()

    def _learn(self, episode_idx):
        t_act = LongTensor(self._memory.actions).to(self._device)
        t_state = FloatTensor(self._memory.states).to(self._device)
        t_qval = FloatTensor(self._memory.q_vals).to(self._device)

        self._optimizer.zero_grad()

        t_logits = self._model(t_state)

        # Compute the policy loss
        policy_loss = -(t_qval * t_logits.log_softmax(dim=1)[range(len(t_state)), t_act]).mean()

        # Compute the entropy and record the original probabilities for later
        entropy = -(t_logits.softmax(dim=1) * t_logits.log_softmax(dim=1)).sum(dim=1).mean()
        old_probs = t_logits.softmax(dim=1)
        (policy_loss - self._beta * entropy).backward()

        self._optimizer.step()

        # Compute KL divergence
        new_probs = self._model(t_state).softmax(dim=1)
        kl_divergence = -((new_probs / old_probs).log() * old_probs).sum(dim=1).mean()
        self._memory.reset()

        # Plot
        self._plotter.add_scalar("Entropy", entropy.item(), episode_idx)
        self._plotter.add_scalar("KL Divergence", kl_divergence.item(), episode_idx)

    def _handle_finished_episode(self, episode_idx, idx, last_episodes_rewards, steps_generator):
        self._memory.compute_qvals(self._gamma)

        new_rewards = steps_generator.pop_per_episode_rewards()
        assert len(new_rewards) == 1
        last_episodes_rewards.extend(new_rewards)

        self._plotter.add_scalar("Total reward per episode", new_rewards[0], episode_idx)
        print(f"At step {idx}, \t the average over the last 100 games is {sum(last_episodes_rewards) / 100}")

    def _remember(self, transition):
        self._memory.states.append(transition.previous_state)
        self._memory.actions.append(transition.action)
        self._memory.rewards.append(transition.reward)

    @property
    def model(self):
        return deepcopy(self._model)
