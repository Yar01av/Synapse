import random
from collections import deque
from copy import deepcopy
from pathlib import Path
import time
import gym
from tensorboardX import SummaryWriter
import torch
from torch import cuda, nn, save, LongTensor, FloatTensor, IntTensor, squeeze, max
import torch.nn.functional as F

from gene.optimisers.division import DivisionOptimiser, ParallelDivisionOptimiser
from gene.selections.top_n import TopNSelection
from gene.targets import make_supervised_loss
from synapse.action_selectors.other import EpsilonActionSelector
from synapse.steps_generators import CompressedStepsGenerator
from synapse.util import can_stop
from ..base import DiscreteAgentTraining
from ...action_selectors.value import GreedyActionSelector


class GeneticGradientDQN(DiscreteAgentTraining):
    def __init__(self,
                 environment: gym.Env,
                 models,
                 gamma=0.99,
                 batch_size=34,
                 max_training_steps=10_000_000,
                 desired_avg_reward=500,
                 max_buffer_size=1_000_000,
                 epsilon=1.0,
                 epsilon_min=0.01,
                 unfolding_steps=1,
                 epsilon_decay=0.996,
                 selection=TopNSelection(100),
                 device="cuda",
                 logdir="./runs"):
        super().__init__()

        self._environment = environment
        self._device = device
        self._unfolding_steps = unfolding_steps
        self._epsilon_decay = epsilon_decay
        self._epsilon_min = epsilon_min
        self._epsilon = epsilon
        self._max_buffer_size = max_buffer_size
        self._desired_avg_reward = desired_avg_reward
        self._n_training_steps = max_training_steps
        self._batch_size = batch_size
        self._gamma = gamma
        cuda.set_device(0)
        self._buffer = deque(maxlen=max_buffer_size)
        self._models = models

        # Optimiser variables
        self._optimizer = DivisionOptimiser(random_function=lambda shape: torch.normal(0,
                                                                                       self._mutation_variance,
                                                                                       shape),
                                            selection=selection,
                                            device=device)

        inner_selector = GreedyActionSelector(model=lambda x: self._models[0](x), model_device=self._device)
        self._action_selector = EpsilonActionSelector(selector=inner_selector,
                                                      n_actions=self._environment.action_space.n,
                                                      init_epsilon=self._epsilon,
                                                      min_epsilon=self._epsilon_min,
                                                      epsilon_decay=self._epsilon_decay)
        self._steps_generator = CompressedStepsGenerator(self._environment,
                                                         self._action_selector,
                                                         n_steps=self._unfolding_steps,
                                                         gamma=self._gamma)

        # Logging related
        self._plotter = SummaryWriter(logdir=f"{logdir}/{time.strftime('%c')}--{self.__class__.__name__}")

    def train(self, save_path):
        last_episodes_rewards = deque(maxlen=100)
        episode_idx = 0

        # Iterate over the transitions and learn if the right transition arrives
        for idx, transition in enumerate(self._steps_generator):
            # Stop if the right number of reward has been reached or the maximum number of the iterations exceeded.
            if can_stop(idx, self._n_training_steps, last_episodes_rewards, self._desired_avg_reward):
                save(self._models[0], save_path)
                break

            # Remember the new transition
            self._buffer.append(transition)

            # Log the results at the end of an episode
            if transition.done:
                episode_idx += 1
                self._log_new_rewards(episode_idx, idx, last_episodes_rewards, self._steps_generator)
                self._plotter.add_scalar("Epsilon", self._action_selector.epsilon, episode_idx)

            # Perform a training step
            if len(self._buffer) >= self._batch_size:
                states, targets = self._replay(self._buffer, self._batch_size)

                for _ in range(100):
                    loss_for_model = make_supervised_loss(
                        loss=nn.MSELoss(),
                        X=states.to(self._device),
                        y=targets
                    )
                    self._models = self._optimizer.step(models=self._models, loss_function=loss_for_model)

            # Reduce exploration
            self._action_selector.decay_epsilon()

        self._plotter.close()

    def _log_new_rewards(self, episode_idx, idx, last_episodes_rewards, steps_generator):
        new_rewards = steps_generator.pop_per_episode_rewards()
        assert len(new_rewards) == 1
        last_episodes_rewards.extend(new_rewards)
        self._plotter.add_scalar("Total reward per episode", new_rewards[0], episode_idx)
        print(f"At step {idx}, \t the average over the last 100 games is "
              f"{sum(last_episodes_rewards) / min(len(last_episodes_rewards), 100)}")

    def _replay(self, memory, batch_size):
        # Sample the minibatch and extract the useful tensors out of it
        minibatch = random.sample(memory, batch_size)
        states = FloatTensor([t.previous_state for t in minibatch]).to(self._device)
        actions = LongTensor([t.action for t in minibatch]).to(self._device)
        rewards = FloatTensor([t.reward for t in minibatch]).to(self._device)
        next_states = FloatTensor([t.next_state for t in minibatch]).to(self._device)
        dones = IntTensor([t.done for t in minibatch]).to(self._device)

        # The states may need to be squeezed
        states = squeeze(states)
        next_states = squeeze(next_states)

        # Use Bellman equation to compute the targets
        targets = rewards + self._gamma * max(self._models[0](next_states).detach(), dim=1)[0].to(self._device) * (
                -dones + 1)
        targets.to(self._device)
        targets_full = self._models[0](states).detach()
        ind = LongTensor([i for i in range(self._batch_size)]).to(self._device)

        # Add the targets to the original model prediction
        targets_full[ind, actions] = targets

        return states, targets_full

    @property
    def model(self):
        return deepcopy(self._models[0])
