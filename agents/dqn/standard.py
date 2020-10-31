from collections import deque

import gym
from tensorboardX import SummaryWriter
from torch import cuda, nn, load, save, LongTensor, FloatTensor, IntTensor, squeeze, max
from torch.optim import Adam
from action_selectors.base import BaseActionSelector
from action_selectors.value import GreedySelector, EpsilonGreedySelector
from agents.base import AgentTraining
from steps_generators import CompressedStepsGenerator
import random

from util import can_stop


# The neural network module
class DQNNetwork(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()

        self._model = nn.Sequential(nn.Linear(input_shape, 150),
                                    nn.ReLU(),
                                    nn.Linear(150, 120),
                                    nn.ReLU(),
                                    nn.Linear(120, n_actions))

    def forward(self, x):
        return self._model(x)


class DQN(AgentTraining):
    def __init__(self,
                 gamma=0.99,
                 lr=0.001,
                 batch_size=34,
                 max_training_steps=10_000_000,
                 desired_avg_reward=500,
                 max_buffer_size=1_000_000,
                 epsilon=1.0,
                 epsilon_min=0.01,
                 unfolding_steps=1,
                 epsilon_decay=0.996):
        super().__init__()

        self._unfolding_steps = unfolding_steps
        self._epsilon_decay = epsilon_decay
        self._epsilon_min = epsilon_min
        self._epsilon = epsilon
        self._max_buffer_size = max_buffer_size
        self._desired_avg_reward = desired_avg_reward
        self._n_training_steps = max_training_steps
        self._batch_size = batch_size
        self._lr = lr
        self._gamma = gamma
        cuda.set_device(0)
        self._buffer = deque(maxlen=max_buffer_size)

        self._ref_env = self.get_environment()  # Reference environment should not be actually used to play episodes.
        self._model = DQNNetwork(self._ref_env.observation_space.shape[0], self._ref_env.action_space.n).cuda()
        self._optimizer = Adam(params=self._model.parameters(), lr=lr)

        # Logging related
        self._plotter = SummaryWriter(comment=f"x{self.__class__.__name__}")

    def train(self, save_path):
        action_selector = EpsilonGreedySelector(model=self._model,
                                                n_actions=self._ref_env.action_space.n,
                                                model_device="cuda",
                                                init_epsilon=self._epsilon,
                                                min_epsilon=self._epsilon_min,
                                                epsilon_decay=self._epsilon_decay)
        steps_generator = CompressedStepsGenerator(self.get_environment(),
                                                   action_selector,
                                                   n_steps=self._unfolding_steps,
                                                   gamma=self._gamma)

        last_episodes_rewards = deque(maxlen=100)
        episode_idx = 0

        # Iterate over the transitions and learn if the right transition arrives
        for idx, transition in enumerate(steps_generator):
            # Stop if the right number of reward has been reached or the maximum number of the iterations exceeded.
            if can_stop(idx, self._n_training_steps, last_episodes_rewards, self._desired_avg_reward):
                save(self._model, save_path)
                break

            # Remember the new transition
            self._buffer.append(transition)

            # Log the results at the end of an episode
            if transition.done:
                episode_idx += 1

                # Log the new rewards
                new_rewards = steps_generator.pop_per_episode_rewards()
                assert len(new_rewards) == 1
                last_episodes_rewards.extend(new_rewards)
                self._plotter.add_scalar("Total reward per episode", new_rewards[0], episode_idx)
                print(f"At step {idx}, \t the average over the last 100 games is "
                      f"{sum(last_episodes_rewards)/min(len(last_episodes_rewards), 100)}")

            # Perform a training step
            if len(self._buffer) >= self._batch_size:
                states, targets = self._replay(self._buffer, self._batch_size)
                self._learn(states, targets, self._optimizer)

            # Reduce exploration
            action_selector.decay_epsilon()

        self._plotter.close()

    def _learn(self, states, targets, optimizer):
        predictions = self._model(states.to("cuda"))
        loss = nn.MSELoss()(predictions, targets.to("cuda"))
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

    def _replay(self, memory, batch_size):
        # Sample the minibatch and extract the useful tensors out of it
        minibatch = random.sample(memory, batch_size)
        states = FloatTensor([t.previous_state for t in minibatch]).to("cuda")
        actions = LongTensor([t.action for t in minibatch]).to("cuda")
        rewards = FloatTensor([t.reward for t in minibatch]).to("cuda")
        next_states = FloatTensor([t.next_state for t in minibatch]).to("cuda")
        dones = IntTensor([t.done for t in minibatch]).to("cuda")

        # The states may need to be squeezed
        states = squeeze(states)
        next_states = squeeze(next_states)

        # Use Bellman equation to compute the targets
        targets = rewards + \
                  self._gamma * max(self._model(next_states).detach(), dim=1)[0].to("cuda") \
                  * (-dones + 1)
        targets.to("cuda")
        targets_full = self._model(states).detach()
        ind = LongTensor([i for i in range(self._batch_size)]).to("cuda")

        # Add the targets to the original model prediction
        targets_full[ind, actions] = targets

        return states, targets_full

    @classmethod
    def load_selector(cls, load_path) -> BaseActionSelector:
        # It's important to return this selector as the algorithm is off-policy.
        return GreedySelector(model=load(load_path))

    @classmethod
    def get_environment(cls):
        env = gym.make("CartPole-v1")
        env.seed(0)

        return env
