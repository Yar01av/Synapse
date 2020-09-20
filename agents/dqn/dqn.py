from collections import deque

import gym
from tensorboardX import SummaryWriter
from torch import cuda, nn, load, save, LongTensor, FloatTensor
from torch.optim import Adam
from action_selectors import BaseActionSelector, ValueSelector
from agents.agent_training import AgentTraining
from memory import CompositeMemory
import torch.nn.utils as nn_utils
from steps_generators import SimpleStepsGenerator, CompressedStepsGenerator, MultiEnvCompressedStepsGenerator
import torch.nn.functional as F

from util import unpack


class DQNNetwork(nn.Module):
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


class DQN(AgentTraining):
    def __init__(self,
                 gamma=0.99,
                 lr=0.001,
                 batch_size=33,
                 max_training_steps=1000,
                 desired_avg_reward=500,
                 max_buffer_size=1_000_000,
                 epsilon=1.0,
                 epsilon_min=0.01,
                 epsilon_decay=0.996):
        super().__init__()

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
        self._optimizer = Adam(params=self._model.parameters(), lr=lr, eps=1e-3)

        # Logging related
        self._plotter = SummaryWriter(comment=f"x{self.__class__.__name__}")

    def train(self, save_path):
        steps_generator = CompressedStepsGenerator(ValueSelector(model=self._model),
                                                   n_steps=self._unfolding_steps, gamma=self._gamma)




        self._plotter.close()

    @classmethod
    def load_selector(cls, load_path) -> BaseActionSelector:
        return ValueSelector(model=load(load_path))

    @classmethod
    def get_environment(cls):
        return gym.make("CartPole-v1")