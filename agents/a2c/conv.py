import gym
import numpy as np
import ptan
from torch import nn, LongTensor, FloatTensor, flatten
from torch.optim import Adam
from agents.a2c.standard import A2C

from util import get_output_size


class A2CConvNetwork(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()

        self._base = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, stride=4, kernel_size=8),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = get_output_size(self._base, input_shape)
        self._value_head = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self._policy_head = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x):
        base_output = self._base(x.float()/256)
        base_output = flatten(base_output, start_dim=1)

        return self._policy_head(base_output), self._value_head(base_output)


class A2CConv(A2C):
    def __init__(self, gamma=0.99, beta=0.01, lr=0.001, batch_size=8, max_training_steps=1000, desired_avg_reward=500,
                 unfolding_steps=2, n_envs=1, clip_grad=0.1):
        super().__init__(gamma, beta, lr, batch_size, max_training_steps, desired_avg_reward, unfolding_steps, n_envs,
                         clip_grad)

        self._model = A2CConvNetwork(self._ref_env.observation_space.shape, self._ref_env.action_space.n).cuda()
        self._optimizer = Adam(params=self._model.parameters(), lr=lr, eps=1e-3)

    @classmethod
    def get_environment(cls):
        return ptan.common.wrappers.wrap_dqn(gym.make("PongNoFrameskip-v4"))