from collections import deque

import gym
import numpy as np
import ptan
from tensorboardX import SummaryWriter
from torch import cuda, nn, load, save, LongTensor, FloatTensor, flatten
from torch.optim import Adam
from action_selectors import ProbValuePolicySelector, BaseActionSelector
from agents.a2c import A2C
from agents.agent_training import AgentTraining
from memory import CompositeMemory
import torch.nn.utils as nn_utils
from steps_generators import SimpleStepsGenerator, CompressedStepsGenerator, MultiEnvCompressedStepsGenerator
import torch.nn.functional as F

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
        #base_output = base_output.view(base_output.size()[0], -1)

        y=self._policy_head(base_output)
        z=self._value_head(base_output)
        return y, z


class A2CConv(A2C):
    def __init__(self, gamma=0.99, beta=0.01, lr=0.001, batch_size=8, max_training_steps=1000, desired_avg_reward=500,
                 unfolding_steps=2, n_envs=1, clip_grad=0.1):
        super().__init__(gamma, beta, lr, batch_size, max_training_steps, desired_avg_reward, unfolding_steps, n_envs,
                         clip_grad)

        self._model = A2CConvNetwork(self._ref_env.observation_space.shape, self._ref_env.action_space.n).cuda()
        self._optimizer = Adam(params=self._model.parameters(), lr=lr, eps=1e-3)

    # Override the method making sure to turn every observation into an ndarray in case tricks were
    # employed to save space.
    @staticmethod
    def _unpack(transitions, model, gamma, unfolding_steps):
        t_acts = LongTensor([t.action for t in transitions]).cuda()
        t_old_states = FloatTensor([np.array(t.previous_state) for t in transitions]).cuda()
        t_new_states = FloatTensor([np.array(t.next_state) for t in transitions]).cuda()
        t_done = LongTensor([t.done for t in transitions]).cuda()
        t_reward = FloatTensor([t.reward for t in transitions]).cuda()

        # Calculate the Q values
        t_next_states_values_predictions = model(t_new_states)[1].view(-1)*(1-t_done)
        t_qvals = t_reward + (gamma**unfolding_steps)*t_next_states_values_predictions

        return t_old_states, t_acts, t_qvals.detach()

    @classmethod
    def get_environment(cls):
        return ptan.common.wrappers.wrap_dqn(gym.make("PongNoFrameskip-v4"))