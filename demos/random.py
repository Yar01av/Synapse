import os
from pathlib import Path
from random import seed

# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import gym
import torch
import numpy as np
from torch import cuda

from synapse.action_selectors.other import RandomDiscreteSelector
from synapse.action_selectors.policy import PolicyActionSelector
from synapse.agents.reinforce.standard import REINFORCE
from synapse.demo import render_local_play
from synapse.models import DQNNetwork, REINFORCENetwork
from synapse.action_selectors.value import GreedyActionSelector
from synapse.agents.dqn.standard import DQN
import random


# Seed to make sure that the results are reproducible
seed(0)
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)
cuda.set_device(0)

torch.set_printoptions(7)


N_EPISODES = 30
MAX_EPISODE_LENGTH = 500


if __name__ == "__main__":
    env = gym.make("CartPole-v1")

    render_local_play(env, RandomDiscreteSelector(env.action_space.n))
