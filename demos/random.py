# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import gym
import numpy as np
import torch
from torch import cuda

import random
from random import seed
from synapse.action_selectors.other import RandomDiscreteSelector
from synapse.demo import render_local_play

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
