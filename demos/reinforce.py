from pathlib import Path

import gym
import numpy as np
import torch
from torch import cuda

import random
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from random import seed
from synapse.action_selectors.policy import PolicyActionSelector
from synapse.agents.reinforce.standard import REINFORCE
from synapse.demo import render_local_play
from synapse.models import REINFORCENetwork

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
    env.seed(0)
    model = REINFORCENetwork(env.observation_space.shape[0], env.action_space.n).cuda()

    training = REINFORCE(env, model=model, max_training_steps=10000, device="cuda")
    training.train(Path("../checkpoints/checkpoint.h5"))

    selector = PolicyActionSelector(training.model, model_device="cuda")

    render_local_play(env, selector)
