from pathlib import Path
from pathlib import Path

# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import gym
import numpy as np
import torch
from torch import cuda

import random
from random import seed
from synapse.action_selectors.policy import PolicyActionSelector
from synapse.agents.a3c.distributed_envs import A3C
from synapse.demo import render_local_play
from synapse.models import A3CNetwork

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
    model = A3CNetwork(env.observation_space.shape[0], env.action_space.n)

    training = A3C(env, model=model, max_training_steps=10000)
    training.train(Path("../checkpoints/checkpoint.h5"))

    selector = PolicyActionSelector(lambda obs: training.model(obs)[0], model_device="cpu")

    render_local_play(env, selector)
