from pathlib import Path

# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import gym
import numpy as np
import torch
from torch import cuda

import random
from synapse.action_selectors.value import GreedyActionSelector
from synapse.agents.dqn.genetic import GeneticDQN
from synapse.agents.dqn.standard import GradientDQN
from synapse.demo import render_local_play
from synapse.models import DQNNetwork

# Seed to make sure that the results are reproducible
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)
cuda.set_device(0)

torch.set_printoptions(7)


N_EPISODES = 30
MAX_EPISODE_LENGTH = 500
DEVICE = "cpu"


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    env.seed(0)
    model = DQNNetwork(env.observation_space.shape[0], env.action_space.n).to(DEVICE)

    training = GeneticDQN(env, models=[model],
                          max_training_steps=20_000,
                          device=DEVICE,
                          logdir=Path("../runs"),
                          selection_limit=10)
    training.train(Path("../checkpoints/checkpoint.h5"))

    selector = GreedyActionSelector(training.model, model_device=DEVICE)

    render_local_play(env, selector)
