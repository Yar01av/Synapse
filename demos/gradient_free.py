from pathlib import Path

# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import gym
import numpy as np
import torch
from torch import cuda

import random
from gene.selections.top_n import TopNSelection
from synapse.action_selectors.value import GreedyActionSelector
from synapse.agents.dqn.genetic_gradient import GeneticGradientDQN
from synapse.agents.dqn.standard import GradientDQN
from synapse.agents.grad_free.genetic import GeneticGradientFree
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

    training = GeneticGradientFree(
        env,
        training_iterations=2_000,
        loss_estimation_steps=500,
        models=[model],
        device=DEVICE,
        logdir=Path("../runs"),
        selection=TopNSelection(10)
    )
    training.train(Path("../checkpoints/checkpoint.h5"))

    selector = GreedyActionSelector(training.model, model_device=DEVICE)

    render_local_play(env, selector)
