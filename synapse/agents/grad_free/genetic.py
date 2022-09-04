import random
from collections import deque
from copy import deepcopy
from pathlib import Path
import time
import gym
import numpy as np
from tensorboardX import SummaryWriter
import torch
from torch import cuda, nn, save, LongTensor, FloatTensor, IntTensor, squeeze, max
import torch.nn.functional as F

from gene.optimisers.division import DivisionOptimiser, ParallelDivisionOptimiser
from gene.selections.top_n import TopNSelection
from gene.targets import make_supervised_loss
from gene.util import get_accuracy
from synapse.action_selectors.other import EpsilonActionSelector
from synapse.steps_generators import CompressedStepsGenerator
from synapse.util import can_stop
from ..base import DiscreteAgentTraining
from ...action_selectors.value import GreedyActionSelector
from ...losses import negative_score_loss


class GeneticGradientFree(DiscreteAgentTraining):
    def __init__(self,
                 environment: gym.Env,
                 models,
                 training_iterations=100,
                 selection=TopNSelection(100),
                 loss_estimation_steps=100,
                 mutation_variance=0.1,
                 device="cuda",
                 logdir="./runs"):
        super().__init__()

        self._training_iterations = training_iterations
        self._environment = environment
        self._device = device
        cuda.set_device(0)
        self._models = models
        self._loss_estimation_steps = loss_estimation_steps
        self._mutation_variance = mutation_variance

        # Optimiser variables
        self._optimizer = DivisionOptimiser(random_function=lambda shape: torch.normal(0,
                                                                                       self._mutation_variance,
                                                                                       shape),
                                            selection=selection,
                                            device=device)

        # Logging related
        self._plotter = SummaryWriter(logdir=f"{logdir}/{time.strftime('%c')}--{self.__class__.__name__}")

    def train(self, save_path):
        for idx in range(self._training_iterations):
            loss = negative_score_loss(
                deepcopy(self._environment),
                n_steps=self._loss_estimation_steps,
                device=self._device
            )
            self._models = self._optimizer.step(
                models=self._models,
                loss_function=loss
            )
            self._log_performance(idx)

        self._plotter.close()

    def _log_performance(self, idx):
        negative_reward_for_model = negative_score_loss(deepcopy(self._environment), 500, self._device)
        average_reward = np.mean([-negative_reward_for_model(m) for m in self._models])
        self._plotter.add_scalar(
            "Average reward per model for one episode",
            average_reward,
            idx
        )
        print(f"Average reward per model for episode {idx}: {average_reward}")

    @property
    def model(self):
        return deepcopy(self._models[0])
