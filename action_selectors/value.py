from abc import abstractmethod

import numpy as np
from torch import FloatTensor
import random

from action_selectors.base import ActionSelector


class BaseValueActionSelector(ActionSelector):
    @abstractmethod
    def pick(self, state):
        pass


class GreedySelector(BaseValueActionSelector):
    def __init__(self, model, model_device="cuda"):
        self._model_device = model_device
        self._model = model

    def pick(self, state):
        return self._model(FloatTensor(np.array(state)).to(self._model_device).unsqueeze(0)).argmax(dim=-1).item()


class EpsilonGreedySelector(GreedySelector):
    def __init__(self, model, n_actions, model_device, init_epsilon, min_epsilon, epsilon_decay):
        super().__init__(model, model_device)

        self._n_actions = n_actions
        self._min_epsilon = min_epsilon
        self._epsilon_decay = epsilon_decay
        self._epsilon = init_epsilon

    def pick(self, state):
        if np.random.rand() <= self._epsilon:
            return random.randrange(self._n_actions)
        else:
            return super().pick(state)

    def decay_epsilon(self):
        if self._epsilon > self._min_epsilon:
            self._epsilon *= self._epsilon_decay
