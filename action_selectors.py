from abc import ABC, abstractmethod
import random

import numpy as np
from torch import FloatTensor, no_grad


class BaseActionSelector(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def pick(self, state):
        """
        Returns a valid action.

        :param state: State in which the action is required
        :return:
        """

        pass


class SimplePolicySelector(BaseActionSelector):
    def __init__(self, action_space_size, model):
        super().__init__()
        self._model = model
        self._action_space_size = action_space_size

    @no_grad()
    def pick(self, state):
        probs = self._model(FloatTensor(state).cuda()).softmax(dim=1)

        return np.random.choice(range(self._action_space_size), p=np.squeeze(probs.cpu().detach().numpy()))


class RandomDiscreteSelector(BaseActionSelector):
    def __init__(self, n_actions):
        super().__init__()
        self._n_actions = n_actions

    def pick(self, state):
        return random.choice(range(self._n_actions))