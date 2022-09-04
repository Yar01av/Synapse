import random

import numpy as np

from .base import ActionSelector


class RandomDiscreteSelector(ActionSelector):
    """
    Action selector that picks actions at random. It assumes that every action is executable
    (even if it doesn't do anything).
    """

    def __init__(self, n_actions):
        self._n_actions = n_actions

    def pick(self, state):
        return random.choice(range(self._n_actions))


class EpsilonActionSelector(ActionSelector):
    def __init__(self, selector: ActionSelector, n_actions, init_epsilon, min_epsilon, epsilon_decay):
        self._selector = selector
        self._n_actions = n_actions
        self._min_epsilon = min_epsilon
        self._epsilon_decay = epsilon_decay
        self._epsilon = init_epsilon

    def pick(self, state):
        if np.random.rand() <= self._epsilon:
            return random.randrange(self._n_actions)
        else:
            return self._selector.pick(state)

    def decay_epsilon(self):
        if self._epsilon > self._min_epsilon:
            self._epsilon *= self._epsilon_decay

    @property
    def epsilon(self):
        return self._epsilon
