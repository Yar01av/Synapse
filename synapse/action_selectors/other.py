import random

from .base import ActionSelector


class RandomDiscreteSelector(ActionSelector):
    """
    Action selector that picks actions at random. It assumes that every action is executable
    (even if it doesn't do anything).
    """

    def __init__(self, n_actions):
        self._n_actions = n_actions

    def pick(self, state, is_batch=False):
        # TODO: implement vectorized random action selection
        if is_batch:
            raise NotImplementedError()

        return random.choice(range(self._n_actions))