from abc import ABC, abstractmethod
import random

import numpy as np
from torch import FloatTensor, no_grad

from util import select_index_from_probs


class BaseActionSelector(ABC):
    """
    Base class for all the algorithms used to pick actions.
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def pick(self, state, is_batch=False):
        """
        Returns a valid action.

        :param is_batch: Does state contain a batch of states?
        :param state: State in which the action is required
        :return:
        """

        pass


class SimplePolicySelector(BaseActionSelector):
    """
    Action selector for models trained to return probabilities (as logits).
    """

    def __init__(self, action_space_size, model):
        super().__init__()
        self._model = model
        self._action_space_size = action_space_size

    @no_grad()
    def pick(self, state, is_batch=False):
        probs = self._model(FloatTensor(state).cuda()).softmax(dim=0)

        if is_batch:
            return select_index_from_probs(probs.item())
        else:
            return np.random.choice(range(self._action_space_size), p=np.squeeze(probs.cpu().detach().numpy()))


class RandomDiscreteSelector(BaseActionSelector):
    """
    Action selector that picks actions at random. It assumes that every action is executable
    (even if it doesn't do anything).
    """

    def __init__(self, n_actions):
        super().__init__()
        self._n_actions = n_actions

    def pick(self, state, is_batch=False):
        # TODO: implement vectorized random action selection
        if is_batch:
            raise NotImplementedError()

        return random.choice(range(self._n_actions))


class ProbValuePolicySelector(BaseActionSelector):
    """
    Similar to SimplePolicySelector, but assumes that the model's second head returns the action probabilities
    (as logits).
    """

    def __init__(self, action_space_size, model):
        super().__init__()
        self._model = model
        self._action_space_size = action_space_size

    @no_grad()
    def pick(self, state, is_batch=False):
        if is_batch:
            probs = self._model(FloatTensor(np.array([np.array(s, copy=False) for s in state], copy=False))
                        .cuda())[0]\
                        .softmax(dim=-1)
            return select_index_from_probs(probs.cpu().numpy())
        else:
            probs = self._model(FloatTensor(np.array(state)).cuda().unsqueeze(0))[0]\
                        .softmax(dim=-1)
            return np.random.choice(range(self._action_space_size), p=np.squeeze(probs.cpu().detach().numpy()))