from abc import ABC, abstractmethod
import random

import numpy as np
from torch import FloatTensor, no_grad, argmax

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


class ModelBasedActionSelector(BaseActionSelector, ABC):
    @abstractmethod
    def __init__(self, model, model_device="cuda"):
        """
        :param model: The model to be used. It can also be passed inside a function
        :param model_device: The device on which the model itself resides. That is, for model(x), is x on cpu or gpu.
        It is the user's responsibility to make sure they align.
        """

        super().__init__()
        self._model = model
        self._model_device = model_device


class SimplePolicySelector(ModelBasedActionSelector):
    """
    Action selector for models trained to return probabilities (as logits).
    """

    def __init__(self, model, model_device="cuda"):
        super().__init__(model, model_device)

    @no_grad()
    def pick(self, state, is_batch=False):
        if is_batch:
            probs = self._model(FloatTensor(np.array([np.array(s, copy=False) for s in state], copy=False))
                                .to(self._model_device))\
                                .softmax(dim=-1)
            return select_index_from_probs(probs.cpu().numpy())
        else:
            probs = self._model(FloatTensor(np.array(state)).to(self._model_device).unsqueeze(0))\
                        .softmax(dim=-1)\
                        .cpu().numpy()
            probs = np.squeeze(probs)
            return np.random.choice(range(len(probs)), p=probs)


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


class ValueSelector(ModelBasedActionSelector):
    def __init__(self, model, model_device="cuda"):
        super().__init__(model, model_device)

    def pick(self, state, is_batch=False):
        if is_batch:
            return self._model(FloatTensor(np.array([np.array(s, copy=False) for s in state], copy=False))
                       .to(self._model_device)) \
                       .argmax(dim=-1)
        else:
            return self._model(FloatTensor(np.array(state)).to(self._model_device).unsqueeze(0)).argmax(dim=-1)
