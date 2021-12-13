from abc import abstractmethod

import numpy as np
from torch import no_grad, FloatTensor

from action_selectors.base import VectorActionSelector, ActionSelector
from util import select_index_from_probs


class PolicyActionSelector(ActionSelector):
    @abstractmethod
    def pick(self, state):
        pass


class LogitActionSelector(PolicyActionSelector):
    """
    Action selector for models trained to return probabilities (as logits).
    """

    def __init__(self, model, model_device="cuda"):
        self._model = model
        self._model_device = model_device

    @no_grad()
    def pick(self, state):
        probs = self._model(FloatTensor(np.array(state)).to(self._model_device).unsqueeze(0))\
                    .softmax(dim=-1)\
                    .cpu().numpy()
        probs = np.squeeze(probs)

        return np.random.choice(range(len(probs)), p=probs)


class VecLogitActionSelector(VectorActionSelector):
    def __init__(self, model, model_device="cuda"):
        self._model = model
        self._model_device = model_device

    @no_grad()
    def pick(self, states):
        probs = self._model(FloatTensor(np.array([np.array(s, copy=False) for s in states], copy=False))
                            .to(self._model_device)) \
            .softmax(dim=-1)

        return select_index_from_probs(probs.cpu().numpy())
