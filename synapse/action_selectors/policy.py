from abc import abstractmethod

import numpy as np
from torch import no_grad, FloatTensor

from .base import ActionsSelector, ActionSelector
from ..util import select_index_from_probs


class PolicyActionSelector(ActionSelector):
    """
    Action selector for models trained to return probabilities (as logits). Combines the implementation and the ABC.
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


class PolicyActionsSelector(ActionsSelector):
    """
    Actions selector for models trained to return probabilities (as logits) for an array of observations.
    Combines the implementation and the ABC.
    """

    def __init__(self, model, model_device="cuda"):
        self._model = model
        self._model_device = model_device

    @no_grad()
    def pick(self, states):
        probs = self._model(FloatTensor(np.array([np.array(s, copy=False) for s in states], copy=False))
                            .to(self._model_device)) \
            .softmax(dim=-1)

        return select_index_from_probs(probs.cpu().numpy())
