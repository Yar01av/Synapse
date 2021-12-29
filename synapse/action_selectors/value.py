from abc import abstractmethod

import numpy as np
from torch import FloatTensor
import random

from .base import ActionSelector


class GreedyActionSelector(ActionSelector):
    def __init__(self, model, model_device="cuda"):
        self._model_device = model_device
        self._model = model

    def pick(self, state):
        return self._model(FloatTensor(np.array(state)).to(self._model_device).unsqueeze(0)).argmax(dim=-1).item()
