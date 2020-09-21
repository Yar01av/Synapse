import numpy as np
from torch import FloatTensor

from action_selectors.base import ModelBasedActionSelector


class EpsilonGreedySelector(ModelBasedActionSelector):
    def __init__(self, model, model_device="cuda"):
        super().__init__(model, model_device)

    def pick(self, state, is_batch=False):
        if is_batch:
            return self._model(FloatTensor(np.array([np.array(s, copy=False) for s in state], copy=False))
                       .to(self._model_device)) \
                       .argmax(dim=-1) \
                       .numpy()
        else:
            return self._model(FloatTensor(np.array(state)).to(self._model_device).unsqueeze(0)).argmax(dim=-1).item()