import numpy as np
from torch import no_grad, FloatTensor

from action_selectors.base import ModelBasedActionSelector
from util import select_index_from_probs


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
