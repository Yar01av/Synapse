import numpy as np
from torch import FloatTensor
import random
from action_selectors.base import ModelBasedActionSelector


class GreedySelector(ModelBasedActionSelector):
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


class EpsilonGreedySelector(GreedySelector):
    def __init__(self, model, n_actions, model_device, init_epsilon, min_epsilon, epsilon_decay):
        super().__init__(model, model_device)

        self._n_actions = n_actions
        self._min_epsilon = min_epsilon
        self._epsilon_decay = epsilon_decay
        self._epsilon = init_epsilon

    def pick(self, state, is_batch=False):
        if np.random.rand() <= self._epsilon:
            return random.randrange(self._n_actions) if not is_batch \
                else np.random.randint(0, self._n_actions, size=len(state))
        else:
            return super().pick(state, is_batch)

    def decay_epsilon(self):
        if self._epsilon > self._min_epsilon:
            self._epsilon *= self._epsilon_decay
