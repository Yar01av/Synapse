from abc import ABC, abstractmethod
from collections import namedtuple
from action_selectors import BaseActionSelector

Transition = namedtuple("Transition", ["state", "action", "reward", "done"])
CompleteTransition = namedtuple("CompleteTransition", ["previous_state", "next_state", "action", "reward", "done"])


class BaseStepsGenerator(ABC):
    @abstractmethod
    def __init__(self, env, action_selector: BaseActionSelector):
        self.action_selector = action_selector
        self.env = env

    @abstractmethod
    def __iter__(self):
        pass


class SimpleStepsGenerator(BaseStepsGenerator):
    def __init__(self, env, action_selector: BaseActionSelector):
        super().__init__(env, action_selector)

    def __iter__(self):
        pass