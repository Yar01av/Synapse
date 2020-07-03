from abc import ABC, abstractmethod

from action_selectors import BaseActionSelector


class AgentTraining(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def train(self, save_path):
        pass

    @abstractmethod
    def load_selector(self, load_path) -> BaseActionSelector:
        pass

    @abstractmethod
    def get_environment(self):
        pass