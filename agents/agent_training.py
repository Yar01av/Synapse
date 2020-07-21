from abc import ABC, abstractmethod

import gym
from action_selectors import BaseActionSelector


class AgentTraining(ABC):
    @abstractmethod
    def __init__(self):
        self._env = self.get_environment()

    @abstractmethod
    def train(self, save_path):
        pass

    @classmethod
    @abstractmethod
    def load_selector(cls, load_path) -> BaseActionSelector:
        pass

    @classmethod
    def get_environment(cls):
        return gym.make("CartPole-v1")