from abc import ABC, abstractmethod

import gym
from torch import load

from action_selectors import BaseActionSelector


class AgentTraining(ABC):
    @abstractmethod
    def __init__(self):
        self._env = self.get_environment()

    @abstractmethod
    def train(self, save_path):
        pass

    @classmethod
    def load_selector(self, load_path) -> BaseActionSelector:
        return load(load_path)

    @classmethod
    def get_environment(self):
        return gym.make("CartPole-v1")