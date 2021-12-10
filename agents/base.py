from abc import ABC, abstractmethod

import gym
from action_selectors.base import BaseActionSelector


class AgentTraining(ABC):
    """
    Base class for classes implementing various training algorithms.
    """

    @abstractmethod
    def __init__(self):
        pass

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
