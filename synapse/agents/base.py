from abc import ABC, abstractmethod

import gym
from synapse.action_selectors.base import ActionSelector


class DiscreteAgentTraining(ABC):
    """
    Base class for classes implementing various training algorithms.
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def train(self, save_path):
        pass

    @property
    @abstractmethod
    def model(self):
        pass
