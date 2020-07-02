from abc import ABC, abstractmethod


class BaseActionSelector(ABC):
    @abstractmethod
    def __init__(self):
        pass

    def pick(self, state):
        """
        Returns a valid action.

        :param state: State in which the action is required
        :return:
        """

        pass