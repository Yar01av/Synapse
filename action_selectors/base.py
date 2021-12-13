from abc import ABC, abstractmethod


class ActionSelector(ABC):
    """
    Base class for all the algorithms used to pick actions given the state.
    They are responsible for interpreting the output of the model (if applicable).
    """

    @abstractmethod
    def pick(self, state):
        """
        Returns a valid action.

        :param state: State in which the action is required
        :return:
        """

        pass


class VectorActionSelector(ABC):
    @abstractmethod
    def pick(self, states):
        pass



