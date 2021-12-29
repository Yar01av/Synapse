from abc import ABC, abstractmethod


class ActionSelector(ABC):
    """
    Base class for all the algorithms used to pick actions given the state.
    They are responsible for interpreting the output of the model (if applicable) and picking an action. They cannot
    rely on any particular form of observation produced by the environment.
    """

    @abstractmethod
    def pick(self, state):
        """
        Returns a valid action.

        :param state: State in which the action is required
        :return:
        """

        pass


class ActionsSelector(ABC):
    @abstractmethod
    def pick(self, states):
        pass



