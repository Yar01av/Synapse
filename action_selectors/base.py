from abc import ABC, abstractmethod


class BaseActionSelector(ABC):
    """
    Base class for all the algorithms used to pick actions.
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def pick(self, state, is_batch=False):
        """
        Returns a valid action.

        :param is_batch: Does state contain a batch of states?
        :param state: State in which the action is required
        :return:
        """

        pass


class ModelBasedActionSelector(BaseActionSelector, ABC):
    @abstractmethod
    def __init__(self, model, model_device="cuda"):
        """
        :param model: The model to be used. It can also be passed inside a function
        :param model_device: The device on which the model itself resides. That is, for model(x), is x on cpu or gpu.
        It is the user's responsibility to make sure they align.
        """

        super().__init__()
        self._model = model
        self._model_device = model_device


