from action_selectors import BaseActionSelector
from agents.agent_training import AgentTraining


class REINFORCE(AgentTraining):
    def __init__(self):
        super().__init__()

    def train(self, save_path):
        pass

    def load_selector(self, load_path) -> BaseActionSelector:
        pass

    def get_environment(self):
        pass
