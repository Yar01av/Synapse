class CompositeMemory:
    def __init__(self):
        self.actions = []
        self.states = []

        self.q_vals = []
        self.rewards = []

    # Compute Q-values for the episode
    def compute_qvals(self, gamma):
        qs = []

        for reward in reversed(self.rewards):
            if len(qs) == 0:
                qs.append(reward)
            else:
                qs.append(reward + gamma*qs[-1])

        self.rewards.clear()

        self.q_vals.extend(list(reversed(qs)))

    def reset(self):
        self.actions.clear()
        self.states.clear()
        self.q_vals.clear()
        self.rewards.clear()