class CompositeMemory:
    """
    A memory that can easily compute the actual Q-values for the experiences stored there.
    """

    def __init__(self):
        self.actions = []
        self.states = []

        self.q_vals = []
        self.rewards = []

    # Compute Q-values for the episode. Make sure to call the method every time a final state is reached.
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
