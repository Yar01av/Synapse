from random import seed

# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch
import numpy as np
from synapse.agents.dqn.standard import DQN
import random


# Seed to make sure that the results are reproducible
seed(0)
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

torch.set_printoptions(7)


N_EPISODES = 30
MAX_EPISODE_LENGTH = 500


if __name__ == "__main__":
    # Uncomment for a proper agent
    training = DQN(max_training_steps=100000)
    env = training.get_environment()

    training.train("checkpoint.h5")
    selector = training.load_selector(load_path="checkpoint.h5")

    # Uncomment for a random baseline
    # env = gym.make("CartPole-v1")
    # selector = RandomDiscreteSelector(env.action_space.n)

    # The game loop
    for episode_idx in range(N_EPISODES):
        current_state = env.reset()
        sum = 0

        for step_idx in range(MAX_EPISODE_LENGTH):
            env.render()
            action = selector.pick(current_state)
            current_state, reward, done, _ = env.step(action)
            sum += reward

            if done or step_idx == MAX_EPISODE_LENGTH-1:
                env.render()
                break

        print(f"At episode {episode_idx}, \t the reward is {sum}")
