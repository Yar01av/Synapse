from random import seed

# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch
#from tensorflow import set_random_seed
import numpy as np
from agents.a2c.standard import A2C
from agents.a3c.distributed_envs import A3C
from agents.dqn.standard import DQN
from agents.reinforce.standard import REINFORCE
import random


# Seed to make sure that the results are reproducible
seed(1)
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

torch.set_printoptions(7)


N_EPISODES = 10
MAX_EPISODE_LENGTH = 500


if __name__ == "__main__":
    # Uncomment for a proper agent
    training = A3C()
    env = training.get_environment()

    training.train("checkpoint2.h5")
    selector = training.load_selector(load_path="checkpoint2.h5")

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
