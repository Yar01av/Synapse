from random import seed

# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch
#from tensorflow import set_random_seed

from agents.a2c.a2c import A2C
from agents.a3c.a3c_dist_env import A3C
from agents.reinforce.reinforce import REINFORCE

N_EPISODES = 10
MAX_EPISODE_LENGTH = 500


# Seed to make sure that the results are reproducible
seed(1)
#set_random_seed(123)
torch.manual_seed(0)

# Uncomment for a proper agent
training = A3C(max_training_steps=10000000,
                   envs_per_thread=50,
                   unfolding_steps=4,
                   desired_avg_reward=200,
                   lr=0.001,
                   batch_size=128)
#training = REINFORCE(max_training_steps=200000, desired_avg_reward=180)
env = training.get_environment()

if __name__ == "__main__":
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
