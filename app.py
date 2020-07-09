from random import seed

import gym
import torch
from tensorflow import set_random_seed

from action_selectors import RandomDiscreteSelector
from agents.REINFORCE import REINFORCE


N_EPISODES = 10
MAX_EPISODE_LENGTH = 500


# Seed to make sure that the results are reproducible
seed(1)
set_random_seed(123)
torch.manual_seed(0)

# Uncomment for a proper agent
training_class = REINFORCE
training_instance = training_class(max_training_steps=1500000)
env = training_class.get_environment()

training_instance.train("checkpoint.h5")
selector = training_class.load_selector(load_path="checkpoint.h5")

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
