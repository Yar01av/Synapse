import gym

from action_selectors import RandomDiscreteSelector
from agents.REINFORCE import REINFORCE


N_EPISODES = 100
MAX_EPISODE_LENGTH = 500
# Uncomment for a proper agent
training_class = REINFORCE
training_instance = training_class()
env = training_class.get_environment()

training_instance.train("checkpoint.h5")
selector = training_class.load_selector("checkpoint.h5")

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
