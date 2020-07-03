import gym
from agents.REINFORCE import REINFORCE


N_EPISODES = 5
MAX_EPISODE_LENGTH = 500
training = REINFORCE()
env = training.get_environment()

training.train("checkpoint.h5")
selector = training.load_selector("checkpoint.h5")

### The game loop ###
current_state = env.reset()

for _ in range(N_EPISODES):
    for step_idx in range(MAX_EPISODE_LENGTH):
        env.render()
        next_state, reward, done, _ = env.step(selector.pick(current_state))
        print(f"At step {step_idx}, the reward is {reward}")

        if done or step_idx == MAX_EPISODE_LENGTH-1:
            env.render()
            break
