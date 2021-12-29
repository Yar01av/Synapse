import time

import gym

from synapse.action_selectors.base import ActionSelector


def render_local_play(env: gym.Env, selector: ActionSelector, n_episodes=30, max_episode_length=500):
    for episode_idx in range(n_episodes):
        current_state = env.reset()
        total_reward = 0

        for step_idx in range(max_episode_length):
            env.render()
            time.sleep(.016666)
            action = selector.pick(current_state)
            current_state, reward, done, _ = env.step(action)
            total_reward += reward

            if done or step_idx == max_episode_length - 1:
                env.render()
                break

        print(f"At episode {episode_idx}, \t the reward is {total_reward}")