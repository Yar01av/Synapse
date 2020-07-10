from abc import ABC, abstractmethod
from collections import namedtuple
from selectors.action_selectors import BaseActionSelector

Transition = namedtuple("Transition", ["state", "action", "reward", "done"])
CompleteTransition = namedtuple("CompleteTransition", ["previous_state", "next_state", "action", "reward", "done"])


class BaseStepsGenerator(ABC):
    @abstractmethod
    def __init__(self, env, action_selector: BaseActionSelector):
        self._action_selector = action_selector
        self._env = env

    @abstractmethod
    def __iter__(self):
        pass


class SimpleStepsGenerator(BaseStepsGenerator):
    def __init__(self, env, action_selector: BaseActionSelector):
        super().__init__(env, action_selector)

    def __iter__(self):
        step_idx = 0
        former_state = self._env.reset()
        reward_sum = 0
        inter_episode_reward_sum = 0
        e = 0

        while True:
            action = self._action_selector.pick(former_state)
            next_state, reward, done, _ = self._env.step(action)
            yield CompleteTransition(previous_state=former_state, next_state=next_state, action=action, reward=reward,
                                     done=done)

            former_state = next_state
            reward_sum += reward

            if done:
                e += 1
                inter_episode_reward_sum += reward_sum
                print("The episode is finished!")
                print(f"At episode {e}, \t the reward is {reward_sum}")
                reward_sum = 0
                former_state = self._env.reset()

            step_idx += 1
