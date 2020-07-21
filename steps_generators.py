from abc import ABC, abstractmethod
from collections import namedtuple
from action_selectors import BaseActionSelector

Transition = namedtuple("Transition", ["state", "action", "reward"])
CompleteTransition = namedtuple("CompleteTransition", ["previous_state", "next_state", "action", "reward", "done"])
CompressedTransition = namedtuple("CompressedTransition", ["previous_state", "next_state", "action", "reward", "done",
                                                           "sub_rewards"])


class BaseStepsGenerator(ABC):
    """
    Base class for generators returning experiences of playing with the environment according to the semantics of MDP.
    """

    @abstractmethod
    def __init__(self, env, action_selector: BaseActionSelector):
        self._action_selector = action_selector
        self._env = env

    @abstractmethod
    def __iter__(self):
        pass


class SimpleStepsGenerator(BaseStepsGenerator):
    """
    A regular steps generator that outputs experiences of playing the environment.
    """

    def __init__(self, env, action_selector: BaseActionSelector):
        super().__init__(env, action_selector)

    def __iter__(self):
        step_idx = 0
        former_state = self._env.reset()
        reward_sum = 0
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
                print("The episode is finished!")
                print(f"At episode {e}, \t the reward is {reward_sum}")
                reward_sum = 0
                former_state = self._env.reset()

            step_idx += 1


class CompressedStepsGenerator(BaseStepsGenerator):
    """
    A generalized steps generator that is equivalent to SimpleStepsGenerator for n_steps=1. In other cases, it unfolds
    Bellman-Ford equation (discounting the subsequent rewards).
    """

    def __init__(self, env, action_selector: BaseActionSelector, n_steps=1, gamma=1):
        assert n_steps >= 1
        super().__init__(env, action_selector)
        self._gamma = gamma
        self._n_steps = n_steps

    def __iter__(self):
        steps = list()
        step_idx = 0
        former_state = self._env.reset()
        reward_sum = 0
        e = 0

        while True:
            action = self._action_selector.pick(former_state)
            next_state, reward, done, _ = self._env.step(action)
            steps.append(Transition(state=former_state, action=action, reward=reward))

            former_state = next_state
            reward_sum += reward

            if done:
                e += 1
                print("The episode is finished!")
                print(f"At episode {e}, \t the reward is {reward_sum}")
                reward_sum = 0
                former_state = self._env.reset()

            if len(steps) == self._n_steps or done:
                discounted_reward = 0

                # Discount the rewards
                for order in range(len(steps)):
                    discounted_reward += steps[order].reward*(self._gamma**order)

                yield CompressedTransition(previous_state=steps[0].state,
                                           next_state=next_state,
                                           action=steps[0].action,
                                           reward=discounted_reward,
                                           done=done,
                                           sub_rewards=[trans.reward for trans in steps])

                steps.clear()

            step_idx += 1
