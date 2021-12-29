from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Callable

from .action_selectors.base import ActionSelector, ActionsSelector

Transition = namedtuple("Transition", ["state", "action", "reward"])
CompleteTransition = namedtuple("CompleteTransition", ["previous_state", "next_state", "action", "reward", "done"])
CompressedTransition = namedtuple("CompressedTransition", ["previous_state", "next_state", "action", "reward", "done",
                                                           "sub_rewards"])


class BaseStepsGenerator(ABC):
    """
    Base class for generators returning experiences of playing with the environment according to the semantics of MDP.
    """

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def pop_per_episode_rewards(self):
        pass


class SingletonEnvStepsGenerator(BaseStepsGenerator, ABC):
    @abstractmethod
    def __init__(self, env, action_selector: ActionSelector):
        self._action_selector = action_selector
        self._env = env


class SimpleStepsGenerator(SingletonEnvStepsGenerator):
    """
    A regular steps generator that outputs experiences of playing the environment.
    """

    def __init__(self, env, action_selector: ActionSelector):
        super().__init__(env, action_selector)
        self._per_episode_rewards = list()

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
                self._per_episode_rewards.append(reward_sum)

                reward_sum = 0
                former_state = self._env.reset()

            step_idx += 1

    def pop_per_episode_rewards(self):
        output = self._per_episode_rewards.copy()
        self._per_episode_rewards.clear()

        return output


class CompressedStepsGenerator(SingletonEnvStepsGenerator):
    """
    A generalized steps generator that is equivalent to SimpleStepsGenerator for n_steps=1. In other cases, it unfolds
    Bellman-Ford equation (discounting the subsequent rewards).
    """

    def __init__(self, env, action_selector: ActionSelector, n_steps=1, gamma=1):
        assert n_steps >= 1
        super().__init__(env, action_selector)
        self._gamma = gamma
        self._n_steps = n_steps
        self._per_episode_rewards = list()

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
                self._per_episode_rewards.append(reward_sum)
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

    def pop_per_episode_rewards(self):
        output = self._per_episode_rewards.copy()
        self._per_episode_rewards.clear()

        return output


class MultiEnvCompressedStepsGenerator(BaseStepsGenerator):
    """
    A generalized steps generator that is equivalent to CompressedStepsGenerator for n_envs=1. It generates the experiences
    by maintaining several environments and sampling them in a round robin fashion.
    """

    def __init__(self, envs, action_selector: ActionsSelector, n_steps=1, gamma=1):
        assert n_steps >= 1

        self._action_selector = action_selector
        self._envs = envs
        self._n_envs = len(envs)
        self._gamma = gamma
        self._n_steps = n_steps
        self._per_episode_rewards = list()

    def __iter__(self):
        steps = [list() for i in range(self._n_envs)]
        step_idx = 0
        former_states = [env.reset() for env in self._envs]
        reward_sums = [0]*self._n_envs
        episode_idxs = [0]*self._n_envs
        discounted_rewards = [0]*self._n_envs
        actions = []

        while True:
            env_idx = step_idx%self._n_envs

            if env_idx == 0:
                actions = self._action_selector.pick(former_states)

            next_state, reward, done, _ = self._envs[env_idx].step(actions[env_idx])
            steps[env_idx].append(Transition(state=former_states[env_idx], action=actions[env_idx], reward=reward))

            former_states[env_idx] = next_state
            reward_sums[env_idx] += reward

            if done:
                episode_idxs[env_idx] += 1
                print("The episode is finished!")
                print(f"At episode {episode_idxs[env_idx]} of environment number {env_idx}, \t the reward is {reward_sums[env_idx]}")
                self._per_episode_rewards.append(reward_sums[env_idx])
                reward_sums[env_idx] = 0
                former_states[env_idx] = self._envs[env_idx].reset()

            if len(steps[env_idx]) == self._n_steps or done:
                discounted_rewards[env_idx] = 0

                # Discount the rewards
                for order in range(len(steps[env_idx])):
                    discounted_rewards[env_idx] += steps[env_idx][order].reward*(self._gamma**order)

                yield CompressedTransition(previous_state=steps[env_idx][0].state,
                                           next_state=next_state,
                                           action=steps[env_idx][0].action,
                                           reward=discounted_rewards[env_idx],
                                           done=done,
                                           sub_rewards=[trans.reward for trans in steps[env_idx]])

                steps[env_idx].clear()

            step_idx += 1

    def pop_per_episode_rewards(self):
        output = self._per_episode_rewards.copy()
        self._per_episode_rewards.clear()

        return output