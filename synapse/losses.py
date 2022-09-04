import gym
from torch import tensor, float


def negative_score_loss(gym_env: gym.Env, n_steps: int = 1, device: str = "cuda"):
    """
    Returns a loss function that returns the negative score of the agent in the environment.
    :param gym_env: The environment to evaluate the agent in.
    :param n_steps: The number of steps to take in the environment.
    :param device: The device which the model expects its inputs to be on.
    :return: A loss function that returns the negative score of the agent in the environment.
    """
    def loss(model):
        score = 0
        obs = gym_env.reset()

        for _ in range(n_steps):
            action = model(tensor(obs, dtype=float, device=device)).argmax().item()
            obs, reward, done, _ = gym_env.step(action)
            score += reward

            if done:
                break

        return -score

    return loss
