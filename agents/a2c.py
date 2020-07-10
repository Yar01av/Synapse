from collections import deque

from tensorboardX import SummaryWriter
from torch import cuda, nn, load, save, LongTensor, FloatTensor
from torch.optim import Adam
from action_selectors import ProbValuePolicySelector, BaseActionSelector
from agents.agent_training import AgentTraining
from memory import CompositeMemory
from steps_generators import SimpleStepsGenerator
import torch.nn.functional as F


class SimpleA2CNetwork(nn.Module):
    def __init__(self, state_size, n_actions):
        super().__init__()

        self._base = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU()
        )

        self._value_head = nn.Sequential(
            nn.Linear(128, 1)
        )

        self._policy_head = nn.Sequential(
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        base_output = self._base(x)
        return self._policy_head(base_output), self._value_head(base_output)


class A2C(AgentTraining):
    def __init__(self, gamma=0.99, beta=0.01, lr=0.01, batch_size=10, max_training_steps=1000, desired_avg_reward=500):
        super().__init__()

        self._desired_avg_reward = desired_avg_reward
        self._beta = beta
        self._n_training_steps = max_training_steps
        self._batch_size = batch_size
        self._lr = lr
        self._gamma = gamma
        cuda.set_device(0)
        self._model = SimpleA2CNetwork(self._env.observation_space.shape[0], self._env.action_space.n).cuda()
        self._optimizer = Adam(params=self._model.parameters(), lr=lr)
        self._memory = CompositeMemory()

        # Logging related
        self._plotter = SummaryWriter(comment="xREINFORCE")

    def train(self, save_path):
        steps_generator = SimpleStepsGenerator(self._env,
                                               ProbValuePolicySelector(self._env.action_space.n, model=self._model))
        batch_count = 0
        last_episodes_rewards = deque(maxlen=100)
        reward_sum = 0
        episode_idx = 0

        for idx, transition in enumerate(steps_generator):
            if idx == self._n_training_steps-1 or self._desired_avg_reward <= sum(last_episodes_rewards)/100:
                save(self._model, save_path)
                break

            reward_sum += transition.reward
            self._memory.states.append(transition.previous_state)
            self._memory.actions.append(transition.action)
            self._memory.rewards.append(transition.reward)

            if transition.done:
                batch_count += 1
                episode_idx += 1

                self._memory.compute_qvals(self._gamma)
                last_episodes_rewards.append(reward_sum)

                self._plotter.add_scalar("Total reward per episode", reward_sum, episode_idx)
                print(f"At step {idx}, \t the average over the last 100 games is {sum(last_episodes_rewards)/100}")

                reward_sum = 0

            if batch_count == self._batch_size:
                t_act = LongTensor(self._memory.actions).cuda()
                t_state = FloatTensor(self._memory.states).cuda()
                t_qval = FloatTensor(self._memory.q_vals).cuda()

                self._optimizer.zero_grad()
                t_logits = self._model(t_state)[0]

                # Compute the policy loss
                t_values = self._model(t_state)[1].detach()
                t_advantage = t_qval - t_values
                policy_loss = -(t_advantage*t_logits.log_softmax(dim=1)[range(len(t_state)), t_act]).mean()
                # Compute the value loss
                value_loss = F.mse_loss(t_values, t_qval)
                # Compute the entropy and record the original probabilities for later
                entropy = -(t_logits.softmax(dim=1)*t_logits.log_softmax(dim=1)).sum(dim=1).mean()
                old_probs = t_logits.softmax(dim=1)

                (policy_loss+value_loss-self._beta*entropy).backward()
                self._optimizer.step()

                # Compute KL divergence
                new_probs = self._model(t_state)[0].softmax(dim=1)
                kl_divergence = -((new_probs/old_probs).log()*old_probs).sum(dim=1).mean()

                batch_count = 0
                self._memory.reset()

                # Plot
                self._plotter.add_scalar("Entropy", entropy.item(), episode_idx)
                self._plotter.add_scalar("KL Divergence", kl_divergence.item(), episode_idx)

        self._plotter.close()

    @classmethod
    def load_selector(cls, load_path) -> BaseActionSelector:
        return ProbValuePolicySelector(action_space_size=cls.get_environment().action_space.n, model=load(load_path))