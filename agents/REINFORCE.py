import gym
from torch.optim import Adam
from action_selectors import BaseActionSelector, SimplePolicySelector
from agents.agent_training import AgentTraining
from torch import load, nn, cuda, save, LongTensor, FloatTensor
import numpy as np
from memory import CompositeMemory
from steps_generators import SimpleStepsGenerator


class REINFORCE(AgentTraining):
    def __init__(self, gamma, beta, lr, batch_size, n_training_steps):
        super().__init__()

        self._beta = beta
        self._n_training_steps = n_training_steps
        self._batch_size = batch_size
        self._lr = lr
        self._gamma = gamma
        cuda.set_device(0)
        self._model = nn.Sequential(nn.Linear(self._env.observation_space.shape, 128),
                                    nn.ReLU(),
                                    nn.Linear(128, self._env.action_space.n)).cuda()
        self._optimizer = Adam(params=self._model.parameters(), lr=lr)
        self._memory = CompositeMemory()

    def train(self, save_path):
        steps_generator = SimpleStepsGenerator(self._env, SimplePolicySelector(self._env.action_space.n, model=self._model))
        batch_count = 0

        for idx, transition in enumerate(steps_generator):
            print(f"Starting training step number {idx}")

            if idx == self._n_training_steps-1:
                save(self._model, save_path)
                break

            self._memory.states.append(transition.previous_state)
            self._memory.actions.append(transition.action)
            self._memory.rewards.append(transition.reward)

            if (idx+1)%self._batch_size == 0:
                t_act = LongTensor(self._memory.actions).cuda()
                t_state = FloatTensor(self._memory.states).cuda()
                t_qval = FloatTensor(self._memory.q_vals).cuda()

                self._optimizer.zero_grad()
                t_logits = self._model(t_state)

                # Compute the policy loss
                policy_loss = -(t_qval*t_logits.log_softmax(dim=1)[range(len(t_state)), t_act]).mean()
                # Compute the entropy
                entropy = -(t_logits.softmax(dim=1)*t_logits.log_softmax(dim=1)).sum(dim=1).mean()

                (policy_loss-self._beta*entropy).backward()
                self._optimizer.step()

                self._memory.reset()
            else:
                self._memory.states.append(transition.previous_state)

            if transition.done:
                self._memory.compute_qvals(self._gamma)
                batch_count=batch_count+1
