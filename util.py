import math

import numpy as np
import torch
from torch import LongTensor, FloatTensor


def select_index_from_probs(prob_matrix):
    s = prob_matrix.cumsum(axis=1)
    r = np.random.rand(prob_matrix.shape[0])
    k = (s < r.reshape((len(s), 1))).sum(axis=1)

    return k


def get_output_size(network, shape):
    out = network(torch.zeros(size=(1, *shape)))

    return torch.flatten(out, start_dim=1).shape[1]


def unpack(transitions, model, gamma, unfolding_steps, device="cuda"):
    t_acts = LongTensor([t.action for t in transitions]).to(device)
    t_old_states = FloatTensor([np.array(t.previous_state) for t in transitions]).to(device)
    t_new_states = FloatTensor([np.array(t.next_state) for t in transitions]).to(device)
    t_done = LongTensor([t.done for t in transitions]).to(device)
    t_reward = FloatTensor([t.reward for t in transitions]).to(device)

    # Calculate the Q values
    t_next_states_values_predictions = model(t_new_states)[1].view(-1)*(1-t_done)
    t_qvals = t_reward + (gamma**unfolding_steps)*t_next_states_values_predictions

    return t_old_states, t_acts, t_qvals.detach()


# Returns true if the training can stop due to the maximum number of steps
# being reached or the average reward reaching the desired level
def can_stop(current_step_idx, total_steps, latest_rewards, desired_avg_reward, minimum_steps=100):
    return current_step_idx == total_steps - 1 or \
           (desired_avg_reward <= sum(latest_rewards) / len(latest_rewards)
            if len(latest_rewards) >= minimum_steps else False)
