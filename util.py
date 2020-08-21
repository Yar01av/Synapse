import math

import numpy as np
import torch


def select_index_from_probs(prob_matrix):
    s = prob_matrix.cumsum(axis=1)
    r = np.random.rand(prob_matrix.shape[0])
    k = (s < r.reshape((len(s), 1))).sum(axis=1)

    return k


def get_output_size(network, shape):
    out = network(torch.zeros(size=(1, *shape)))

    return torch.flatten(out, start_dim=1).shape[1]