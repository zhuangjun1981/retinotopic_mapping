import numpy as np


def weighted_average(values, weights):
    if values.shape != weights.shape:
        raise ValueError('"values" and "weights" should have same shape.')

    if np.amin(weights) < 0.:
        raise ValueError('all weights should be no less than 0.')

    return np.sum((values * weights).flat) / np.sum(weights.flat)
