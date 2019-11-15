from gym_brt.control import flip_and_hold_policy
import numpy as np


class HighPassFilter(object):
    def __init__(self, frequency, fc=50):
        self.x_dot_cstate = 0
        self.frequency = frequency
        self.fc = fc

    def __call__(self, x):
        x_dot = -(self.fc ** 2) * self.x_dot_cstate + self.fc * x
        self.x_dot_cstate += (-self.fc * self.x_dot_cstate + x) / self.frequency
        return x_dot


def mean_percent_error(nn, xs, ys):
    """
    Calculate the mean percent error of the network.

    Input:
        nn: The neural network object
        xs: The set of inputs to test on, shape (N, D)
        ys: The set of labels to test on, shape (N, M)

    Return:
        mpe: The mean percent error on the examples given.
    """
    N, D = xs.shape
    Ny, M = ys.shape
    assert N == Ny

    y_hats = nn.prediction(xs)  # Get the predicted outputs
    assert y_hats.shape == ys.shape

    return np.mean(np.abs((ys - y_hats) / ys))


def test_networks(filenames, xs, ys):
    from neural_net import NeuralNet

    N, D = xs.shape
    Ny, M = ys.shape
    assert N == Ny

    nn = NeuralNet(input_dim=D, output_dim=M)
    mpes = []
    filenames = sorted(filenames)
    for filename in filenames:
        nn.load(filename)
        mpes.append(mean_percent_error(nn, xs, ys))

    return mpes, filenames
