import numpy as np


def weight_init(input_dim, output_dim):
    """
    Create a weight matrix with shape: (input_dim, output_dim)
    
    He initialization for neural networks with ReLu activations: 
        https://arxiv.org/pdf/1502.01852.pdf
    """
    stddev = np.sqrt(2 / input_dim)
    return np.random.randn(input_dim, output_dim) * stddev


def bias_init(output_dim, scale=1e-5):
    """Create a bias vector with shape: (output_dim,)"""
    return np.random.randn(output_dim) * scale


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
