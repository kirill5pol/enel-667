from gym_brt.control import flip_and_hold_policy
from scipy.signal import savgol_filter
import numpy as np


class HighPassFilter(object):
    def __init__(self, frequency, fc=50):
        self.x_dot_cstate = 0
        self.frequency = frequency
        self.fc = fc

    def reset(self):
        self.x_dot_cstate = 0

    def __call__(self, x):
        x_dot = -(self.fc ** 2) * self.x_dot_cstate + self.fc * x
        self.x_dot_cstate += (-self.fc * self.x_dot_cstate + x) / self.frequency
        return x_dot


class LowPassFilter(object):
    def __init__(self, frequency, rc=50):
        self.y_prev = 0
        self.frequency = frequency
        self.alpha = 1 / (1 + frequency * rc)

    def reset(self):
        self.y_prev = 0

    def __call__(self, x):
        y = self.alpha * x + (1 - self.alpha) * self.y_prev
        return y


class SGFilter(object):
    def __init__(self, window_size=5, order=1, deriv=0):
        self.ws = window_size
        self.o = order
        self.deriv = deriv
        self.data = [0.0 for _ in range(self.ws)]

    def reset(self):
        self.data = [0.0 for _ in range(self.ws)]

    def __call__(self, x):
        self.data.append(x)
        length = len(self.data)
        npd = np.array(self.data[length - self.ws :])

        if length >= self.ws:
            return savgol_filter(npd, self.ws, self.o, deriv=self.deriv)[-1]
        else:
            raise ValueError("Data length is not sufficient!!!")


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

    a = np.abs((ys - y_hats))
    b = ys
    c = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    mpe = np.mean(c)
    # mpe = np.mean(np.abs((ys - y_hats) / ys))
    return mpe


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


# ==============================================================================
# Neural Network Weight Initialization
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


# ==============================================================================
# Neural Network Layers
def relu(x):
    """
    Forward pass for a rectified linear unit (ReLU).

    Input:
        x: Inputs, any shape
    Return:
        out: Output, same shape as x
        cache: x
    """
    out = np.maximum(x, 0)
    cache = x
    return out, cache


def d_relu(dout, cache):
    """
    Backward pass for a rectified linear unit (ReLU).

    Input:
        dout: Upstream derivatives, any shape == dloss/dout
        cache: Input x, of same shape as dout
    Returns:
        dx: Gradient with respect to x (input to layer)
    """
    x = cache
    dx = dout * (x > 0)
    return dx


def affine(x, w, b):
    """
    Forward pass for an affine (fully-connected) layer.

    The input x has shape (N, D) and contains a minibatch of N
    examples, where each example x[i] has shape (D).

    Input:
        x: A numpy array of input data, shape (N, D)
        w: A numpy array of weights, shape (D, M)
        b: A numpy array of biases, shape (M,)
    Return:
        out: output, of shape (N, M)
        cache: (x, w, b)
    """
    out = x @ w + b.reshape(1, -1)  # Matrix multiple then add bias
    cache = (x, w, b)

    return out, cache


def d_affine(dout, cache):
    """
    Backward pass for an affine layer.

    Input:
        dout: Upstream derivative, shape (N, M) == dloss/dout
        cache: Tuple of:
            x: Input data, shape (N, D)
            w: Weights, shape (D, M)
            b: Biases, shape (M,)
    Return:
        dx: Gradient with respect to x, shape (N, D)  == dloss/dx
        dw: Gradient with respect to w, shape (D, M)  == dloss/dw
        db: Gradient with respect to b, shape (M,)  == dloss/db
    """
    x, w, b = cache

    dx = dout @ w.T  # d_loss/dx = d_loss/dout * dout/dx
    dw = x.T @ dout
    db = np.sum(dout.T, axis=1)  # Sum to make shape (M,)
    return dx, dw, db


def dropout(x, p):
    """
    Performs the (train time) forward pass for (inverted) dropout.

    Inverted dropout works by scaling the activations by the inverse of keep
    probability `p`. This allows for test time dropout to work without rescaling
    the activations (ie acts as if the netwrok was not trained with dropout).


    Input:
        x: Input data, of any shape
        p: Dropout parameter. The probability of keeping each neuron (not 
            probability of dropping!)

    Outputs:
        out: Array of the same shape as x.
        cache: Mask of values to be set to 0. (True is set to 0, False is kept).
    """
    out = x
    mask = np.random.uniform(size=x.shape) > p  # True values are set to 0
    out[mask] = 0.0
    # Scale by the inverse of p, so when you remove dropout the sum of the
    # activations are going to be (on average) the same as the one with dropout
    out /= p
    cache = mask

    return out, cache


def d_dropout(dout, cache):
    """
    Perform the (train time) backward pass for (inverted) dropout.


    Input:
        dout: Upstream derivatives, of any shape
        cache: mask from dropout_forward.

    Return:
        dx: Gradient with respect to x (input to layer)
    """
    mask = cache
    dx = dout
    dout[mask] = 0
    return dx
