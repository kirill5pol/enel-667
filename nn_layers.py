import numpy as np


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
