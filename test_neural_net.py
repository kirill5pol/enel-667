import numpy as np
from neural_net import *


# Numerical grad check from: https://cs231n.github.io
def eval_numerical_gradient_array(f, x, df, h=1e-5):
    """
    Evaluate a numeric gradient for a function that accepts a numpy
    array and returns a numpy array.
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        ix = it.multi_index

        oldval = x[ix]
        x[ix] = oldval + h
        pos = f(x).copy()
        x[ix] = oldval - h
        neg = f(x).copy()
        x[ix] = oldval

        grad[ix] = np.sum((pos - neg) * df) / (2 * h)
        it.iternext()
    return grad


# Numerical grad check from: https://cs231n.github.io
def eval_numerical_gradient(f, x, verbose=True, h=0.00001):
    """
    a naive implementation of numerical gradient of f at x
    - f should be a function that takes a single argument
    - x is the point (numpy array) to evaluate the gradient at
    """

    fx = f(x)  # evaluate function value at original point
    grad = np.zeros_like(x)
    # iterate over all indexes in x
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:

        # evaluate function at x+h
        ix = it.multi_index
        oldval = x[ix]
        x[ix] = oldval + h  # increment by h
        fxph = f(x)  # evalute f(x + h)
        x[ix] = oldval - h
        fxmh = f(x)  # evaluate f(x - h)
        x[ix] = oldval  # restore

        # compute the partial derivative with centered formula
        grad[ix] = (fxph - fxmh) / (2 * h)  # the slope
        if verbose:
            print(ix, grad[ix])
        it.iternext()  # step to next dimension

    return grad


# ===============================================================================
# Test the affine_backward function
np.random.seed(231)
x = np.random.randn(10, 6)
w = np.random.randn(6, 5)
b = np.random.randn(5)
dout = np.random.randn(10, 5)

dx_num = eval_numerical_gradient_array(lambda x: affine(x, w, b)[0], x, dout)
dw_num = eval_numerical_gradient_array(lambda w: affine(x, w, b)[0], w, dout)
db_num = eval_numerical_gradient_array(lambda b: affine(x, w, b)[0], b, dout)

_, cache = affine(x, w, b)
dx, dw, db = d_affine(dout, cache)

# The error should be around e-10 or less
print("Testing affine_backward function:")
print("dx error: ", rel_error(dx_num, dx))
print("dw error: ", rel_error(dw_num, dw))
print("db error: ", rel_error(db_num, db))


# ===============================================================================
x = np.random.randn(10, 10)
dout = np.random.randn(*x.shape)

dx_num = eval_numerical_gradient_array(lambda x: relu(x)[0], x, dout)

_, cache = relu(x)
dx = d_relu(dout, cache)

# The error should be on the order of e-12
print("Testing d_relu function:")
print("dx error: ", rel_error(dx_num, dx))


# ===============================================================================
np.random.seed(231)
N, D, H1, H2, O = 5, 4, 10, 10, 1
X = np.random.randn(N, D)
y = np.random.randn(N, O)
model = NeuralNet(hidden_dims=[H1, H2], input_dim=D, output_dim=O)
loss, grads = model.loss(X, y)
print("Initial loss: ", loss)
# Most of the errors should be on the order of e-7 or smaller.
# NOTE: It is fine however to see an error for W2 on the order of e-5
# for the check when reg = 0.0
for name in sorted(grads):
    f = lambda _: model.loss(X, y)[0]
    grad_num = eval_numerical_gradient(f, model.params[name], verbose=False, h=1e-5)
    print("%s relative error: %.2e" % (name, rel_error(grad_num, grads[name])))


# Test on square root problem ==================================================
xs = np.random.randint(100, size=int(1e4)).reshape(-1, 1)
ys = np.sqrt(xs)
nn = NeuralNet(input_dim=1, output_dim=1, reg=0.0)
nn.train_loop(xs, ys, batch_size=32, n_steps=int(1e4), print_steps=500, lr=0.01)
for i, p in [(i, f"{nn.prediction_no_cache([i])[0][0]:6.3f}") for i in range(-10, 10)]:
    print(np.sqrt(i), p)


# Test on squaring problem =====================================================
xs = np.random.randint(20, size=int(1e5)).reshape(-1, 1) - 10
ys = xs ** 2
nn = NeuralNet(input_dim=1, output_dim=1, hidden_dims=[10, 10], reg=0.001)
nn.train_loop(xs, ys, batch_size=64, n_steps=int(1e4), print_steps=100, lr=0.001)
for i, p in [(i, f"{nn.prediction_no_cache([i])[0][0]:6.3f}") for i in range(-10, 10)]:
    print(i ** 2, p)
