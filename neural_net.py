from nn_layers import relu, d_relu, affine, d_affine
from nn_utils import weight_init, bias_init

import numpy as np
import pickle


class NeuralNet(object):
    def __init__(
        self, hidden_dims=[30, 30], input_dim=4, output_dim=1, grad_clip=5, reg=0.01
    ):
        """
        Create a simple multi-layer perceptron network with a number of hidden
        layers.
        
        Input:
            hidden_dims : A tuple or list of hidden dimensions
            input_dim   : Dimensions of the input (x)
            output_dim  : Dimensions of the output/prediction (y_hat)
            grad_clip   : Clips the gradient for stability in gradient descent
                Either False (disable gradient clipping) or a number maximum abs 
                value of the gradient
            reg         : L2 regularization scale (==0.0 for no regularization)
        """
        self.n_hidden = len(hidden_dims)
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.grad_clip = grad_clip
        self.params = {}
        self.reg = reg  # Strength of L2 regulariation

        # For each layer get the input and output dimensions
        input_dims = [input_dim] + hidden_dims  # By default: [4, 10, 10]
        output_dims = hidden_dims + [output_dim]  # By default: [10, 10, 1]

        # Create all of the initial weights and biases, save to dictionary (aka hashtable)
        self.params = {}
        for l in range(self.n_hidden):
            i, o = input_dims[l], output_dims[l]
            self.params["w" + str(l)] = weight_init(i, o)
            self.params["b" + str(l)] = bias_init(o)
        self.params["w_out"] = weight_init(input_dims[-1], output_dim)
        self.params["b_out"] = bias_init(output_dim)

    def save(self, filename):
        """Save paramters to `filename`."""
        with open(filename, "wb") as f:
            pickle.dump(self.params, f)
        print("Saved neural network parameters to:", filename)

    def load(self, filename):
        """Load paramters from `filename`."""
        with open(filename, "rb") as f:
            self.params = pickle.load(f)

    def prediction_save_cache(self, x):
        """
        Compute prediction for the fully-connected net and save intermediate 
        activations.

        N samples, D dims per sample, each sample is a row vec, M is the dims of
        y/prediction

        Input: 
            x: A numpy array of input data, shape (N, D)
        Return:
            output: Output prediction/prediction of label, shape (N, M)
            caches: Saved intermediate activations for use in backprop
        """
        caches = {}
        h = x  # Input into the next layer or previous hidden activation
        for l in range(self.n_hidden):
            w, b = self.params["w" + str(l)], self.params["b" + str(l)]
            h, cache = affine(h, w, b)  # Affine layer
            caches["affine" + str(l)] = cache
            h, cache = relu(h)  # Activation (ReLU)
            caches["relu" + str(l)] = cache

        # Output layer, simply an affine
        output, cache = affine(h, self.params["w_out"], self.params["b_out"])
        caches["affine_out"] = cache
        return output, caches

    def prediction(self, x):
        """
        Compute prediction for the fully-connected net without saving cache.

        Input: 
            x: A numpy array of input data, shape (N, D)
        Return:
            output: Output prediction/prediction of label, shape (N, M)
        """
        h = x  # Input into the next layer or previous hidden activation
        for l in range(self.n_hidden):
            w, b = self.params["w" + str(l)], self.params["b" + str(l)]
            h, _ = affine(h, w, b)  # Affine layer
            h, _ = relu(h)  # Activation (ReLU)

        # Output layer, simply an affine
        output, _ = affine(h, self.params["w_out"], self.params["b_out"])
        return output

    def loss(self, x, y=None):
        """Compute loss and gradient for the fully-connected net."""
        if len(y.shape) == 2:
            N, M = y.shape  # N is n_samples, M is dims of each sample
        elif len(y.shape) == 1:
            M = y.shape[0]
        else:
            raise ValueError("y has incorrect shape")

        output, caches = self.prediction_save_cache(x)  # Forward pass
        grads = {}

        # Calculate the loss for the current batch =============================
        # Get the mean squared error loss (1/2 to simplify derivative)
        loss = 0.5 * np.mean((output - y) ** 2)
        # Add a regularization term
        for l in range(self.n_hidden):
            loss += 0.5 * self.reg * np.sum(self.params[f"w{l}"] ** 2)
            loss += 0.5 * self.reg * np.sum(self.params[f"b{l}"] ** 2)
        loss += 0.5 * self.reg * np.sum(self.params["w_out"] ** 2)
        loss += 0.5 * self.reg * np.sum(self.params["b_out"] ** 2)

        # Get the gradients through backprop ===================================
        # Gradient from the MSE loss
        dout = (output - y) / N
        # Backprop through output layer
        dout, dw, db = d_affine(dout, caches["affine_out"])
        grads["w_out"] = dw + self.reg * self.params["w_out"]
        grads["b_out"] = db + self.reg * self.params["b_out"]
        # Backprop through each hidden layer
        for l in reversed(range(self.n_hidden)):
            dout = d_relu(dout, caches["relu" + str(l)])
            dout, dw, db = d_affine(dout, caches["affine" + str(l)])
            # Save gradients into a dictionary where the key matches the param key
            grads["w" + str(l)] = dw + self.reg * self.params["w" + str(l)]
            grads["b" + str(l)] = db + self.reg * self.params["b" + str(l)]

        # Clip gradients if enabled - really helps stability! ==================
        if self.grad_clip is not False:
            for key, grad in grads.items():
                grads[key] = np.clip(grads[key], -self.grad_clip, self.grad_clip)

        return loss, grads  # , dout

    def gradient_step(self, grads, lr=0.01):
        """
        A single step of gradient descent.

        Input:
            grads: a dictionary where the elements match the parameters in 
                self.params
            lr: the learning rate
        """
        for key, grad in grads.items():
            self.params[key] -= lr * grad  # Gradient step

    def train_loop(
        self, xs, ys, batch_size=32, n_steps=1e5, print_steps=1000, lr=0.001
    ):
        """
        Run a training loop on the neural network with inputs xs and outputs ys.

        Input:
            xs: A numpy array of inputs, shape (N, D)
            ys: A numpy array of matching outputs (y = f(x)), shape (N, M)
            batch_size: Number of samples to use in a single step
            n_steps: Number of gradient steps to run training
            print_steps: Either a int or a function that take step number and
                returns `True` if it should print at that step.
            lr: learning rate (or beta)
        """
        xs = np.asarray(xs)  # Convert to numpy array if not already one
        ys = np.asarray(ys)
        N, D = xs.shape
        Ny, M = ys.shape
        assert M == self.output_dim  # Check that the output dim matches the NN
        assert D == self.input_dim  # Check that the output dim matches the NN
        assert N == Ny  # Check that xs and ys have the same number of samples

        # If print_steps is not a function convert it to one
        if not callable(print_steps):
            # Return true if step is a multiple of print_steps
            print_steps = lambda step: step % print_steps == 0

        for step in range(int(n_steps)):
            # Sample a minibatch from the samples
            indices = np.random.choice(N, size=batch_size, replace=False)
            x_batch, y_batch = xs[indices], ys[indices]
            loss, grads = self.loss(x_batch, y=y_batch)
            self.gradient_step(grads, lr=lr)

            if print_steps(step):
                print(f"MSE loss: {loss:6.3f}, train step: {step}")
