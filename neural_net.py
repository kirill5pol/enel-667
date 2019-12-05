from utils import (
    mean_percent_error,
    weight_init,
    bias_init,
    relu,
    d_relu,
    affine,
    d_affine,
    dropout,
    d_dropout,
)

import numpy as np
import pickle


class NeuralNetBase(object):
    def __init__(
        self,
        hidden_dims=[30, 30],
        input_dim=4,
        output_dim=1,
        grad_clip=5,
        reg=0.01,
        dropout=0.7,
        beta=0.05,
        nu=0.05,
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
            dropout     : The probability of keeping each neuron in dropout
            beta        : (β) Adaption gain (or learning rate)
            nu          : (ν) E-mod gain
        """
        self.n_hidden = len(hidden_dims)
        self.M = output_dim
        self.D = input_dim
        self.grad_clip = grad_clip
        self.params = {}
        self.reg = reg  # Strength of L2 regulariation
        self.dropout = dropout  # The `keep` probability for dropout
        self.beta = beta
        self.nu = nu

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

    def prediction(self, x, z):
        raise NotImplementedError


class NeuralNetOffline(NeuralNetBase):
    def prediction(self, x, z):
        """
        Compute prediction for the fully-connected net as test time (without
        saving cache and no-dropout).

        Input: 
            x: A numpy array of input data, shape (N, D)
            x_des: Compatibility with the adaptive NN (not used)
        Return:
            output: Output prediction/prediction of label, shape (N, M)
        """
        h = x  # Input into the next layer or previous hidden activation
        for l in range(self.n_hidden):
            l = str(l)
            w = self.params["w" + l]
            b = self.params["b" + l]
            h, _ = affine(h, w, b)  # Affine layer
            h, _ = relu(h)  # Activation (ReLU)

        # Output layer, simply an affine
        output, _ = affine(h, self.params["w_out"], self.params["b_out"])
        return output

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
            l = str(l)
            w, b = self.params["w" + l], self.params["b" + l]
            h, caches["affine" + l] = affine(h, w, b)  # Affine layer
            h, caches["relu" + l] = relu(h)  # Activation (ReLU)
            # Dropout layer (train-time dropout)
            h, caches["dropout" + l] = dropout(h, self.dropout)

        # Output layer, simply an affine
        output, cache = affine(h, self.params["w_out"], self.params["b_out"])
        caches["affine_out"] = cache
        return output, caches

    def prediction_baysian_dropout(self, x, k=10):
        """
        Runs test time prediction k times to get a variance on the output.
        Essentially bayesian dropout.

        Input: 
            x: A numpy array of input data, shape (N, D)
        Return:
            mean: The mean prediction from the dropout ensemble, shape (N, M)
            var: The variance on the prediction from the ensemble, shape (N, M)
        """
        N, D = x.shape
        outputs = np.zeros((k, N, self.M))

        for i in range(k):
            h = x  # Input into the next layer or previous hidden activation
            for l in range(self.n_hidden):
                w, b = self.params["w" + str(l)], self.params["b" + str(l)]
                h, _ = affine(h, w, b)  # Affine layer
                h, _ = relu(h)  # Activation (ReLU)

            # Output layer, simply an affine
            outputs[i], _ = affine(h, self.params["w_out"], self.params["b_out"])

        mean = np.mean(outputs)
        var = np.var(output)

        return mean, var

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
            l = str(l)
            dout = d_dropout(dout, caches["dropout" + l])
            dout = d_relu(dout, caches["relu" + l])
            dout, dw, db = d_affine(dout, caches["affine" + l])

            # Save gradients into a dictionary where the key matches the param key
            grads["w" + l] = dw + self.reg * self.params["w" + l]
            grads["b" + l] = db + self.reg * self.params["b" + l]

        # Clip gradients if enabled - really helps stability! ==================
        if self.grad_clip is not False:
            for key, grad in grads.items():
                grads[key] = np.clip(grads[key], -self.grad_clip, self.grad_clip)

        return loss, grads

    def gradient_step(self, grads, lr=0.001):
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
        self,
        xs,
        ys,
        xs_test,
        ys_test,
        batch_size=32,
        n_steps=1e5,
        print_interval=1000,
        lr=0.001,
    ):
        """
        Run a training loop on the neural network with inputs xs and outputs ys.

        Note the test data should *NOT* be used for training. Only testing to
        see what the mean percent error is on the dataset.

        Input:
            xs: A numpy array of inputs, shape (N, D)
            ys: A numpy array of matching outputs (y = f(x)), shape (N, M)
            xs_test: A numpy array of inputs for testing, shape (NT, D)
            ys_test: A numpy array of matching outputs (y = f(x)), shape (NT, M)
            batch_size: Number of samples to use in a single step
            n_steps: Number of gradient steps to run training
            print_interval: Either a int or a function that take step number and
                returns `True` if it should print at that step.
            save_interval: Either a int or a function that take step number and
                returns `True` if it should save at that step.
            lr: learning rate (or beta)
        """
        xs = np.asarray(xs)  # Convert to numpy array if not already one
        ys = np.asarray(ys)
        N, D = xs.shape
        Ny, M = ys.shape
        assert M == self.output_dim  # Check that the output dim matches the NN
        assert D == self.input_dim  # Check that the output dim matches the NN
        assert N == Ny  # Check that xs and ys have the same number of samples

        Nx_test, D_test = xs_test.shape
        Ny_test, M_test = ys_test.shape
        assert M == M_test  # Check that the test data shapes match train
        assert D == D_test  # Check that the test data shapes match train

        # If print_interval is not a function convert it to one
        if not callable(print_interval):
            # Return true if step is a multiple of print_interval
            print_interval = lambda step: step % print_interval == 0

        # If save_interval is not a function convert it to one
        if not callable(save_interval):
            # Return true if step is a multiple of save_interval
            save_interval = lambda step: step % save_interval == 0

        rand = np.random.randint(1000)

        for step in range(int(n_steps)):
            # Sample a minibatch from the samples
            indices = np.random.choice(N, size=batch_size, replace=False)
            x_batch, y_batch = xs[indices], ys[indices]
            loss, grads = self.loss(x_batch, y=y_batch)
            self.gradient_step(grads, lr=lr)

            if print_interval(step):
                # Calculate mean percent error
                mpe_train = mean_percent_error(self, xs, ys)
                mpe_test = mean_percent_error(self, xs_test, ys_test)
                print(
                    f"MSE loss: {loss:6.3f}, MPE train: {mpe_train:6.3f}, MPE test: {mpe_test:6.3f}, train step: {step}"
                )
            if save_interval(step):
                self.save(f"data/partial-train/model-{rand}_step-{step}")


class NeuralNetAdaptive(NeuralNetBase):
    def prediction_old(self, x, target):
        """
        Compute prediction for the fully-connected net as test time (without
        saving cache and no-dropout).

        Input: 
            x: A numpy array of input data, shape (N, D)
            target: The target for the adaptive NN
        Return:
            output: Output prediction/prediction of label, shape (N, M)
        """
        h = x  # Input into the next layer or previous hidden activation
        for l in range(self.n_hidden):
            l = str(l)
            w = self.params["w" + l]
            b = self.params["b" + l]
            h, _ = affine(h, w, b)  # Affine layer
            h, _ = relu(h)  # Activation (ReLU)
        # Output layer, simply an affine
        output, cache = affine(h, self.params["w_out"], self.params["b_out"])
        return output

        # Technically this is not the real z but the 1/N term only scales z (we
        # can think of this as equivalent to scaling β by 1/N).
        # This is to match how dout works in NeuralNetOffline (see line: 190)
        N, D = x.shape
        z = (output - target) / N

        # Only trainable paramters in the adaptive case are the last layer weights
        # So we only update the output layer weights (using e-mod)
        _, dw, db = w_hat_dot_e_mod(z, cache)

        # Update the weights
        self.params["w_out"] -= self.beta * dw
        self.params["b_out"] -= self.beta * db
        return output

    def prediction(self, x, z):
        """
        Compute prediction for the fully-connected net as test time (without
        saving cache and no-dropout).

        Input: 
            x: A numpy array of input data, shape (N, D)
            z: Diff between y-y_des for the func approx (N, M) (1,2) in this case
        Return:
            output: Output prediction/prediction of label, shape (N, M)
        """
        h = x  # Input into the next layer or previous hidden activation
        for l in range(self.n_hidden):
            l = str(l)
            w = self.params["w" + l]
            b = self.params["b" + l]
            h, _ = affine(h, w, b)  # Affine layer
            h, _ = relu(h)  # Activation (ReLU)
        # Output layer, simply an affine
        output, cache = affine(h, self.params["w_out"], self.params["b_out"])

        # Technically this is not the real z but the 1/N term only scales z (we
        # can think of this as equivalent to scaling β by 1/N).
        # This is to match how dout works in NeuralNetOffline (see line: 190)
        N, D = x.shape
        z = z / N

        # Only trainable paramters in the adaptive case are the last layer weights
        # So we only update the output layer weights (using e-mod)
        _, dw, db = self.w_hat_dot_e_mod(z, cache)

        # Update the weights
        self.params["w_out"] -= self.beta * dw
        self.params["b_out"] -= self.beta * db
        return output

    def w_hat_dot_e_mod(self, dout, cache):
        """
        Modification of d_affine (from utils.py) that adds e-modification.

        Note: This can ONLY be used for the last layer of weights (at least with
        this version of e-mod... there may be variants that allow for backprop)

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

        # E-mod portion
        w_emod = self.nu * np.sum(dout.T, axis=1) * w
        b_emod = self.nu * np.sum(dout.T, axis=1) * b  # 0 # TODO: figure this out!!!

        dx = np.clip(dx, -self.grad_clip, self.grad_clip)
        dw = np.clip(dw, -self.grad_clip, self.grad_clip)
        db = np.clip(db, -self.grad_clip, self.grad_clip)

        # fmt: off
        print(f"Sum of weights: dw={np.sum(dw)}, db={np.sum(db)}, w_emod={np.sum(w_emod)}, b_emod={np.sum(b_emod)}")
        # fmt: on
        return dx, dw, db
