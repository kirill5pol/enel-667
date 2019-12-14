from neural_net import NeuralNetOffline, NeuralNetAdaptive
from gym_brt.envs import QubeSwingupEnv, QubeBalanceEnv
from gym_brt.control import flip_and_hold_policy as fhp

import numpy as np
import pickle
import fire


class DataRunner(object):
    """
    This is a class to record data based on my old controller to collect data 
    for offline training. Otherwise there is no reasonable way for me to collect
    data about what are the α_dots (as those are dependent on the controller...)
    """

    def __init__(self, n_examples):
        self.xs = np.zeros((n_examples, 4))
        self.x_dots = np.zeros((n_examples, 4))
        self.us = np.zeros((n_examples, 1))
        self.αs = np.zeros((n_examples, 4))
        self.α_dots = np.zeros((n_examples, 4))

        # Note: while I'm using b=49.149 here, there would be different values
        # of b if I was doing the full flip up problem rather than stabilization
        self.bs = np.zeros((n_examples, 1))
        self.i = 0
        self.n_examples = n_examples

        # Reason why this isn't local is that you may want to print this out later
        self.filename = f"data/old_controller_run/{np.random.randint(1000)}.npy"

    def step(self, x, u, αs, α_dots, b):
        if self.i % 100 == 0:
            print("\n\n\n", self.i, "\n\n\n")
        if self.i < self.n_examples:
            g = 9.81  # Gravity constant
            Rm, kt, km = 8.4, 0.042, 0.042  # Motor
            mr, Lr, Dr = 0.095, 0.085, 0.00027  # Rotary Arm
            mp, Lp, Dp = 0.024, 0.129, 0.00005  # Pendulum Link
            Jp, Jr = mp * Lp ** 2 / 12, mr * Lr ** 2 / 12  # Moments of inertia

            theta, alpha, theta_dot, alpha_dot = x
            tau = -(km * (u - km * theta_dot)) / Rm  # torque

            # fmt: off
            # From Rotary Pendulum Workbook
            theta_dot_dot = (-Lp*Lr*mp*(-8.0*Dp*alpha_dot + Lp**2*mp*theta_dot**2*np.sin(2.0*alpha) + 4.0*Lp*g*mp*np.sin(alpha))*np.cos(alpha) + (4.0*Jp + Lp**2*mp)*(4.0*Dr*theta_dot + Lp**2*alpha_dot*mp*theta_dot*np.sin(2.0*alpha) + 2.0*Lp*Lr*alpha_dot**2*mp*np.sin(alpha) - 4.0*tau))/(4.0*Lp**2*Lr**2*mp**2*np.cos(alpha)**2 - (4.0*Jp + Lp**2*mp)*(4.0*Jr + Lp**2*mp*np.sin(alpha)**2 + 4.0*Lr**2*mp))
            alpha_dot_dot = (2.0*Lp*Lr*mp*(4.0*Dr*theta_dot + Lp**2*alpha_dot*mp*theta_dot*np.sin(2.0*alpha) + 2.0*Lp*Lr*alpha_dot**2*mp*np.sin(alpha) - 4.0*tau)*np.cos(alpha) - 0.5*(4.0*Jr + Lp**2*mp*np.sin(alpha)**2 + 4.0*Lr**2*mp)*(-8.0*Dp*alpha_dot + Lp**2*mp*theta_dot**2*np.sin(2.0*alpha) + 4.0*Lp*g*mp*np.sin(alpha)))/(4.0*Lp**2*Lr**2*mp**2*np.cos(alpha)**2 - (4.0*Jp + Lp**2*mp)*(4.0*Jr + Lp**2*mp*np.sin(alpha)**2 + 4.0*Lr**2*mp))
            # fmt: on

            self.xs[self.i, :] = x
            self.x_dots[self.i, :] = theta_dot, alpha_dot, theta_dot_dot, alpha_dot_dot
            self.us[self.i, :] = u
            self.αs[self.i, :] = αs  # The virtual controls NOT the alpha angles
            self.α_dots[self.i, :] = α_dots  # The virtual controls NOT the alpha angles
            self.bs[self.i, :] = b
            self.i += 1
            if self.i >= self.n_examples:
                self.save()

    def save(self):
        # Save to a random filename then print out filename
        params = {}  # Python dictionary (basically a hashtable)
        params["xs"] = self.xs
        params["us"] = self.us
        params["αs"] = self.αs
        params["bs"] = self.bs
        params["x_dots"] = self.x_dots
        params["α_dots"] = self.α_dots

        with open(self.filename, "wb") as f:
            pickle.dump(params, f)
        print("Saved neural network parameters to:", self.filename)


def f_approx(x, zs, nn):
    """
    Function approximation for the backtepping controller.
    
    Input:
        x: The current state, shape (4,)
        zs: The 'dout' for the NN, shape (2,)
        nn: Neural network predictor
    Return:
        fs: The estimated solutions for f1, f2, f3, f4
    """
    _, f2, f3, f4 = nn.prediction(x.reshape(1, -1), zs.reshape(1, -1))[0]
    return [0.0, f2, f3, f4]


class BacksteppingController(object):
    def __init__(
        self,
        approximator,
        frequency,
        grad_clip,
        beta,
        nu,
        adaptive=True,
        data_runner=None,
        loadfile=None,
    ):
        if adaptive:
            self.nn = NeuralNetAdaptive(
                hidden_dims=[500],
                input_dim=4,
                output_dim=4,
                grad_clip=grad_clip,
                beta=beta,
                nu=nu,
            )
        else:
            self.nn = NeuralNetOffline()

        if loadfile:
            self.nn.load(loadfile)

        if approximator == "nn":
            self.approximator = f_approx
        else:
            raise ValueError("approximator must be 'nn'.")

        self.frequency = frequency  # Hz
        self.α1_prev = 0
        self.α2_prev = 0
        self.α3_prev = 0
        self.α4_prev = 0

        self.v_dot_max = -1e6  # Some large initial negative number
        self.prev_u = 0

        if data_runner:  # Should I save the run data?
            self.dr = DataRunner(int(data_runner))
        else:
            self.dr = None

    def action(self, state, step):
        # b information:
        #     The value of k_u_alpha_dot = 49.1493074 in the linearized version
        #     For the full system max was: 9829.27
        b = 49.1493074

        k_scale = 2
        k1, k2, k3, k4 = k_scale * 1, k_scale * 1, k_scale * 2, k_scale * 2
        x1, x2, x3, x4 = state

        α1 = 0
        z1 = x1 - α1
        α1_dot = (α1 - self.α1_prev) * self.frequency  # For DataRunner (not for ctrl)

        # No need to get the f1 approximator (f1_hat = -α1_dot = 0)
        f1_hat = 0

        α2 = -k1 * z1 - f1_hat
        z2 = x2 - α2
        α2_dot = (α2 - self.α2_prev) * self.frequency  # For DataRunner (not for ctrl)

        # Get f2 approximator output, update weights that push z2 to 0. ====== #
        zs = np.array([0, z2, 0, 0])  # Hack that won't update any of the other weights
        _, f2_hat, _, _ = self.approximator(state, zs, self.nn)
        f2_hat = np.clip(f2_hat, -300.0, 300.0)

        α3 = -k2 * z2 - f2_hat - z1
        z3 = x3 - α3
        α3_dot = (α3 - self.α3_prev) * self.frequency  # For DataRunner (not for ctrl)

        # Get f3 approximator output, update weights that push z3 to 0. ====== #
        zs = np.array([0, 0, z3, 0])  # Hack that won't update any of the other weights
        _, _, f3_hat, _ = self.approximator(state, zs, self.nn)
        f3_hat = np.clip(f3_hat, -300.0, 300.0)

        α4 = -(k3 / 2) * (2 * z3 + α3) - f3_hat - (z2 / 4)
        z4 = x4 - α4
        α4_dot = (α4 - self.α4_prev) * self.frequency  # For DataRunner (not for ctrl)

        # Get f4 approximator output, update weights that push z4 to 0. ====== #
        zs = np.array([0, 0, 0, z4])  # Hack that won't update any of the other weights
        _, _, _, f4_hat = self.approximator(state, zs, self.nn)
        f4_hat = np.clip(f4_hat, -300.0, 300.0)

        u = (1 / (2 * b)) * (
            -k4 * (2 * z4 + α4)
            - 2 * f4_hat
            - (2 * z3 + α3)
            - ((α4 * (2 * z3 + α3)) / (2 * z4 + α4))
            + ((z2 + α2) / (2 * (2 * z4 + α4)))
        )

        if self.dr:
            self.dr.step(
                state, u, [α1, α2, α3, α4], [α1_dot, α2_dot, α3_dot, α4_dot], b
            )
        return (u,)


def run_backstepping(
    use_simulator=True,
    frequency=1000,
    adaptive=True,
    approximator="nn",  # Options are `real` and `nn`
    data_runner=None,  # Should be the number of steps to save after
    loadfile="data/model-465_step-140000",
    render=True,
    grad_clip=5,  # Can be a number or False
    beta=0.001,
    nu=0.001,
):
    if render == False:
        render_fn = lambda step: False
    elif frequency > 1000:
        # Useful for very large frequencies so that the render time doesn't
        # slow down the simulation too much (render at max ~1khz)
        render_fn = lambda step, f=frequency: step % int(frequency / 1000) == 0
    else:
        render_fn = lambda step: True

    bs = BacksteppingController(
        approximator, frequency, grad_clip, beta, nu, adaptive, data_runner, loadfile
    )

    try:
        env = QubeBalanceEnv(use_simulator=use_simulator, frequency=frequency)
        while True:
            state = env.reset()
            state, _, done, _ = env.step(np.array([0]))
            step = 0

            while not done:
                action = bs.action(state, step)
                state, _, done, _ = env.step(action)
                if render_fn(step):
                    env.render()
                step += 1
    finally:
        if bs.dr:  # Print out the filename after you kill python if you ran DataRunner
            print(f"The saved file is at: {bs.dr.filename}")


if __name__ == "__main__":
    fire.Fire(run_backstepping)
