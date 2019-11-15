# from train_backstepping_unknown_dynamics import generate_unkown_dyn_labels as f2f4
from neural_net import NeuralNet
from utils import HighPassFilter
from gym_brt.control import flip_and_hold_policy
import numpy as np


def f_real(x, u, nn=None, eta=2000):
    """
    Exact solutions to the functions in the backstepping controller.
    
    Input:
        x: The current state, shape (4,)
        u: The current action, real num/float
        nn: *Not used* (for compatibility with f_approx)
        eta: The gain on the control
    Return:
        fs: The exact solutions for f1, f2, f3, f4

    Eta information:
        The value of k_u_alpha_dot = 49.1493074 in the linearized version
        For the full system I got this: 9829.27
        (from wolfram alpha: https://tinyurl.com/y3kgcn38)
        So if 100 doesn't work try 10_000 or 20_000
    """
    g = 9.81  # Gravity constant
    Rm, kt, km = 8.4, 0.042, 0.042  # Motor
    mr, Lr, Dr = 0.095, 0.085, 0.00027  # Rotary Arm
    mp, Lp, Dp = 0.024, 0.129, 0.00005  # Pendulum Link
    Jp, Jr = mp * Lp ** 2 / 12, mr * Lr ** 2 / 12  # Moments of inertia

    theta = x[0]
    alpha = x[1]
    theta_dot = x[2]
    alpha_dot = x[3]
    tau = -(km * (u - km * theta_dot)) / Rm  # torque

    # fmt: off
    # From Rotary Pendulum Workbook
    theta_dot_dot = (-Lp*Lr*mp*(-8.0*Dp*alpha_dot + Lp**2*mp*theta_dot**2*np.sin(2.0*alpha) + 4.0*Lp*g*mp*np.sin(alpha))*np.cos(alpha) + (4.0*Jp + Lp**2*mp)*(4.0*Dr*theta_dot + Lp**2*alpha_dot*mp*theta_dot*np.sin(2.0*alpha) + 2.0*Lp*Lr*alpha_dot**2*mp*np.sin(alpha) - 4.0*tau))/(4.0*Lp**2*Lr**2*mp**2*np.cos(alpha)**2 - (4.0*Jp + Lp**2*mp)*(4.0*Jr + Lp**2*mp*np.sin(alpha)**2 + 4.0*Lr**2*mp))
    alpha_dot_dot = (2.0*Lp*Lr*mp*(4.0*Dr*theta_dot + Lp**2*alpha_dot*mp*theta_dot*np.sin(2.0*alpha) + 2.0*Lp*Lr*alpha_dot**2*mp*np.sin(alpha) - 4.0*tau)*np.cos(alpha) - 0.5*(4.0*Jr + Lp**2*mp*np.sin(alpha)**2 + 4.0*Lr**2*mp)*(-8.0*Dp*alpha_dot + Lp**2*mp*theta_dot**2*np.sin(2.0*alpha) + 4.0*Lp*g*mp*np.sin(alpha)))/(4.0*Lp**2*Lr**2*mp**2*np.cos(alpha)**2 - (4.0*Jp + Lp**2*mp)*(4.0*Jr + Lp**2*mp*np.sin(alpha)**2 + 4.0*Lr**2*mp))
    # fmt: on

    f1 = theta_dot  # f1
    f2 = theta_dot_dot - alpha  # f2 = -x3 + theta_dot_dot = theta_dot_dot - alpha
    f3 = alpha_dot  # f3
    f4 = alpha_dot_dot - ETA * u  # f4 = -eta*u + alpha_dot_dot
    return [f1, f2, f3, f4]


def f_approx(x, u, nn, eta=2000):
    """
    Function approximation for the backtepping controller.
    
    Input:
        x: The current state, shape (4,)
        u: The current action, real num/float
        nn: Neural network predictor
        eta: Gain on the control
    Return:
        fs: The estimated solutions for f1, f2, f3, f4

    Eta information:
        The value of k_u_alpha_dot = 49.1493074 in the linearized version
        For the full system I got this: 9829.27
        (from wolfram alpha: https://tinyurl.com/y3kgcn38)
        So if 100 doesn't work try 10_000 or 20_000
    """
    s = np.zeros((5,))
    s[:4] = x
    s[4] = u
    f2_hat, f4_hat = nn.prediction(s.reshape(1, -1))[0]
    # If the neural network doesn't include the -alpha, and -eta*u terms in the approximation
    # f2_hat -= alphas
    # f4_hat -= eta * taus

    f1 = theta_dot  # f1
    f2 = theta_dot_dot - alpha  # f2 = -x3 + theta_dot_dot = theta_dot_dot - alpha
    f3 = alpha_dot  # f3
    f4 = alpha_dot_dot - ETA * u  # f4 = -eta*u + alpha_dot_dot
    return [f1, f2, f3, f4]


class BacksteppingController(object):
    def __init__(self):
        self.nn = NeuralNet()
        self.nn.load("data/BEST-MODEL--model-91_step-16000")
        self.freq = 250.0  # Hz
        self.x1d_prev = 0
        self.x2d_prev = 0
        self.x3d_prev = 0
        self.x4d_prev = 0

    def action(self, state, step):
        # Eta information:
        #     The value of k_u_alpha_dot = 49.1493074 in the linearized version
        #     For the full system I got this: 9829.27
        #     (from wolfram alpha: https://tinyurl.com/y3kgcn38)
        #     So if 100 doesn't work try 10_000 or 20_000
        eta = 2000

        c1, c2, c3, c4 = 1, 1, 2, 2
        x1, x2, x3, x4 = state

        f1_hat, f2_hat, f3_hat, f4_hat = f_real(
            state, flip_and_hold_policy(state), self.nn, eta
        )

        x1d = np.sin(0.1 * step / self.freq)  # Sin wave? # Desired theta
        z1 = x1 - x1d
        x1d_dot = (x1d - self.x1d_prev) * self.freq

        x2d = -c1 * z1 - f1_hat + x1d_dot
        z2 = x2 - x2d
        x2d_dot = (x2d - self.x2d_prev) * self.freq

        x3d = 2 * (-c2 * z2 - z1 - f2_hat + x2d_dot)
        z3 = x3 - x3d
        x3d_dot = (x3d - self.x3d_prev) * self.freq

        x4d = -c3 * (2 * z3 + x3d) - (z2 / 2) - 2 * f3_hat + x3d_dot
        z4 = x4 - x4d
        x4d_dot = (x4d - self.x4d_prev) * self.freq

        u = 0.5 * eta * (-c4 * (2 * z4 + x4d) - 2 * z3 - 2 * f4_hat - x3d + x4d_dot)
        print(u)
        return u


def run_backstepping():
    from gym_brt.envs import QubeSwingupEnv, QubeBalanceEnv

    with QubeBalanceEnv(use_simulator=True, frequency=250) as env:
        while True:
            state = env.reset()
            # env.qube.state += np.random.randn(4) * 0.01
            state, _, done, _ = env.step(np.array([0]))

            bs = BacksteppingController()
            step = 0
            while not done:
                action = bs.action(state, step)
                state, _, done, _ = env.step(action)
                env.render()
                step += 1


if __name__ == "__main__":
    run_backstepping()
