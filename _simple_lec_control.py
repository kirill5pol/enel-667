from neural_net import NeuralNet
from gym_brt.control import flip_and_hold_policy as fhp
import numpy as np


def flip_and_hold_policy(state, **kwargs):
    return np.clip(fhp(state, **kwargs), -3, 3)  # return fhp(state, **kwargs)#


def f_real(x, u, nn):
    """
    Exact solutions to the functions in the backstepping controller.
    
    Input:
        x: The current state, shape (4,)
        u: The current action, real num/float
        nn: *Not used* (for compatibility with f_approx)
    Return:
        f: The exact solutions for the sum of states
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

    return theta_dot + theta_dot_dot + alpha_dot + alpha_dot_dot


def f_approx(x, u, nn):
    theta_dot = x[2]
    alpha_dot = x[3]
    s = np.zeros((5,))
    s[:4] = x
    s[4] = u
    theta_dot_dot_approx, alpha_dot_dot_approx = nn.prediction(s.reshape(1, -1))[0]

    return theta_dot + theta_dot_dot_approx + alpha_dot + alpha_dot_dot_approx


class BacksteppingController(object):
    def __init__(self):
        self.nn = NeuralNet()
        self.nn.load("data/BEST-MODEL--model-91_step-16000")
        self.prev_u = 0
        self.v_dot_max = -100000

    def action(self, state, step):
        x1, x2, x3, x4 = state
        # print(state)

        k = 2000
        b = 50

        # =======================================================================
        f = (1 / b) * f_real(state, flip_and_hold_policy(state), self.nn)
        # f = (1 / b) * f_real(state, self.prev_u, self.nn)
        z = x1 + x2 + x3 + x4

        zqtw = f * z
        if zqtw > 0:
            u = -f - k * (z ** 2)
        else:
            u = -k * (z ** 2)

        u = np.clip(u, -18, 18)
        # =======================================================================
        # f = (1 / b) * f_real(state, u, self.nn)
        # z = x1 + x2 + x3 + x4

        # zqtw = f * z
        # if zqtw > 0:
        #     u = -f - k * (z ** 2)
        # else:
        #     u = -k * (z ** 2)
        # # =======================================================================
        u = np.clip(u, -18, 18)

        f2 = (1 / b) * f_real(state, u, self.nn)
        dx_mag = np.abs(f2 - f)
        v_dot = dx_mag * np.abs(z) - k * z ** 2

        if v_dot > self.v_dot_max:
            self.v_dot_max = v_dot
        print(f"u={u}, v_dot={v_dot}, v_dot_max={self.v_dot_max}")

        self.prev_u = u
        return u


def run_backstepping():
    from gym_brt.envs import QubeSwingupEnv, QubeBalanceEnv

    frequency = 100000
    with QubeBalanceEnv(use_simulator=True, frequency=frequency) as env:
        bs = BacksteppingController()
        while True:
            print("resetting")
            state = env.reset()
            # env.qube.state += np.random.randn(4) * 0.01
            state, _, done, _ = env.step(np.array([0]))

            step = 0
            while not done:
                action = bs.action(state, step)
                state, _, done, _ = env.step(action)
                if frequency > 1000:
                    if step % int(frequency / 1000) == 0:
                        env.render()
                step += 1


if __name__ == "__main__":
    run_backstepping()
