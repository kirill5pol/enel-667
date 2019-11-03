from gym_brt.control import flip_and_hold_policy
from neural_net import NeuralNet

import numpy as np
import argparse

# Constant
MIN_U, MAX_U = -3.0, 3.0
MIN_ALPHA, MAX_ALPHA = -np.pi, np.pi
MIN_THETA, MAX_THETA = -np.pi, np.pi
MIN_ALPHA_DOT, MAX_ALPHA_DOT = -30.0, 30.0  # Double check this
MIN_THETA_DOT, MAX_THETA_DOT = -30.0, 30.0  # Double check this


def diff_forward_model(xs):
    N, D = xs.shape

    g = 9.81  # Gravity constant
    Rm, kt, km = 8.4, 0.042, 0.042  # Motor
    mr, Lr, Dr = 0.095, 0.085, 0.00027  # Rotary Arm
    mp, Lp, Dp = 0.024, 0.129, 0.00005  # Pendulum Link
    Jp, Jr = mp * Lp ** 2 / 12, mr * Lr ** 2 / 12  # Moments of inertia

    thetas = xs[:, 0]
    alphas = xs[:, 1]
    theta_dots = xs[:, 2]
    alpha_dots = xs[:, 3]
    us = xs[:, 4]
    taus = -(km * (us - km * theta_dots)) / Rm  # torque

    # fmt: off
    # From Rotary Pendulum Workbook
    theta_dot_dots = (-Lp*Lr*mp*(-8.0*Dp*alpha_dots + Lp**2*mp*theta_dots**2*np.sin(2.0*alphas) + 4.0*Lp*g*mp*np.sin(alphas))*np.cos(alphas) + (4.0*Jp + Lp**2*mp)*(4.0*Dr*theta_dots + Lp**2*alpha_dots*mp*theta_dots*np.sin(2.0*alphas) + 2.0*Lp*Lr*alpha_dots**2*mp*np.sin(alphas) - 4.0*taus))/(4.0*Lp**2*Lr**2*mp**2*np.cos(alphas)**2 - (4.0*Jp + Lp**2*mp)*(4.0*Jr + Lp**2*mp*np.sin(alphas)**2 + 4.0*Lr**2*mp))
    alpha_dot_dots = (2.0*Lp*Lr*mp*(4.0*Dr*theta_dots + Lp**2*alpha_dots*mp*theta_dots*np.sin(2.0*alphas) + 2.0*Lp*Lr*alpha_dots**2*mp*np.sin(alphas) - 4.0*taus)*np.cos(alphas) - 0.5*(4.0*Jr + Lp**2*mp*np.sin(alphas)**2 + 4.0*Lr**2*mp)*(-8.0*Dp*alpha_dots + Lp**2*mp*theta_dots**2*np.sin(2.0*alphas) + 4.0*Lp*g*mp*np.sin(alphas)))/(4.0*Lp**2*Lr**2*mp**2*np.cos(alphas)**2 - (4.0*Jp + Lp**2*mp)*(4.0*Jr + Lp**2*mp*np.sin(alphas)**2 + 4.0*Lr**2*mp))
    # fmt: on

    d_state = np.zeros((N, 4))
    d_state[:, 0] = theta_dots
    d_state[:, 1] = alpha_dots
    d_state[:, 2] = theta_dot_dots
    d_state[:, 3] = alpha_dot_dots
    return d_state


def generate_u(xs):
    N, D = xs.shape
    assert D == 5

    us = np.zeros((N,))
    for i in range(D):
        us[i] = flip_and_hold_policy(xs[i, :4])
    return us


def generate_training_set(N, dist_type="uniform"):
    """
    Generate an arbitrary number of training points from the set of possible 
    points. Data is distributed along the reasonable values for state and 
    actions. (Actions are assumed to be closeish to the actions from PD control
    or energy control and are normally sampled)

    Input:
        N: Number of training points to create
        dist_type : Uniform or normal distribution. Must be lowercase string.
            - Uniform: This spreads the dataset out, good for locations far away
                the linear region when inverted.
            - Normal: This ends up making the dataset much denser close to
                inverted balance.
    Return:
        xs: Each sample is a state vector appended with action, shape (N, 5)
        ys: Each sample is a derivative of the state vector, shape (N, 4)
    """
    xs = np.zeros((N, 5))  # State vectors + 5th element is u (action)
    if dist_type == "uniform":
        xs[:, 0] = np.random.uniform(low=MIN_ALPHA, high=MAX_ALPHA, size=N)
        xs[:, 1] = np.random.uniform(low=MIN_THETA, high=MAX_THETA, size=N)
        xs[:, 2] = np.random.uniform(low=MIN_ALPHA_DOT, high=MAX_ALPHA_DOT, size=N)
        xs[:, 3] = np.random.uniform(low=MIN_THETA_DOT, high=MAX_THETA_DOT, size=N)
    elif dist_type == "normal":
        xs[:, 0] = np.random.uniform(loc=0.0, scale=0.25 * MAX_ALPHA, size=N)
        xs[:, 1] = np.random.uniform(loc=0.0, scale=0.25 * MAX_THETA, size=N)
        xs[:, 2] = np.random.uniform(loc=0.0, scale=0.25 * MAX_ALPHA_DOT, size=N)
        xs[:, 3] = np.random.uniform(loc=0.0, scale=0.25 * MAX_THETA_DOT, size=N)
    else:
        raise ValueError("Choose one of ['normal', 'uniform']")

    # U is likely going to be close to what a swingup + LQR controller will give
    # so condition it on the state: u_dataset = u_lqr + noise
    xs[:, 4] = generate_u(xs) + np.random.rand(N) * 3
    ys = diff_forward_model(xs)
    return xs, ys


def main():
    # Parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--train-steps",
        default="1e5",
        type=str,
        help="Number of training steps to take",
    )
    parser.add_argument(
        "-bs",
        "--batch-size",
        default=64,
        type=int,
        help="The frequency of samples on the Quanser hardware.",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=0.001,
        type=float,
        help="The frequency of samples on the Quanser hardware.",
    )
    parser.add_argument(
        "-ne",
        "--num-examples",
        default=1e6,
        type=float,
        help="How many samples to generate for the derivative aprroximation (ie"
        + "how many points are generated).",
    )
    parser.add_argument(
        "-ps", "--print-steps", default=1000, type=float, help="How often to print."
    )
    args, _ = parser.parse_known_args()

    ns = int(float(args.train_steps))
    bs = args.batch_size
    lr = args.learning_rate
    ps = args.print_steps
    ne = int(args.num_examples)

    def print_fn(step, ps=ps):
        """If step < ps, print at squares, else print multiples of ps"""
        if step < ps:
            if np.sqrt(step) - int(np.sqrt(step)) < 1e-5:
                return True
        else:
            if step % ps == 0:
                return True
        return False

    xs, ys = generate_training_set(ne)
    N, D = xs.shape
    Ny, M = ys.shape
    nn = NeuralNet(input_dim=D, output_dim=M, hidden_dims=[10, 10], reg=0.001)
    try:
        nn.train_loop(xs, ys, batch_size=bs, n_steps=ns, print_steps=print_fn, lr=lr)
    finally:
        nn.save(
            f"data/nn_deriv_approx_bs{bs}_ns{args.train_steps}_"
            + str(np.random.randint(1000))
        )


if __name__ == "__main__":
    main()
