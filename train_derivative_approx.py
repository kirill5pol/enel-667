from neural_net import NeuralNet
import numpy as np
import argparse


# Constant
MIN_U, MAX_U = -3.0, 3.0  # This is limited to prevent damage to hardware
MIN_ALPHA, MAX_ALPHA = -np.pi, np.pi
MIN_THETA, MAX_THETA = -np.pi, np.pi
MIN_ALPHA_DOT, MAX_ALPHA_DOT = -30.0, 30.0  # Fairly reasonable assumption
MIN_THETA_DOT, MAX_THETA_DOT = -30.0, 30.0  # Fairly reasonable assumption


def gen_us(xs):
    """
    Run the energy + pd controller to get the optimal action u.

    Input:
        xs: The states, shape (N,4)
    Output:
        us: The actions, shape (N,)
    """
    N, D = xs.shape
    assert D == 4

    us = np.zeros((N,))
    for i in range(N):
        us[i] = flip_and_hold_policy(xs[i])
    return us


def gen_ys(xs):
    """
    Create the labels for the neural network.
    We use just the derivates to make it easier to tweak backstepping control
    architectures in the future (easier to just train f(x,u) = x_dot than
    f2 = -x3 + theta_dot_dot & f4 = -eta*u + alpha_dot_dot ->  for example we 
    can change eta after training the neural net!)
    """
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

    ys = np.zeros((N, 2))
    ys[:, 0] = theta_dot_dots
    ys[:, 1] = alpha_dot_dots
    return ys


def gen_xs(N, dist_type="uniform"):
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
        xs[:, 0] = np.random.normal(loc=0.0, scale=0.25 * MAX_ALPHA, size=N)
        xs[:, 1] = np.random.normal(loc=0.0, scale=0.25 * MAX_THETA, size=N)
        xs[:, 2] = np.random.normal(loc=0.0, scale=0.25 * MAX_ALPHA_DOT, size=N)
        xs[:, 3] = np.random.normal(loc=0.0, scale=0.25 * MAX_THETA_DOT, size=N)
    else:
        raise ValueError("Choose one of ['normal', 'uniform']")

    # U is likely going to be close to what a swingup + LQR controller will give
    # so condition it on the state: u_dataset = u_lqr + noise
    # Not this actually gets relativly far away from the LQR controller (control
    # range is -3.0 to 3.0, so this gets as far as +- 1.0 away often)
    xs[:, 4] = gen_us(xs[:, :4]) + np.random.rand(N)
    ys = gen_ys(xs)
    return xs, ys


def main():
    # Parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--train-steps",
        default="1e6",
        type=str,
        help="Number of training steps to take.",
    )
    parser.add_argument(
        "-bs",
        "--batch-size",
        default=64,
        type=int,
        help="The batch size for training (larger means faster training, smaller generalizes better).",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=0.001,
        type=float,
        help="The learning rate for gradient descent.",
    )
    parser.add_argument(
        "-r",
        "--regularization",
        default=0.001,
        type=str,
        help="The L2 regularization multiplier for the neural network training.",
    )
    parser.add_argument(
        "-d",
        "--dropout",
        default=0.7,
        type=str,
        help="The `keep` probability for dropout.",
    )
    parser.add_argument(
        "-ne",
        "--num-examples",
        default=1e5,
        type=float,
        help="How many samples to generate for the derivative aprroximation (ie"
        + "how many points are generated).",
    )
    parser.add_argument(
        "-pi", "--print-interval", default=1000, type=float, help="How often to print."
    )
    parser.add_argument(
        "-si", "--save-interval", default=10000, type=float, help="How often to save."
    )
    args, _ = parser.parse_known_args()

    ns = int(float(args.train_steps))
    bs = args.batch_size
    lr = args.learning_rate
    pi = int(args.print_interval)
    si = int(args.save_interval)
    ne = int(args.num_examples)
    reg = float(args.regularization)

    xs, ys = gen_xs(ne, generate_derivatives, generate_u, "normal")
    N, D = xs.shape
    Ny, M = ys.shape

    # Generate a testing set for looking at the mean percent error
    xs_test, ys_test = gen_xs(ne, generate_derivatives, generate_u, "normal")

    # Create the neural network
    nn = NeuralNet([200, 200], D, M, reg, dropout)

    # Run the train loop and save the network (works even if you stop training early)
    try:
        nn.train_loop(xs, ys, xs_test, ys_test, bs, ns, pi, si, lr)
    finally:
        nn.save(
            f"data/deriv_approx-bs{bs}-ns{args.train_steps}-reg{reg}-{np.random.randint(1000)}"
        )


if __name__ == "__main__":
    main()
