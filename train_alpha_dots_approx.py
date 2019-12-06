from neural_net import NeuralNetOffline
import numpy as np
import argparse
import pickle


def get_data(loadfile, frac_test=0.3):
    """
    Gets the data from a file and split the train/test sets.
    """
    params = None
    with open(loadfile, "rb") as f:
        params = pickle.load(f)
    xs = params["xs"]  # Shape (N, 4)
    us = params["us"]  # Shape (N, 1)
    αs = params["αs"]  # Shape (N, 4)
    bs = params["bs"]  # Shape (N, 1)
    x_dots = params["x_dots"]  # Shape (N, 4)
    α_dots = params["α_dots"]  # Shape (N, 4)
    N, _ = xs.shape
    # Since α_dots that we get are meaningless...
    α1_dots = 0
    # α2_dots = np.random.randn(N) * 5
    # α3_dots = np.random.randn(N) * 5
    # α4_dots = np.random.randn(N) * 5
    α2_dots = α_dots[:, 1]
    α3_dots = α_dots[:, 2]
    α4_dots = α_dots[:, 3]
    # Get the values that we actually want (as vectors)
    f2s = -xs[:, 2] + x_dots[:, 1] - α1_dots  # -x3 + x2_dot - α2_dot
    f3s = 0.5 * α2_dots  # -α3_dot / 2
    f4s = -0.5 * α3_dots  # - α4_dot / 2
    # f4s = -bs[:, 0] * us[:, 0] - 0.5 * α3_dots  # -bu + x4_dot - α4_dot / 2
    f_hats = np.zeros((N, 3))  # 4 function approximators
    f_hats[:, 0] = f2s
    f_hats[:, 1] = f3s
    f_hats[:, 2] = f4s
    # Get number of examples in test & train
    n_test = int(frac_test * N)
    n_train = N - n_test
    # Create train and test sets
    xs_train = xs[:n_train, :]
    f_hats_train = f_hats[:n_train, :]
    xs_test = xs[n_train:, :]
    f_hats_test = f_hats[n_train:, :]
    return xs_train, f_hats_train, xs_test, f_hats_test


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
        default=0.01,
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
        "-l",
        "--loadfile",
        default="data/old_controller_run/384.npy",
        type=str,
        help="File to load data from.",
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
    reg = float(args.regularization)

    xs, ys, xs_test, ys_test = get_data(args.loadfile, frac_test=0.3)
    N, D = xs.shape
    Ny, M = ys.shape

    # Create the neural network
    nn = NeuralNetOffline([50, 50], D, M, reg, args.dropout)

    # For training and test error plotting
    mpe_hist = np.zeros((2, ns))

    # Run the train loop and save the network (works even if you stop training early)
    # fmt: off
    try:
        mpe_hist[0,:], mpe_hist[1,:] = nn.train_loop(xs, ys, xs_test, ys_test, bs, ns, pi, si, lr)
    finally:
        filename = f"data/deriv_approx-bs{bs}-ns{args.train_steps}-reg{reg}-{np.random.randint(1000)}"
        nn.save(filename)

        np.save(filename + "train-hist", mpe_hist)  # Save the training history
        print("Train:", mpe_hist[0,:])
        print("Test:", mpe_hist[1,:])
    # fmt: on


if __name__ == "__main__":
    main()
