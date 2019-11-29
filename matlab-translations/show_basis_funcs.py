import numpy as np
import matplotlib.pyplot as plt


### First part of staticTraining.m (fig1) ######################################

# Define constants
n_show_ind_train = 5
n_show_overall_train = 5

beta = 0.1  # learning rate
n_basis_functions = 10
training_spacing = 0.1
evaluation_spacing = 0.01
n_iterations = 100

xmin = -1.25
xmax = 1.25
low = -0.1
high = 1.5

x_eval = np.arange(-1.5, 1.5, step=evaluation_spacing)
x_train = np.arange(-1, 1, step=training_spacing)

# Illustrate different basis functions
m = n_basis_functions - 1
weights = np.arange(0, 1, step=1 / m).reshape(-1, 1)  # .T
centres = np.arange(-1, 1, step=2 / m)
radius = 0.5  # for rectangles, triangles, splines
variance = 0.2  # for Gaussians
sig_factor = 10  # for sigmoids


out_rect = np.zeros(len(x), n_basis_functions)
out_tri = np.zeros(len(x), n_basis_functions)
out_gauss = np.zeros(len(x), n_basis_functions)
out_spline = np.zeros(len(x), n_basis_functions)
out_sig = np.zeros(len(x), n_basis_functions)


x = x_eval
for i in range(len(x)):
    for j in range(n_basis_functions):
        out_rect[i, j] = rectangles(centres[j], radius, x[i]) * weights[j]
        out_tri[i, j] = triangles(centres[j], radius, x[i]) * weights[j]
        out_gauss[i, j] = gaussians(centres[j], variance, x[i]) * weights[j]
        out_spline[i, j] = splines(centres[j], radius, x[i]) * weights[j]
        out_sig[i, j] = sigmoids(centres[j], sig_factor, x[i]) * weights[j]
        # out_tri[i] = triangles(centres, radius,x[i]).T * weights_tri


for j in range(n_basis_functions):
    subplot(5, 1, 1)
    title("Basis Functions")
    plot(x, out_rect[:, j])
    axis([xmin, xmax, low, high])
    # hold on
    subplot(5, 1, 2)
    plot(x, out_tri[:, j])
    axis([xmin, xmax, low, high])
    # hold on
    subplot(5, 1, 3)
    plot(x, out_gauss[:, j])
    axis([xmin, xmax, low, high])
    # hold on
    subplot(5, 1, 4)
    plot(x, out_spline[:, j])
    axis([xmin, xmax, low, high])
    # hold on
    subplot(5, 1, 5)
    plot(x, out_sig[:, j])
    axis([xmin, xmax, low, high])
    # hold on
