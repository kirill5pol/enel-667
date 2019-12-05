import numpy as np


def gaussians(center, sigma, x):
    output = exp(-(x - center) ** 2 / sigma ** 2)
    return output


def rectangles(centers, radius, x):
    output = np.zeros_like(centers)
    for i in range(len(centers)):
        if abs(x - centers[i]) < radius:
            output[i] = 1
        else:
            output[i] = 0
    return output


def sigmoids(center, sigFactor, x):
    output = (tanh((x - center) * sigFactor) + 1) / 2
    return output


def splines(centers, r, x):
    output = np.zeros_like(centers)
    for i in range(len(centers)):
        if abs(x - centers[i]) > r:
            output[i] = 0
        else:
            h = (x - (centers[i] - r)) / (2 * r)
            output[i] = 16 * (h ** 2 - 2 * h ** 3 + h ** 4)
    return output


def triangles(center, r, x):
    output = np.zeros_like(center)
    for i in range(len(center)):
        if abs(x - center[i]) > r:
            output[i] = 0
        else:
            output[i] = (x - center[i]) / r * sign(center[i] - x) + 1
    return output
