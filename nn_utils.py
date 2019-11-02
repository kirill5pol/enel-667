import numpy as np


def weight_init(input_dim, output_dim):
    """
    Create a weight matrix with shape: (input_dim, output_dim)
    
    He initialization for neural networks with ReLu activations: 
        https://arxiv.org/pdf/1502.01852.pdf
    """
    stddev = np.sqrt(2 / input_dim)
    return np.random.randn(input_dim, output_dim) * stddev


def bias_init(output_dim, scale=1e-5):
    """Create a bias vector with shape: (output_dim,)"""
    return np.random.randn(output_dim) * scale
