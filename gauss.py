import numpy as np
from numba import njit

@njit
def gaussian1(x, sigma):
    exp_val = np.exp(-x**2 / (2 * sigma**2))
    return exp_val / (sigma * np.sqrt(2 * np.pi))

@njit
def grad_gaussian1(x, sigma, gaussian_val=None):
    if gaussian_val is None:
        gaussian_val = gaussian1(x, sigma=sigma)
    return -x * gaussian_val / (sigma**2)

@njit
def gaussian2(P, sigma):
    gaussian_val_0 = gaussian1(P[0], sigma=sigma)
    gaussian_val_1 = gaussian1(P[1], sigma=sigma)
    return gaussian_val_0 * gaussian_val_1

@njit
def grad_gaussian2(P, sigma):
    gaussian_val_0 = gaussian1(P[0], sigma=sigma)
    gaussian_val_1 = gaussian1(P[1], sigma=sigma)
    grad_gaussian_0 = grad_gaussian1(P[0], sigma=sigma, gaussian_val=gaussian_val_0)
    grad_gaussian_1 = grad_gaussian1(P[1], sigma=sigma, gaussian_val=gaussian_val_1)
    return np.array([grad_gaussian_0 * gaussian_val_1,
                     grad_gaussian_1 * gaussian_val_0])