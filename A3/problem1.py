"""
Wenhua Bao: 2512664
Zhenfan Song: 2864671
Kexin Wang: 2540047

"""
import numpy as np
import scipy.ndimage
from scipy.ndimage import convolve, maximum_filter


def gauss2d(sigma, fsize):
    """ Create a 2D Gaussian filter

    Args:
        sigma: width of the Gaussian filter
        fsize: (w, h) dimensions of the filter
    Returns:
        *normalized* Gaussian filter as (h, w) np.array
    """
    m, n = fsize
    x = np.arange(-m / 2 + 0.5, m / 2)
    y = np.arange(-n / 2 + 0.5, n / 2)
    xx, yy = np.meshgrid(x, y, sparse=True)
    g = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return g / np.sum(g)


def derivative_filters():
    """ Create derivative filters for x and y direction

    Returns:
        fx: derivative filter in x direction
        fy: derivative filter in y direction
    """
    fx = np.array([[0.5, 0, -0.5]])
    fy = fx.transpose()
    return fx, fy


def compute_hessian(img, gauss, fx, fy):
    """ Compute elements of the Hessian matrix

    Args:
        img:
        gauss: Gaussian filter
        fx: derivative filter in x direction
        fy: derivative filter in y direction

    Returns:
        I_xx: (h, w) np.array of 2nd derivatives in x direction
        I_yy: (h, w) np.array of 2nd derivatives in y direction
        I_xy: (h, w) np.array of 2nd derivatives in x-y direction
    """

    #
    smoothed_img = convolve(img, gauss, mode='mirror')
    I_xx = convolve(convolve(smoothed_img, fx, mode='mirror'), fx, mode='mirror')
    I_yy = convolve(convolve(smoothed_img, fy, mode='mirror'), fy, mode='mirror')
    I_xy = convolve(convolve(smoothed_img, fx, mode='mirror'), fy, mode='mirror')

    return I_xx, I_yy, I_xy
    #


def compute_criterion(I_xx, I_yy, I_xy, sigma):
    """ Compute criterion function

    Args:
        I_xx: (h, w) np.array of 2nd derivatives in x direction
        I_yy: (h, w) np.array of 2nd derivatives in y direction
        I_xy: (h, w) np.array of 2nd derivatives in x-y direction
        sigma: scaling factor

    Returns:
        criterion: (h, w) np.array of scaled determinant of Hessian matrix
    """

    #
    determinant = I_xx * I_yy - I_xy ** 2
    creterion = determinant * sigma ** 4

    return creterion
    #


def nonmaxsuppression(criterion, threshold):
    """ Apply non-maximum suppression to criterion values
        and return Hessian interest points

        Args:
            criterion: (h, w) np.array of criterion function values
            threshold: criterion threshold
        Returns:
            rows: (n,) np.array with y-positions of interest points
            cols: (n,) np.array with x-positions of interest points
    """

    #
    max_local = maximum_filter(criterion, size=5)
    point = (max_local > threshold)
    rows, cols = np.nonzero(point)
    return rows, cols
    #
