"""
Wenhua Bao: 2512664
Zhenfan Song: 2864671
Kexin Wang: 2540047

"""
import numpy as np
from PIL import Image
from scipy.signal import convolve2d
from scipy.ndimage import convolve
from scipy import interpolate

######################
# Basic Lucas-Kanade #
######################

def compute_derivatives(im1, im2):
    """Compute dx, dy and dt derivatives.
    
    Args:
        im1: first image
        im2: second image
    
    Returns:
        Ix, Iy, It: derivatives of im1 w.r.t. x, y and t
    """
    assert im1.shape == im2.shape
    
    Ix = np.empty_like(im1)
    Iy = np.empty_like(im1)
    It = np.empty_like(im1)

    #
    # Your code here
    #
    fy = np.array([[-0.5, 0, 0.5]])
    Iy = convolve(im1, fy, mode='mirror')
    fx = fy.T
    Ix = convolve(im2, fx, mode='mirror')
    It = im2 - im1
    assert Ix.shape == im1.shape and \
           Iy.shape == im1.shape and \
           It.shape == im1.shape

    return Ix, Iy, It

def compute_motion(Ix, Iy, It, patch_size=15, aggregate="const", sigma=2):
    """Computes one iteration of optical flow estimation.
    
    Args:
        Ix, Iy, It: image derivatives w.r.t. x, y and t
        patch_size: specifies the side of the square region R in Eq. (1)
        aggregate: 0 or 1 specifying the region aggregation region
        sigma: if aggregate=='gaussian', use this sigma for the Gaussian kernel
    Returns:
        u: optical flow in x direction
        v: optical flow in y direction
    
    All outputs have the same dimensionality as the input
    """
    assert Ix.shape == Iy.shape and \
            Iy.shape == It.shape

    u = np.empty_like(Ix)
    v = np.empty_like(Iy)

    #
    # Your code here
    #
    row, colon = Ix.shape[0], Ix.shape[1]

    w = int(patch_size / 2)  # the radius of the patch
    padded_Ix = np.pad(Ix, w, mode='reflect').copy()
    padded_Iy = np.pad(Iy, w, mode='reflect').copy()
    padded_It = np.pad(It, w, mode='reflect').copy()

    a = np.zeros((2, 2))
    b = np.zeros((2, 1))
    d = np.zeros((2, 1))
    i = 0
    for i in range(row):
        j = 0
        for j in range(colon):
            a[0, 0] = np.sum(padded_Ix[i:i + patch_size, j:j + patch_size] ** 2)
            a[1, 1] = np.sum(padded_Iy[i:i + patch_size, j:j + patch_size] ** 2)
            a[0, 1] = a[1, 0] = np.sum(padded_Ix[i:i + patch_size, j:j + patch_size] * padded_Iy[i:i + patch_size,
                                                                                       j:j + patch_size])  # a is A matrix
            b[0, 0] = -np.sum(
                padded_Ix[i:i + patch_size, j:j + patch_size] * padded_It[i:i + patch_size, j:j + patch_size])
            b[1, 0] = -np.sum(padded_Iy[i:i + patch_size, j:j + patch_size] * padded_It[i:i + patch_size,
                                                                              j:j + patch_size])  # b is B matrix
            # I = np.concatenate((Ix, Iy))
            d = np.dot(np.linalg.inv(a), b).copy()
            u[i, j] = d[0, 0]
            v[i, j] = d[1, 0]
    # print(aggregate)
    if aggregate == 'gaussian':
        gk = gaussian_kernel(aggregate, sigma)
        u = convolve2d(u, gk, mode='same')
        v = convolve2d(v, gk, mode='same')
    
    assert u.shape == Ix.shape and \
            v.shape == Ix.shape
    return u, v

def warp(im, u, v):
    """Warping of a given image using provided optical flow.
    
    Args:
        im: input image
        u, v: optical flow in x and y direction
    
    Returns:
        im_warp: warped image (of the same size as input image)
    """
    assert im.shape == u.shape and \
            u.shape == v.shape
    
    im_warp = np.empty_like(im)
    #
    # Your code here
    #
    h, w = im.shape
    points = np.empty((h * w, 2))

    for i in range(h):
        for j in range(w):
            points[i * w + j][0] = i - u[i, j]
            points[i * w + j][1] = j - v[i, j]

    values = im.flatten()
    xi, yi = np.mgrid[0:h, 0:w]

    im_warp = interpolate.griddata(points, values, (xi, yi), method='linear', fill_value=0)

    assert im_warp.shape == im.shape
    return im_warp

def compute_cost(im1, im2):
    """Implementation of the cost minimised by Lucas-Kanade."""
    assert im1.shape == im2.shape

    d = 0.0
    #
    # Your code here
    #
    a = im1 - im2
    d = np.mean([k ** 2 for k in a])

    assert isinstance(d, float)
    return d

####################
# Gaussian Pyramid #
####################

#
# this function implementation is intentionally provided
#
def gaussian_kernel(fsize, sigma):
    """
    Define a Gaussian kernel

    Args:
        fsize: kernel size
        sigma: deviation of the Guassian

    Returns:
        kernel: (fsize, fsize) Gaussian (normalised) kernel
    """

    _x = _y = (fsize - 1) / 2
    x, y = np.mgrid[-_x:_x + 1, -_y:_y + 1]
    G = np.exp(-0.5 * (x**2 + y**2) / sigma**2)

    return G / G.sum()

def downsample_x2(x, fsize=5, sigma=1.4):
    """
    Downsampling an image by a factor of 2
    Hint: Don't forget to smooth the image beforhand (in this function).

    Args:
        x: image as numpy array (H x W)
        fsize and sigma: parameters for Guassian smoothing
                         to apply before the subsampling
    Returns:
        downsampled image as numpy array (H/2 x W/2)
    """

    g_k = gaussian_kernel(fsize, sigma)
    x = convolve2d(x, g_k)

    return x[::2, ::2]

def gaussian_pyramid(img, nlevels=3, fsize=5, sigma=1.4):
    '''
    A Gaussian pyramid is a sequence of downscaled images
    (here, by a factor of 2 w.r.t. the previous image in the pyramid)

    Args:
        img: face image as numpy array (H * W)
        nlevels: num of level Gaussian pyramid, in this assignment we will use 3 levels
        fsize: gaussian kernel size, in this assignment we will define 5
        sigma: sigma of guassian kernel, in this assignment we will define 1.4

    Returns:
        GP: list of gaussian downsampled images in ascending order of resolution
    '''
    
    #
    # Your code here
    #
    pyramid = [img]
    for i in range(0, nlevels - 1):
        pyramid.append(downsample_x2(pyramid[i], fsize, sigma))

    return pyramid

###############################
# Coarse-to-fine Lucas-Kanade #
###############################

def coarse_to_fine(im1, im2, pyramid1, pyramid2, n_iter=3):
    """Implementation of coarse-to-fine strategy
    for optical flow estimation.
    
    Args:
        im1, im2: first and second image
        pyramid1, pyramid2: Gaussian pyramids corresponding to im1 and im2
        n_iter: number of refinement iterations
    
    Returns:
        u: OF in x direction
        v: OF in y direction
    """

    assert im1.shape == im2.shape
    
    u = np.zeros_like(im1)
    v = np.zeros_like(im1)


    Ix, Iy, It = compute_derivatives(im1, im2)
    u, v = compute_motion(Ix, Iy, It, patch_size=15, aggregate="const", sigma=2)
    d = compute_cost(im1, im2) / 4
    # Ix_p, Iy_p, It_p = compute_derivatives(pyramid1[2], pyramid2[2])
    # u_p, v_p = compute_motion(Ix_p, Iy_p, It_p, patch_size=15, aggregate="const", sigma=2)
    for i in range(n_iter - 1):
        u = 2 * (u + d)
        v = 2 * (v + d)
    u = np.array(u)
    v = np.array(v)
    assert u.shape == im1.shape and v.shape == im1.shape
    return u, v

