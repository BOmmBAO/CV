from copy import deepcopy
import numpy as np
from PIL import Image
from scipy.special import binom
#from scipy.ndimage import convolve
from scipy.signal import convolve2d


def loadimg(path):
    """ Load image file

    Args:
        path: path to image file
    Returns:
        image as (H, W) np.array normalized to [0, 1]
    """

    #
    # You code here
    #
    image = np.array(Image.open(path),dtype=float)/255
    #print(image.shape)
    return image


def gauss2d(sigma, fsize):
    """ Create a 2D Gaussian filter

    Args:
        sigma: width of the Gaussian filter
        fsize: (W, H) dimensions of the filter
    Returns:
        *normalized* Gaussian filter as (H, W) np.array
    """

    #
    # You code here
    #
    h, w = fsize
    kernel = np.zeros([h, w])
    for row in range(h):
        for col in range(w):
            kernel[row, col] = (1 / (2 * np.pi * (sigma ** 2))) * \
                               np.exp(
                                   -1 * ((row - int(w / 2)) ** 2 + (col - int(h / 2)) ** 2) / (2 * sigma ** 2))
    nor_kernel = kernel/np.max(kernel)
    #print(nor_kernel)
    return nor_kernel




def binomial2d(fsize):
    """ Create a 2D binomial filter

    Args:
        fsize: (W, H) dimensions of the filter
    Returns:
        *normalized* binomial filter as (H, W) np.array
    """

    #
    # You code here
    #
    bin1d = np.zeros(fsize[0])
    for i in range(fsize[0]):
        bin1d[i] = binom(fsize[0]-1,i)
    bin1d = np.diag(bin1d)
    bin2d = np.dot(bin1d,bin1d.T)
    bin2d = bin2d/np.max(bin2d)
    #print(bin2d)
    return bin2d


def downsample2(img, f):
    """ Downsample image by a factor of 2
    Filter with Gaussian filter then take every other row/column

    Args:
        img: image to downsample
        f: 2d filter kernel
    Returns:
        downsampled image as (H, W) np.array
    """

    #
    # You code here
    #
    downsample = np.zeros((np.array(img.shape) / 2).astype(int))
    for row in range(downsample.shape[0]):
        for col in range(downsample.shape[1]):
            downsample[row, col] = img[2 * row, 2 * col]
    return downsample


def upsample2(img, f):
    """ Upsample image by factor of 2

    Args:
        img: image to upsample
        f: 2d filter kernel
    Returns:
        upsampled image as (H, W) np.array
    """

    #
    # You code here
    #
    upsample = np.zeros((np.array(img.shape) * 2).astype(int))
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            upsample[row*2, col*2] = img[row, col]
    return upsample


def gaussianpyramid(img, nlevel, f):
    """ Build Gaussian pyramid from image

    Args:
        img: input image for Gaussian pyramid
        nlevel: number of pyramid levels
        f: 2d filter kernel
    Returns:
        Gaussian pyramid, pyramid levels as (H, W) np.array
        in a list sorted from fine to coarse
    """

    #
    # You code here
    #
    GP = []
    currentImg = img
    GP.append(currentImg)
    for level in range(nlevel - 1):
        currentImg = convolve2d(currentImg, f, mode='same')
        currentImg = downsample2(currentImg, f)
        GP.append(currentImg)
    return GP


def laplacianpyramid(gpyramid, f):
    """ Build Laplacian pyramid from Gaussian pyramid

    Args:
        gpyramid: Gaussian pyramid
        f: 2d filter kernel
    Returns:
        Laplacian pyramid, pyramid levels as (H, W) np.array
        in a list sorted from fine to coarse
    """

    #
    # You code here
    #
    LP = []
    gpyramid = np.array(gpyramid, dtype=object)
    key = len(gpyramid)
    for i in range(key-1):
        temp_pyr = upsample2(gpyramid[i+1],f)
        temp_lap = np.subtract(deepcopy(gpyramid[i])/np.max(gpyramid[i]), temp_pyr/np.max(temp_pyr))
        LP.append(temp_lap)
    LP.append(gpyramid[-1]/np.max(gpyramid[-1]))
    return LP




def reconstructimage(lpyramid, f):
    """ Reconstruct image from Laplacian pyramid

    Args:
        lpyramid: Laplacian pyramid
        f: 2d filter kernel
    Returns:
        Reconstructed image as (H, W) np.array clipped to [0, 1]
    """

    #
    # You code here
    #
    x = len(lpyramid)
    pyramid_temp = deepcopy(lpyramid)
    for i in range(x - 1):
        pyramid_temp[x - i - 2] = upsample2(pyramid_temp[x - i - 1], f)
        pyramid_temp[x - i - 2] /= np.max(pyramid_temp[x - i - 2])
        img_temp = lpyramid[x - i - 2]
        pyramid_temp[x - i - 2] += img_temp
    return pyramid_temp[0]


def amplifyhighfreq(lpyramid, l0_factor=1.0, l1_factor=1.0):
    """ Amplify frequencies of the finest two layers of the Laplacian pyramid

    Args:
        lpyramid: Laplacian pyramid
        l0_factor:

        l1_factor: amplification factor for the second finest pyramid level
    Returns:
        Amplified Laplacian pyramid, data format like input pyramid
    """

    key = len(lpyramid)
    lpyramid[key - 1] *= l0_factor
    lpyramid[key - 2] *= l1_factor
    return lpyramid



def createcompositeimage(pyramid):
    """ Create composite image from image pyramid
    Arrange from finest to coarsest image left to right, pad images with
    zeros on the bottom to match the hight of the finest pyramid level.
    Normalize each pyramid level individually before adding to the composite image

    Args:
        pyramid: image pyramid
    Returns:
        composite image as (H, W) np.array
    """

    #
    # You code here
    #
    length = len(pyramid)
    sizey = 0
    x, y = np.shape(pyramid[0])
    for i in range(0, length):
       x_t, y_t = np.shape(pyramid[i])
       sizey += y_t
    img1 = np.zeros((x, sizey))
    y_s = 0
    for i in range(0 , length):
        x, y = np.shape(pyramid[i])
        img_temp = pyramid[i]
        img1[0:x , y_s : y_s + y] = img_temp
        y_s += y

    return img1