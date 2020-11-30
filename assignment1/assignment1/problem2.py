import numpy as np
from scipy.ndimage import convolve


def loaddata(path):
    """ Load bayerdata from file

    Args:
        Path of the .npy file
    Returns:
        Bayer data as numpy array (H,W)
    """

    return np.load(path)


def separatechannels(bayerdata):
    """ Separate bayer data into RGB channels so that
    each color channel retains only the respective
    values given by the bayer pattern and missing values
    are filled with zero

    Args:
        Numpy array containing bayer data (H,W)
    Returns:
        red, green, and blue channel as numpy array (H,W)
    """

    idate = bayerdata.shape[0]
    jdate = bayerdata.shape[1]
    red = np.zeros((idate,jdate))
    green = np.zeros((idate,jdate))
    blue = np.zeros((idate,jdate))

    for i in range(idate):
        for j in range(jdate):
            if (i+j)%2==0:
                green[i,j]=bayerdata[i,j]
            elif i%2==0:
                red[i,j]=bayerdata[i,j]
            else:
                blue[i,j]=bayerdata[i,j]

    return red, green, blue

def assembleimage(r, g, b):
    """ Assemble separate channels into image

    Args:
        red, green, blue color channels as numpy array (H,W)
    Returns:
        Image as numpy array (H,W,3)
    """

    return np.dstack([r,g,b])

def interpolate(r, g, b):
    """ Interpolate missing values in the bayer pattern
    by using bilinear interpolation

    Args:
        red, green, blue color channels as numpy array (H,W)
    Returns:
        Interpolated image as numpy array (H,W,3)
    """

    f1 = np.array([[0,1/4,0],[1/4,0,1/4],[0,1/4,0]])
    f2 = np.array([[0,1/4,0],[0,0,0],[0,1/4,0]])

    g = convolve(g,f1, mode = 'nearest')
    r = convolve(r,f2, mode = 'nearest')
    b = convolve(b,f2, mode = 'nearest')

    return np.dstack([r,g,b])