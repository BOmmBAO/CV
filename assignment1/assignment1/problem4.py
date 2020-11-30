import math
import numpy as np
from scipy import ndimage


def gauss2d(sigma, fsize):
  """
  Args:
    sigma: width of the Gaussian filter
    fsize: dimensions of the filter

  Returns:
    g: *normalized* Gaussian filter
  """

  xsize = fsize[0]
  ysize = fsize[1]
  xcenter = (xsize-1)/2
  ycenter = (ysize-1)/2

  gf = np.zeros((xsize, ysize))
  for i in range(xsize):
    for j in range(ysize):
      gf[i,j] = 1/(2*math.pi*sigma**2)*np.exp(-((i-xcenter)**2+(j-ycenter)**2)/(2*sigma**2))
  g = gf/np.sum(gf)

  return (g)


def createfilters():
  """
  Returns:
    fx, fy: filters as described in the problem assignment
  """

  c = np.array([[1/2, 0, -1/2]])
  g = gauss2d(0.9,[3,1])
  fx = c*g

  cT = np.transpose(c)
  gT = np.transpose(g)
  fy = gT*cT

  return fx,fy

def filterimage(I, fx, fy):
  """ Filter the image with the filters fx, fy.
  You may use the ndimage.convolve scipy-function.

  Args:
    I: a (H,W) numpy array storing image data
    fx, fy: filters

  Returns:
    Ix, Iy: images filtered by fx and fy respectively
  """

  Ix = ndimage.convolve(I,fx,mode='constant')
  Iy = ndimage.convolve(I,fy,mode='constant')

  return Ix,Iy

def detectedges(Ix, Iy, thr):
  """ Detects edges by applying a threshold on the image gradient magnitude.

  Args:
    Ix, Iy: filtered images
    thr: the threshold value

  Returns:
    edges: (H,W) array that contains the magnitude of the image gradient at edges and 0 otherwise
  """

  magnitude = np.sqrt(Ix**2 + Iy**2)
  magnitude[magnitude<thr]=0

  return magnitude

def nonmaxsupp(edges, Ix, Iy):
  """ Performs non-maximum suppression on an edge map.

  Args:
    edges: edge map containing the magnitude of the image gradient at edges and 0 otherwise
    Ix, Iy: filtered images

  Returns:
    edges2: edge map where non-maximum edges are suppressed
  """

  theta = np.arctan2(Ix,Iy)
  itheta = theta.shape[0]
  jtheta = theta.shape[1]
  ref_edges = np.copy(edges)

  for i in range(itheta):
    for j in range(jtheta):
      if i == 0 or i == itheta-1:
        continue
      if j == 0 or j == jtheta-1:
        continue

  # handle top-to-bottom edges: theta in [-90, -67.5] or (67.5, 90]

      if (theta[i,j] >= -90/180*math.pi and theta[i,j] <= -67.5/180*math.pi):
        if edges[i,j] < ref_edges[i+1,j] or edges[i,j] < ref_edges[i-1,j]:
          edges[i,j]=0
      if (theta[i,j] > 67.5/180*math.pi and theta[i,j] <= 90/180*math.pi):
        if edges[i, j] < ref_edges[i+1, j] or edges[i, j] < ref_edges[i-1, j]:
          edges[i, j] = 0

  # handle left-to-right edges: theta in (-22.5, 22.5]

      if (theta[i,j] > -22.5/180*math.pi and theta[i,j] <= 22.5/180*math.pi):
        if edges[i,j] < ref_edges[i,j+1] or edges[i,j] < ref_edges[i,j-1]:
          edges[i,j]=0

  # handle bottomleft-to-topright edges: theta in (22.5, 67.5]

      if (theta[i, j] > 22.5 / 180 * math.pi and theta[i, j] <= 67.5 / 180 * math.pi):
        if edges[i, j] < ref_edges[i+1, j-1] or edges[i, j] < ref_edges[i-1, j+1]:
          edges[i, j] = 0

  # handle topleft-to-bottomright edges: theta in [-67.5, -22.5]

      if (theta[i, j] >= -67.5 / 180 * math.pi and theta[i, j] <= -22.5 / 180 * math.pi):
        if edges[i, j] < ref_edges[i-1, j-1] or edges[i, j] < ref_edges[i+1, j+1]:
          edges[i, j] = 0

  return edges

# I tried at the beginning with the threshold=1, the output however is a total black image,
# which means the value of threshold is too high that none of the value in edges matrix can
# be retained but become 0. So i reduced the threshold till all the key edges appeared.
# At last i chose 0,1 as threshold.