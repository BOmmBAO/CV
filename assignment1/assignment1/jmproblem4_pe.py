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
  sizeX = fsize[0]
  sizeY = fsize[1]
    
  g = np.zeros((sizeX,sizeY))
  imageCenterX = (sizeX-1)/2
  imageCenterY = (sizeY-1)/2
  for i in range(sizeX):
      for j in range(sizeY):
          g[i,j] = 1/(2*math.pi*sigma*sigma)*np.exp(-((i-imageCenterX)*(i-imageCenterX)+(j-imageCenterY)*(j-imageCenterY))/(2*sigma*sigma))
  g = g/np.sum(np.abs(g))
  return(g)
  #
  # You code here
  #


def createfilters():
  """
  Returns:
    fx, fy: filters as described in the problem assignment
  """
  f1 = gauss2d(0.9, [3,1])
  f2 = np.array([1/2,0,-1/2])
  fx = f2*f1
  f3 = np.array([[1/2],[0],[-1/2]])
  fy = np.transpose(f1)*f3
  return fx,fy
  
  #
  # You code here
  #


def filterimage(I, fx, fy):
  """ Filter the image with the filters fx, fy.
  You may use the ndimage.convolve scipy-function.

  Args:
    I: a (H,W) numpy array storing image data
    fx, fy: filters

  Returns:
    Ix, Iy: images filtered by fx and fy respectively
  """
  Ix = ndimage.convolve(I,fx, mode='constant', cval=0.0)
  Iy = ndimage.convolve(I,fy, mode='constant', cval=0.0)
  #
  # You code here
  #
  return Ix,Iy

def detectedges(Ix, Iy, thr):
  """ Detects edges by applying a threshold on the image gradient magnitude.

  Args:
    Ix, Iy: filtered images
    thr: the threshold value

  Returns:
    edges: (H,W) array that contains the magnitude of the image gradient at edges and 0 otherwise
  """
  magnitudeImage = np.sqrt(Ix*Ix+Iy*Iy)
  
  magnitudeImage[magnitudeImage < thr] = 0 
  #
  # You code here
  #
  return magnitudeImage

def nonmaxsupp(edges, Ix, Iy):
  """ Performs non-maximum suppression on an edge map.

  Args:
    edges: edge map containing the magnitude of the image gradient at edges and 0 otherwise
    Ix, Iy: filtered images

  Returns:
    edges2: edge map where non-maximum edges are suppressed
  """
  theta = np.arctan2(Ix,Iy)
  H = edges.copy()
  
  for i in range(theta.shape[0]):
    for j in range(theta.shape[1]):
        if edges[i,j]>0:
            if theta[i,j] >= -90/180*np.pi and theta[i,j] <= -67.5/180*np.pi:
                if i+1>= edges.shape[0] or j+1>=edges.shape[1] or i-1<0 or j-1<0:
                    continue
                elif H[i,j]<H[i,j+1] or H[i,j]<H[i,j-1]:
                    edges[i,j]=0
            elif theta[i,j] > 67.5/180*np.pi and theta[i,j] <= 90/180*np.pi:
                if i+1>= edges.shape[0] or j+1>=edges.shape[1] or i-1<0 or j-1<0:
                    continue
                elif H[i,j]<H[i,j+1] or H[i,j]<H[i,j-1]:
                    edges[i,j]=0
            elif theta[i,j] > -22.5/180*np.pi and theta[i,j] <= 22.5/180*np.pi:
                if i+1>= edges.shape[0] or j+1>=edges.shape[1] or i-1<0 or j-1<0:
                    continue
                elif H[i,j]<H[i+1,j] or H[i,j]<H[i+1,j]:
                    edges[i,j]=0
            elif theta[i,j] > 22.5/180*np.pi and theta[i,j] <= 67.5/180*np.pi:
                if i+1>= edges.shape[0] or j+1>=edges.shape[1] or i-1<0 or j-1<0:
                    continue
                elif H[i,j]<H[i-1,j+1] or H[i,j]<H[i+1,j-1]:
                    edges[i,j]=0
            elif theta[i,j] >= -67.5/180*np.pi and theta[i,j] <= -22.5/180*np.pi:
                if i+1>= edges.shape[0] or j+1>=edges.shape[1] or i-1<0 or j-1<0:
                    continue
                elif H[i,j]<H[i-1,j-1] or H[i,j]<H[i+1,j+1]:
                    edges[i,j]=0
  return(edges)