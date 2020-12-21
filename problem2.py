import numpy as np
import os
from PIL import Image


def load_faces(path, ext=".pgm"):
    """Load faces into an array (N, H, W),
    where N is the number of face images and
    H, W are height and width of the images.
    
    Hint: os.walk() supports recursive listing of files 
    and directories in a path
    
    Args:
        path: path to the directory with face images
        ext: extension of the image files (you can assume .pgm only)
    
    Returns:
        imgs: (N, H, W) numpy array
    """

    image_array = np.zeros((760,96,84))
    n=0
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            image_array[n,:,:] = Image.open(os.path.join(root, name))
            n=n+1
    return image_array


def vectorize_images(imgs):
    """Turns an  array (N, H, W),
    where N is the number of face images and
    H, W are height and width of the images into
    an (N, M) array where M=H*W is the image dimension.
    
    Args:
        imgs: (N, H, W) numpy array
    
    Returns:
        x: (N, M) numpy array
    """

    N = np.shape(imgs)[0]
    H = np.shape(imgs)[1]
    W = np.shape(imgs)[2]
    return np.reshape(imgs,(N,H*W))


def compute_pca(X):
    """PCA implementation
    
    Args:
        X: (N, M) an numpy array with N M-dimensional features
    
    Returns:
        mean_face: (M,) numpy array representing the mean face
        u: (M, M) numpy array, bases with D principal components
        cumul_var: (N, ) numpy array, corresponding cumulative variance
    """

    mean_face = np.mean(X,axis=0)
    u, s, vh = np.linalg.svd(np.transpose(X-mean_face), full_matrices=True)
    lam = s**2/X.shape[0]
    cumul_val = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        cumul_val[i] = np.sum(lam[:i+1])
    return mean_face,u,cumul_val

def basis(u, cumul_var, p = 0.5):
    """Return the minimum number of basis vectors 
    from matrix U such that they account for at least p percent
    of total variance.
    
    Hint: Do the singular values really represent the variance?
    
    Args:
        u: (M, M) numpy array containing principal components.
        For example, i'th vector is u[:, i]
        cumul_var: (N, ) numpy array, variance along the principal components.
    
    Returns:
        v: (M, D) numpy array, contains M principal components from N
        containing at most p (percentile) of the variance.
    
    """

    summation = cumul_var[-1]
    for i in range(cumul_var.shape[0]):
        if cumul_var[i]>p*summation:
            D = i
            break
    return u[:,0:D+1]



def compute_coefficients(face_image, mean_face, u):
    """Computes the coefficients of the face image with respect to
    the principal components u after projection.
    
    Args:
        face_image: (M, ) numpy array (M=h*w) of the face image a vector
        mean_face: (M, ) numpy array, mean face as a vector
        u: (M, D) numpy array containing D principal components. 
        For example, (:, 1) is the second component vector.
    
    Returns:
        a: (D, ) numpy array, containing the coefficients
    """

    a = np.zeros(u.shape[1])
    for i in range(u.shape[1]):
        a[i] = np.matmul(u[:,i],(face_image-mean_face))

    return a

def reconstruct_image(a, mean_face, u):
    """Reconstructs the face image with respect to
    the first D principal components u.
    
    Args:
        a: (D, ) numpy array containings the image coefficients w.r.t
        the principal components u
        mean_face: (M, ) numpy array, mean face as a vector
        u: (M, D) numpy array containing D principal components. 
        For example, (:, 1) is the second component vector.
    
    Returns:
        image_out: (M, ) numpy array, projected vector of face_image on 
        principal components
    """

    image_out = np.zeros(u.shape[0])
    for i in range(u.shape[1]):
        image_out = image_out+a[i]*u[:,i]
    image_out = image_out+mean_face

    return image_out

def compute_similarity(Y, x, u, mean_face):
    """Compute the similarity of an image x to the images in Y
    based on the cosine similarity.

    Args:
        Y: (N, M) numpy array with N M-dimensional features
        x: (M, ) image we would like to retrieve
        u: (M, D) bases vectors. Note, we already assume D has been selected.
        mean_face: (M, ) numpy array, mean face as a vector

    Returns:
        sim: (N, ) numpy array containing the cosine similarity values
    """
    
    dim_redY = np.matmul(u.T,(Y-mean_face).T)
    dim_redx = np.matmul(u.T,x-mean_face)
    sim = np.zeros(Y.shape[0])
    for i in range(Y.shape[0]):
        sim[i] = np.dot(dim_redx,dim_redY[:,i])/(np.linalg.norm(dim_redx)*np.linalg.norm(dim_redY[:,i]))

    return sim

def search(Y, x, u, mean_face, top_n):
    """Search for the top most similar images
    based on a given number of components in their PCA decomposition.
    
    Args:
        Y: (N, M) numpy array with N M-dimensional features
        x: (M, ) numpy array, image we would like to retrieve
        u: (M, D) numpy arrray, bases vectors. Note, we already assume D has been selected.
        mean_face: (M, ) numpy array, mean face as a vector
        top_n: integer, return top_n closest images in L2 sense.
    
    Returns:
        Y: (top_n, M) numpy array containing the top_n most similar images
        sorted by similarity
    """

    sim = compute_similarity(Y, x, u, mean_face)
    
    idx = (-sim).argsort()[:top_n]
    return Y[idx,:]

def interpolate(x1, x2, u, mean_face, n):
    """Search for the top most similar images
    based on a given number of components in their PCA decomposition.
    
    Args:
        x1: (M, ) numpy array, the first image
        x2: (M, ) numpy array, the second image
        u: (M, D) numpy array, bases vectors. Note, we already assume D has been selected.
        mean_face: (M, ) numpy array, mean face as a vector
        n: number of interpolation steps (including x1 and x2)

    Hint: you can use np.linspace to generate n equally-spaced points on a line
    
    Returns:
        Y: (n, M) numpy arrray, interpolated results.
        The first dimension is in the index into corresponding
        image; Y[0] == project(x1, u); Y[-1] == project(x2, u)
    """

    a1 = compute_coefficients(x1, mean_face, u)
    a2 = compute_coefficients(x2,mean_face,u)
    step = np.linspace(a1, a2, num=n)
    Y = np.zeros((n,u.shape[0]))
    for i in range(n):
        Y[i,:] = reconstruct_image(step[i,:], mean_face, u)
        
    return Y