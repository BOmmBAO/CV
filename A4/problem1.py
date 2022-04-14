"""
Wenhua Bao: 2512664
Zhenfan Song: 2864671
Kexin Wang: 2540047

"""
import numpy as np


def condition_points(points):
    """ Conditioning: Normalization of coordinates for numeric stability 
    by substracting the mean and dividing by half of the component-wise
    maximum absolute value.
    Args:
        points: (l, 3) numpy array containing unnormalized homogeneous coordinates.

    Returns:
        ps: (l, 3) numpy array containing normalized points in homogeneous coordinates.
        T: (3, 3) numpy array, transformation matrix for conditioning
    """
    t = np.mean(points, axis=0)[:-1]
    s = 0.5 * np.max(np.abs(points), axis=0)[:-1]
    T = np.eye(3)
    T[0:2,2] = -t
    T[0:2, 0:3] = T[0:2, 0:3] / np.expand_dims(s, axis=1)
    ps = points @ T.T
    return ps, T


def enforce_rank2(A):
    """ Enforces rank 2 to a given 3 x 3 matrix by setting the smallest
    eigenvalue to zero.
    Args:
        A: (3, 3) numpy array, input matrix

    Returns:
        A_hat: (3, 3) numpy array, matrix with rank at most 2
    """

    u, d, v = np.linalg.svd(A)
    d[2] = 0
    A_hat = u * d @ v
    return A_hat



def compute_fundamental(p1, p2):
    """ Computes the fundamental matrix from conditioned coordinates.
    Args:
        p1: (n, 3) numpy array containing the conditioned coordinates in the left image
        p2: (n, 3) numpy array containing the conditioned coordinates in the right image

    Returns:
        F: (3, 3) numpy array, fundamental matrix
    """

    n = p1.shape[0]
    M = np.empty((n, 9))
    for i in range(n):
        x1 = p1[i][0]
        x2 = p2[i][0]
        y1 = p1[i][1]
        y2 = p2[i][1]
        M[i] = [x2 * x1, y2 * x1, x1, x2 * y1, y2 * y1, y1, x2, y2, 1]
    u, d, v = np.linalg.svd(M)
    F1 = v.T[:, -1].reshape(3, 3)
    F_hat = enforce_rank2(F1)
    return F_hat



def eight_point(p1, p2):
    """ Computes the fundamental matrix from unconditioned coordinates.
    Conditions coordinates first.
    Args:
        p1: (n, 3) numpy array containing the unconditioned homogeneous coordinates in the left image
        p2: (n, 3) numpy array containing the unconditioned homogeneous coordinates in the right image

    Returns:
        F: (3, 3) numpy array, fundamental matrix with respect to the unconditioned coordinates
    """

    ps1, T1 = condition_points(p1)
    ps2, T2 = condition_points(p2)
    F_hat = compute_fundamental(ps1, ps2)
    F = T1.T @ F_hat @ T2
    return F




def draw_epipolars(F, p1, img):
    """ Computes the coordinates of the n epipolar lines (X1, Y1) on the left image border and (X2, Y2)
    on the right image border.
    Args:
        F: (3, 3) numpy array, fundamental matrix 
        p1: (n, 2) numpy array, cartesian coordinates of the point correspondences in the image
        img: (H, W, 3) numpy array, image data

    Returns:
        X1, X2, Y1, Y2: (n, ) numpy arrays containing the coordinates of the n epipolar lines
            at the image borders
    """

    p1_h = np.concatenate([p1, np.ones((p1.shape[0], 1))], axis=1)
    H = img.shape[0]
    W = img.shape[1]
    n = p1.shape[0]
    l = F @ p1_h.T

    a = l[0]
    b = l[1]
    c = l[2]

    X1 = np.full((n,), 0)
    Y1 = -(a * X1 + c) / b
    X2 = np.full((n,), W)
    Y2 = -(a * X2 + c) / b

    return X1, X2, Y1, Y2



def compute_residuals(p1, p2, F):
    """
    Computes the maximum and average absolute residual value of the epipolar constraint equation.
    Args:
        p1: (n, 3) numpy array containing the homogeneous correspondence coordinates from image 1
        p2: (n, 3) numpy array containing the homogeneous correspondence coordinates from image 2
        F:  (3, 3) numpy array, fundamental matrix

    Returns:
        max_residual: maximum absolute residual value
        avg_residual: average absolute residual value
    """

    residual = np.linalg.norm(p1 @ F @ p2.T, axis=1)
    max_residual = np.max(residual)
    avg_residual = np.average(residual)
    return max_residual, avg_residual


def compute_epipoles(F):
    """ Computes the cartesian coordinates of the epipoles e1 and e2 in image 1 and 2 respectively.
    Args:
        F: (3, 3) numpy array, fundamental matrix

    Returns:
        e1: (2, ) numpy array, cartesian coordinates of the epipole in image 1
        e2: (2, ) numpy array, cartesian coordinates of the epipole in image 2
    """

    def null_space(A, rcond=None):
        u, s, vh = np.linalg.svd(A, full_matrices=True)
        M, N = u.shape[0], vh.shape[1]
        if rcond is None:
            rcond = np.finfo(s.dtype).eps * max(M, N)
        tol = np.amax(s) * rcond
        num = np.sum(s > tol, dtype=int)
        Q = vh[num:, :].T.conj()
        return Q

    right_null = null_space(F)

    e2 = np.empty(2)
    e2[0] = right_null[0, 0] / right_null[2, 0]
    e2[1] = right_null[1, 0] / right_null[2, 0]

    rank = np.linalg.matrix_rank(F)
    U, s, V = np.linalg.svd(F, full_matrices=True)
    t_U_A = np.transpose(U)
    nrow = t_U_A.shape[0]
    left_null = t_U_A[rank:nrow, :]
    e1 = np.empty(2)
    e1[0] = left_null[0, 0] / left_null[0, 2]
    e1[1] = left_null[0, 1] / left_null[0, 2]

    return e1, e2
