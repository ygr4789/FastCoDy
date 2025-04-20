import numpy as np
from scipy.linalg import svd
import scipy.sparse as sp
from numba import njit

from .kron import kron_dphidX_eye3

@njit
def dpsi_arap_dS(S, params):
    """
    Gradient of ARAP energy w.r.t. singular values.
    S: (3,)
    params: (1,)
    Returns: (3,)
    """
    stiffness = params[0]
    return 2.0 * stiffness * (S - 1.0)

@njit
def dpsi_stretch_dF(F, dpsi_func, params):
    """
    Compute the derivative of stretch energy wrt F
    Returns: (9,)
    """
    F = F.astype(np.float64)
    U, S, Vh = np.linalg.svd(F)
    V = Vh.T

    if np.prod(S) <= -1e-10:
        S = np.abs(S)
    if np.linalg.det(U) <= 0:
        U[:, 2] *= -1
    if np.linalg.det(V) <= 0:
        V[:, 2] *= -1

    Plam = dpsi_func(S, params)
    dF = (U @ np.diag(Plam) @ V.T).T.flatten()
    return dF

@njit
def linear_tet_arap_dq(q, element, dphidX, params, volume):
    qe = np.empty(12, dtype=np.float64)
    for i in range(4):
        qe[3*i:3*i+3] = q[3*element[i]:3*element[i]+3]
    B = kron_dphidX_eye3(dphidX)
    F = (B @ qe).reshape((3, 3)).T
    dF = dpsi_stretch_dF(F, dpsi_arap_dS, params)
    out = B.T @ dF * volume
    return out

@njit
def linear_tetmesh_arap_dq(V, E, q, dphidX, volume, params):
    num_tets = E.shape[0]
    N = V.shape[0]
    out = np.zeros(3 * N)

    for t in range(num_tets):
        element = E[t]
        dphi_matrix = dphidX[t].reshape(3, 4)
        param_t = params[t]
        H = linear_tet_arap_dq(q, element, dphi_matrix, param_t, volume[t])

        for i in range(4):
            idx = element[i]
            out[3 * idx: 3 * idx + 3] += H[3 * i: 3 * i + 3]

    return out