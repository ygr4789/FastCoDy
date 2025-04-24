import numpy as np
from scipy.linalg import svd
import scipy.sparse as sp
from numba import njit, prange

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

@njit(parallel=True)
def linear_tetall_arap_dq(V, E, q, dphidX, volume, params):
    num_tets = E.shape[0]
    H_all = np.zeros((num_tets, 12), dtype=np.float64)

    for t in prange(num_tets):
        element = E[t]
        dphi_matrix = dphidX[t].reshape(3, 4)
        param_t = params[t]
        H_all[t] = linear_tet_arap_dq(q, element, dphi_matrix, param_t, volume[t])
        
    return H_all

def linear_tetmesh_arap_dq(V, E, q, dphidX, volume, params):
    H_all = linear_tetall_arap_dq(V, E, q, dphidX, volume, params)
    
    N = V.shape[0]
    out = np.zeros(3 * N, dtype=H_all.dtype)

    indices = (3 * E[..., None]) + np.array([0, 1, 2])  # shape (num_tets, 4, 3)
    indices = indices.reshape(-1)                       # (num_tets * 12,)
    values = H_all.reshape(-1)  
    np.add.at(out, indices, values)

    return out