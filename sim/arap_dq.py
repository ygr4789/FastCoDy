import numpy as np
from scipy.linalg import svd
import scipy.sparse as sp
from numba import njit

from .kron import kron_dphidX_eye3

# def dpsi_arap_dS(S, params):
#     """
#     Compute ARAP energy derivative with respect to singular values S.
#     """
#     stiffness = params[0]
#     return 2.0 * stiffness * (S - 1.0)

# def dpsi_stretch_dF(F, dpsi_func, params):
#     """
#     Compute dpsi/dF using SVD.
#     Args:
#         F: (3, 3) deformation gradient
#         dpsi_func: callable, computes dpsi w.r.t. singular values
#         params: parameters for dpsi_func
#     Returns:
#         dF: (9,) flattened matrix derivative
#     """
#     F = F.astype(np.float32)
#     U, S, Vh = svd(F)
#     V = Vh.T

#     # Fix inverted elements
#     if np.prod(S) <= -1e-10:
#         S = np.abs(S)
#     if np.linalg.det(U) <= 0:
#         U[:, 2] *= -1
#     if np.linalg.det(V) <= 0:
#         V[:, 2] *= -1

#     Plam = dpsi_func(S, params)  # (3,)
#     dF = (U @ np.diag(Plam) @ V.T).T.flatten()  # (9,)
#     return dF

# def linear_tet_arap_dq(q, element, dphidX, params, volume):
#     """
#     Compute per-element ARAP force derivative (energy gradient).
#     Args:
#         q: (3N,) global displacements
#         element: (4,) indices of vertices in this tet
#         dphidX: (3, 4) gradient of shape functions
#         params: (1,) stiffness
#         volume: float
#     Returns:
#         out: (12,) vector
#     """
#     # Gather local displacements
#     qe = np.concatenate([q[3*i:3*i+3] for i in element])  # (12,)
#     B = np.kron(dphidX, np.eye(3))  # (9, 12)
#     F = (B @ qe).reshape(3, 3)  # deformation gradient

#     dF = dpsi_stretch_dF(F, dpsi_arap_dS, params)  # (9,)
#     out = B.T @ dF * volume  # (12,)
#     return out
    
# def linear_tetmesh_arap_dq(V, E, q, dphidX, volume, params, func=None):
#     """
#     Assembles the ARAP derivative vector for all tetrahedra in the mesh.

#     Args:
#         V: (N, 3) vertex positions
#         E: (M, 4) tetrahedral element indices
#         q: (3N,) global displacement vector
#         dphidX: (M, 12) per-tet flattened shape gradients
#         volume: (M,) tetrahedral volumes
#         params: (M, K) or (M,) stiffness per tet
#         func: callback function that takes (H, e, out) to accumulate per-tet contribution
#     Returns:
#         out: (3N,) global assembled vector
#     """

#     num_tets = E.shape[0]
#     N = V.shape[0]
#     out = np.zeros(3 * N)

#     for t in range(num_tets):
#         element = E[t]
#         dphi_matrix = dphidX[t].reshape(3, 4, order='F')
#         param_t = params[t]

#         H = linear_tet_arap_dq(q, element, dphi_matrix, param_t, volume[t])  # (12,)

#         # Scatter H (12,) into f (3N,)
#         for i in range(4):
#             idx = element[i]
#             out[3 * idx: 3 * idx + 3] += H[3 * i: 3 * i + 3]
            
#     return out


@njit
def dpsi_arap_dS(S, params):
    stiffness = params[0]
    return 2.0 * stiffness * (S - 1.0)

@njit
def dpsi_stretch_dF(F, dpsi_func, params):
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
    # F = (B @ qe).reshape((3, 3), order='F')
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
        # dphi_matrix = dphidX[t].reshape((3, 4), order='F')
        dphi_matrix = dphidX[t].reshape(4, 3).T
        param_t = params[t]
        H = linear_tet_arap_dq(q, element, dphi_matrix, param_t, volume[t])

        for i in range(4):
            idx = element[i]
            out[3 * idx: 3 * idx + 3] += H[3 * i: 3 * i + 3]

    return out