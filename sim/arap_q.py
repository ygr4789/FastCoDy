import numpy as np
from numba import njit

from .kron import kron_dphidX_eye3

@njit
def psi_arap_S(S, param):
    return param * np.sum((S - 1.0) ** 2)

@njit
def psi_stretch_F(F, psi, param):
    U, S, Vt = np.linalg.svd(F.astype(np.float32))
    return psi(S, param)

@njit
def linear_tet_arap_q(q, element, dphidX, param, volume):
    qe = np.empty(12)
    for j in range(4):
        idx = element[j]
        qe[3 * j: 3 * j + 3] = q[3 * idx: 3 * idx + 3]
    B = kron_dphidX_eye3(dphidX)
    F = (B @ qe).reshape(3, 3).T
    return volume * psi_stretch_F(F, psi_arap_S, param)

@njit
def linear_tetmesh_arap_q(V, E, q, dphidX_all, volume, params):
    total_energy = 0.0
    for i in range(E.shape[0]):
        element = E[i]
        dphidX = dphidX_all[i].reshape(3, 4)
        param = params[i] if params.ndim == 1 else params[i, 0]
        total_energy += linear_tet_arap_q(q, element, dphidX, param, volume[i])
    return total_energy