import numpy as np
from numba import njit, prange

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
    
@njit(parallel=True)
def linear_tetmesh_arap_q(V, E, q, dphidX_all, volume, params):
    num_tets = E.shape[0]
    energy_all = np.zeros(num_tets, dtype=np.float64)
    
    for t in prange(num_tets):
        element = E[t]
        dphidX = dphidX_all[t].reshape(3, 4)
        param = params[t, 0]
        energy_all[t] = linear_tet_arap_q(q, element, dphidX, param, volume[t])
        
    return energy_all.sum()