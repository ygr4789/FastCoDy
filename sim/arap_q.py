import numpy as np
from numba import njit

from .kron import kron_dphidX_eye3

def psi_arap_S(S, param):
    """
    ARAP energy from singular values.
    """
    return param * np.sum((S - 1.0) ** 2)

def psi_stretch_F(F, psi, param):
    """
    Apply scalar energy function `psi` to singular values of deformation F.
    """
    U, S, Vt = np.linalg.svd(F.astype(np.float32))
    return psi(S, param)

def linear_tet_arap_q(q, element, dphidX, param, volume):
    """
    Local ARAP energy for a single tet.
    """
    qe = np.concatenate([q[3 * i: 3 * i + 3] for i in element])  # (12,)
    B = np.kron(np.eye(3), dphidX)  # (9, 12)
    F = (B @ qe).reshape(3, 3, order='F')
    return volume * psi_stretch_F(F, psi_arap_S, param)

def linear_tetmesh_arap_q(V, E, q, dphidX_all, volume, params):
    """
    Total ARAP energy over all tetrahedra.
    """
    # total_energy = 0.0
    # for i in range(E.shape[0]):
    #     element = E[i]
    #     dphidX = dphidX_all[i].reshape(3, 4, order='F')
    #     param = params[i] if params.ndim == 1 else params[i, 0]
    #     total_energy += linear_tet_arap_q(q, element, dphidX, param, volume[i])
    # return total_energy
    
    dphidX_all_reshaped = dphidX_all.reshape(-1, 3, 4, order='F')
    energy = linear_tetmesh_arap_q_numba(E, q, dphidX_all_reshaped, volume, params)
    return energy

@njit
def linear_tetmesh_arap_q_numba(E, q, dphidX_all, volume, params):
    total_energy = 0.0
    num_tets = E.shape[0]
    
    for i in range(num_tets):
        element = E[i]
        dphidX = dphidX_all[i]  # shape: (3, 4)
        param = params[i] if params.ndim == 1 else params[i, 0]

        qe = np.empty(12)
        for j in range(4):
            qe[3 * j:3 * j + 3] = q[3 * element[j]:3 * element[j] + 3]

        B = kron_dphidX_eye3(dphidX)  # (9, 12)
        F = np.zeros((3, 3))
        for c in range(3):
            for r in range(3):
                F[r, c] = B[3 * c + r] @ qe

        U, S, Vt = np.linalg.svd(F.astype(np.float32))
        energy = param * np.sum((S - 1.0) ** 2)
        total_energy += volume[i] * energy

    return total_energy