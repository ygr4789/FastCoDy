import numpy as np
import scipy.sparse as sp
from numba import njit

from .dsvd import dsvd

@njit
def dpsi_arap_dS(S, params):
    """
    Gradient of ARAP energy w.r.t. singular values.
    S: (3,)
    params: (1,)
    Returns: (3,)
    """
    return 2.0 * params[0] * (S - 1.0)

@njit
def d2psi_arap_dS2(S, params):
    """
    Hessian (diagonal) of ARAP energy w.r.t. singular values.
    Returns: (3, 3) identity scaled by 2 * params[0]
    """
    return np.eye(3) * (2.0 * params[0])

@njit
def d2psi_stretch_dF2(F, dpsi_func, d2psi_func, params):
    """
    Compute the second derivative of stretch energy wrt F using SVD-based chain rule.
    Returns ddw: (9, 9)
    """
    F = F.astype(np.float64)
    # Change scipy.linalg.svd to np.linalg.svd
    U, S, Vt = np.linalg.svd(F)
    V = Vt.T

    # Recompute SVD with slight perturbation if singular values are not unique
    if np.abs(S[0] - S[1]) < 1e-5 or np.abs(S[1] - S[2]) < 1e-5 or np.abs(S[0] - S[2]) < 1e-5:
        F += np.random.randn(3, 3).astype(np.float64) * 1e-5
        U, S, Vt = np.linalg.svd(F)
        V = Vt.T

    Plam = dpsi_func(S, params)       # (3,)
    Plam2 = d2psi_func(S, params)     # (3, 3)

    dU, dS, dV = dsvd(U, S, V)

    ddw = np.zeros((9, 9), dtype=np.float64)
    for r in range(3):
        for s in range(3):
            dS_rs = dS[r, s, :]  # (3,)
            PlamVec = Plam2 @ dS_rs  # (3,)

            # Matrix multiplications only — all (3,3)
            A = dU[r, s] @ np.diag(Plam) @ V.T
            B = U @ np.diag(Plam) @ dV[r, s].T
            C = U @ np.diag(PlamVec) @ V.T

            # rowMat = (A + B + C).T  # Transpose like in Eigen
            # ddw[3 * r + s, :] = rowMat.flatten(order='F')
            rowMat = (A + B + C)
            ddw[3 * r + s, :] = rowMat.flatten()

    return ddw

def linear_tetmesh_arap_dq2(V, E, q, dphidX, volume, params, func=None):
    N = V.shape[0]
    
    values, row_indices, col_indices = linear_tetmesh_arap_dq2_(V, E, q, dphidX, volume, params, func)
    H_sparse = sp.coo_matrix((values, (row_indices, col_indices)), shape=(3 * N, 3 * N)).tocsr()
    return H_sparse
    
@njit
def linear_tetmesh_arap_dq2_(V, E, q, dphidX, volume, params, func=None):
    """
    Assemble a global sparse matrix from per-tetrahedron ARAP derivatives.

    Args:
        V: (N, 3) vertex positions
        E: (M, 4) tetrahedron indices
        q: (3N,) global displacement vector
        dphidX: (M, 12)
        volume: (M,)
        params: (M, K) or (M,)
        func: optional callback for H (e.g., to collect triplets or modify H)
    Returns:
        out: scipy.sparse.csr_matrix of shape (3N, 3N)
    """
    num_tets = E.shape[0]

    row_indices = []
    col_indices = []
    values = []

    for t in range(num_tets):
        element = E[t]
        # dphi = dphidX[t].reshape(3, 4, order='F')
        dphi = dphidX[t].reshape(4, 3).T
        param = params[t]
        vol = volume[t]

        H_local = linear_tet_arap_dq2(q, element, dphi, param, vol)  # (12, 12)

        for i in range(4):
            for j in range(4):
                vi = element[i]
                vj = element[j]
                for ri in range(3):
                    for rj in range(3):
                        global_row = 3 * vi + ri
                        global_col = 3 * vj + rj
                        local_val = H_local[3 * i + ri, 3 * j + rj]

                        row_indices.append(global_row)
                        col_indices.append(global_col)
                        values.append(local_val)

    return values, row_indices, col_indices

@njit
def linear_tet_arap_dq2(q, element, dphidX, params, volume):
    """
    Compute per-element ARAP stiffness matrix (12x12) for one tetrahedron.

    Args:
        q: (3N,) global displacement vector
        element: (4,) indices of this tetrahedron
        dphidX: (3, 4) shape function gradients in world space
        params: (1,) or (K,) parameter vector
        volume: float, volume of this tetrahedron

    Returns:
        H: (12, 12) local stiffness matrix
    """
    # Replace list comprehension + concatenate with direct array assignment
    qe = np.empty(12, dtype=np.float64)
    for i in range(4):
        qe[3*i:3*i+3] = q[3*element[i]:3*element[i]+3]

    # Compute B matrix (9x12)
    B = np.kron(dphidX, np.eye(3))

    # Compute deformation gradient F = unflatten(B @ qe)
    # F = (B @ qe).reshape(3, 3, order='F')
    F = (B @ qe).reshape(3, 3).T

    # Compute second derivative of energy: dF (9x9)
    dF = d2psi_stretch_dF2(F, dpsi_arap_dS, d2psi_arap_dS2, params)

    # Final stiffness matrix H = Bᵀ * dF * B * volume
    H = B.T @ dF @ B * volume
    return H



