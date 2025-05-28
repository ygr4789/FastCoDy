import numpy as np
import scipy.sparse as sp
from numpy.linalg import svd

# Define dtype constant for consistency
DTYPE = np.float64

def dpsi_arap_dS(S, params):
    """
    Gradient of ARAP energy w.r.t. singular values.
    Handles batched inputs.
    
    Args:
        S: (..., 3) numpy array (double), where ... can be any number of dimensions
        params: (..., 1) numpy array (double), with same batch dimensions as S
    Returns:
        (..., 3) numpy array (double) with same batch dimensions as S
    """
    S = np.asarray(S, dtype=DTYPE)
    params = np.asarray(params, dtype=DTYPE)
    stiffness = params[..., 0:1]  # (..., 1)
    return 2.0 * stiffness * (S - 1.0)

def d2psi_arap_dS2(S, params):
    """
    Hessian (diagonal) of ARAP energy w.r.t. singular values.
    Handles batched inputs.
    
    Args:
        S: (..., 3) numpy array (double), where ... can be any number of dimensions
        params: (..., 1) numpy array (double), with same batch dimensions as S
    Returns:
        (..., 3, 3) numpy array (double) with same batch dimensions as S
    """
    S = np.asarray(S, dtype=DTYPE)
    params = np.asarray(params, dtype=DTYPE)
    
    # Get batch shape
    batch_shape = S.shape[:-1]
    
    # Create identity matrices for each batch element
    eye = np.eye(3, dtype=DTYPE)  # (3, 3)
    eye = np.broadcast_to(eye, (*batch_shape, 3, 3))  # (..., 3, 3)
    
    stiffness = params[..., 0:1, None]  # (..., 1, 1)
    return 2.0 * stiffness * eye

def d2psi_stretch_dF2(F, params):
    """
    Compute the second derivative of stretch energy wrt F using SVD-based chain rule.
    Handles batched inputs.
    
    Args:
        F: (..., 3, 3) numpy array (double), where ... can be any number of dimensions
        params: (..., 1) numpy array (double), with same batch dimensions as F
    Returns:
        (..., 9, 9) numpy array (double) with same batch dimensions as F
    """
    F = np.asarray(F, dtype=DTYPE)
    batch_shape = F.shape[:-2]
    
    # Compute batched SVD using numpy.linalg.svd
    U, S, Vt = svd(F, full_matrices=False)
    V = Vt.transpose(0, 2, 1)  # (..., 3, 3)

    # Compute derivatives for each batch element
    Plam = dpsi_arap_dS(S, params)       # (..., 3)
    Plam2 = d2psi_arap_dS2(S, params)     # (..., 3, 3)

    # Compute SVD derivatives once
    dU, dS, dV = dsvd(U, S, V)  # These will be batched

    # Initialize output array
    ddw = np.zeros((*batch_shape, 9, 9), dtype=DTYPE)

    # Create batched diagonal matrices
    Plam_diag = np.zeros((*batch_shape, 3, 3), dtype=DTYPE)
    for i in range(3):
        Plam_diag[..., i, i] = Plam[..., i]

    # For each batch element, compute the derivatives
    for r in range(3):
        for s in range(3):
            dS_rs = dS[..., r, s, :]    # (..., 3)
            
            # Compute PlamVec for each batch element
            PlamVec = np.matmul(Plam2, dS_rs[..., None]).squeeze(-1)  # (..., 3)
            
            # Create batched diagonal matrix for PlamVec
            PlamVec_diag = np.zeros((*batch_shape, 3, 3), dtype=DTYPE)
            for i in range(3):
                PlamVec_diag[..., i, i] = PlamVec[..., i]

            # Matrix multiplications for each batch element
            A = np.matmul(np.matmul(dU[..., r, s], Plam_diag), V.transpose(0, 2, 1))
            B = np.matmul(np.matmul(U, Plam_diag), dV[..., r, s].transpose(0, 2, 1))
            C = np.matmul(np.matmul(U, PlamVec_diag), V.transpose(0, 2, 1))

            rowMat = (A + B + C)
            ddw[..., 3 * r + s, :] = rowMat.reshape(*batch_shape, 9)

    return ddw

def linear_tet_arap_dq2(q, element, dphidX, params, volume):
    """
    Compute per-element ARAP stiffness matrix for batched tetrahedra using numpy.
    
    Args:
        q: (3N,) numpy array (double), global displacement vector (same for all batches)
        element: (..., 4) numpy array (long), indices of tetrahedra
        dphidX: (..., 12) numpy array (double), flattened shape function gradients
        params: (..., 1) numpy array (double), parameters for each element
        volume: (..., 1) numpy array (double), volume of each element
    Returns:
        (..., 12, 12) numpy array (double), local stiffness matrices
    """
    
    # Get batch shape
    batch_shape = element.shape[:-1]
    
    # Reshape q to (N, 3) for easier indexing
    q_reshaped = q.reshape(-1, 3)
    
    # Create empty array for element displacements
    qe = np.empty((*batch_shape, 12), dtype=DTYPE)
    
    # For each vertex in the element
    for i in range(4):
        # Get the vertex indices for this element
        vert_idx = element[..., i]  # (...,)
        # Get the displacements for these vertices
        vert_disp = q_reshaped[vert_idx]  # (..., 3)
        # Store in qe
        qe[..., 3*i:3*i+3] = vert_disp

    # Reshape dphidX from (..., 12) to (..., 3, 4)
    dphidX = dphidX.reshape(*batch_shape, 3, 4)

    # Compute B matrix (..., 9, 12) for each element
    # First create batched identity matrices
    I = np.eye(3, dtype=DTYPE)
    I = np.broadcast_to(I, (*batch_shape, 3, 3))  # (..., 3, 3)
    
    # Compute kron product for last two dimensions while preserving batch dimensions
    B = np.einsum('...ij,...kl->...ikjl', dphidX, I)  # (..., 3, 3, 4, 3) 
    B = B.reshape(*batch_shape, 9, 12)  # (..., 9, 12)
    
    # Compute deformation gradient F = unflatten(B @ qe)
    # Reshape qe for matrix multiplication
    qe_reshaped = qe.reshape(*batch_shape, 12, 1)  # (..., 12, 1)
    F_flat = np.matmul(B, qe_reshaped).reshape(*batch_shape, 9)  # (..., 9)
    F = F_flat.reshape(*batch_shape, 3, 3).transpose(0, 2, 1)  # (..., 3, 3)

    # Compute second derivative of energy: dF (..., 9, 9)
    dF = d2psi_stretch_dF2(F, params)

    # Final stiffness matrix H = Bᵀ * dF * B * volume
    # Reshape volume for broadcasting
    volume = volume.reshape(*batch_shape, 1, 1)  # (..., 1, 1)
    
    # Compute H = B^T @ dF @ B * volume for each batch element
    H = np.matmul(np.matmul(B.transpose(0, 2, 1), dF), B) * volume

    return H

def linear_tetall_arap_dq2(V, E, q, dphidX, volume, params):
    """
    Compute ARAP stiffness matrices for all tetrahedra.
    All inputs are batched numpy arrays.
    """
    H_all = linear_tet_arap_dq2(q, E, dphidX, params, volume)  # (num_tets, 12, 12)
    return H_all

def dsvd(U, S, V):
    """
    Compute derivatives of SVD components U, S, V with respect to input matrix.
    Handles batched inputs using numpy.
    
    Args:
        U: (..., 3, 3) numpy array (double), left singular vectors
        S: (..., 3) numpy array (double), singular values
        V: (..., 3, 3) numpy array (double), right singular vectors
    Returns:
        dU: (..., 3, 3, 3, 3) numpy array (double)
        dS: (..., 3, 3, 3) numpy array (double)
        dV: (..., 3, 3, 3, 3) numpy array (double)
    """
    # Get batch shape
    batch_shape = U.shape[:-2]
    
    # Initialize output arrays
    dU = np.zeros((*batch_shape, 3, 3, 3, 3), dtype=DTYPE)
    dS = np.zeros((*batch_shape, 3, 3, 3), dtype=DTYPE)
    dV = np.zeros((*batch_shape, 3, 3, 3, 3), dtype=DTYPE)

    tol = 1e-8
    S0, S1, S2 = S[..., 0], S[..., 1], S[..., 2]  # (...,)

    d01 = S1**2 - S0**2  # (...,)
    d02 = S2**2 - S0**2  # (...,)
    d12 = S2**2 - S1**2  # (...,)

    # Handle division with broadcasting, suppressing warnings
    with np.errstate(divide='ignore', invalid='ignore'):
        d01 = np.where(np.abs(d01) >= tol, 1.0 / d01, np.zeros_like(d01))
        d02 = np.where(np.abs(d02) >= tol, 1.0 / d02, np.zeros_like(d02))
        d12 = np.where(np.abs(d12) >= tol, 1.0 / d12, np.zeros_like(d12))

    # Create batched diagonal matrices
    S_diag = np.zeros((*batch_shape, 3, 3), dtype=DTYPE)
    for i in range(3):
        S_diag[..., i, i] = S[..., i]

    for r in range(3):
        for s in range(3):
            # Compute outer products for each batch element
            UVT = np.matmul(U[..., r:r+1, :].transpose(0, 2, 1), V[..., s:s+1, :])  # (..., 3, 3)
            diag = np.diagonal(UVT, axis1=-2, axis2=-1)  # (..., 3)
            dS[..., r, s, :] = diag

            # Remove diagonal elements
            UVT_no_diag = UVT.copy()
            for i in range(3):
                UVT_no_diag[..., i, i] = 0

            # Compute skew-symmetric matrices
            tmp = np.matmul(S_diag, UVT_no_diag) + np.matmul(UVT_no_diag.transpose(0, 2, 1), S_diag)
            w01 = tmp[..., 0, 1] * d01  # (...,)
            w02 = tmp[..., 0, 2] * d02  # (...,)
            w12 = tmp[..., 1, 2] * d12  # (...,)

            # Create batched skew-symmetric matrices
            skew = np.zeros((*batch_shape, 3, 3), dtype=DTYPE)
            skew[..., 0, 1] = w01
            skew[..., 0, 2] = w02
            skew[..., 1, 2] = w12
            skew[..., 1, 0] = -w01
            skew[..., 2, 0] = -w02
            skew[..., 2, 1] = -w12

            # Compute derivatives
            dV[..., r, s] = np.matmul(V, skew)
            
            tmp = np.matmul(UVT_no_diag, S_diag) + np.matmul(S_diag, UVT_no_diag.transpose(0, 2, 1))
            w01 = tmp[..., 0, 1] * d01
            w02 = tmp[..., 0, 2] * d02
            w12 = tmp[..., 1, 2] * d12

            skew = np.zeros((*batch_shape, 3, 3), dtype=DTYPE)
            skew[..., 0, 1] = w01
            skew[..., 0, 2] = w02
            skew[..., 1, 2] = w12
            skew[..., 1, 0] = -w01
            skew[..., 2, 0] = -w02
            skew[..., 2, 1] = -w12

            dU[..., r, s] = np.matmul(U, skew)

    return dU, dS, dV

def linear_tetmesh_arap_dq2(V, E, q, dphidX, volume, params):
    # Compute using numpy
    H_all = linear_tetall_arap_dq2(V, E, q, dphidX, volume, params)
    
    # Convert to sparse matrix
    num_tets = E.shape[0]
    N = V.shape[0]
    
    offsets = np.array([0, 1, 2], dtype=np.int64)
    global_idx = 3 * E[:, :, None] + offsets  # shape (num_tets, 4, 3)
    global_idx = global_idx.reshape(num_tets, 12)
    
    row_indices = np.repeat(global_idx, 12, axis=1).reshape(-1)  # (num_tets * 144,)
    col_indices = np.tile(global_idx, (1, 12)).reshape(-1)       # (num_tets * 144,)

    values = H_all.reshape(-1)
    H_sparse = sp.coo_matrix((values, (row_indices, col_indices)), shape=(3 * N, 3 * N)).tocsr()
    return H_sparse
