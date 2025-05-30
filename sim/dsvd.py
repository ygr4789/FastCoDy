import numpy as np

DTYPE = np.float64

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