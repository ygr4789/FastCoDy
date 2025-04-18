from numba import njit
import numpy as np

@njit
def dsvd(U, S, V):
    """
    Compute derivatives of SVD components U, S, V with respect to input matrix.
    All inputs and outputs should be float64.
    """
    # Convert inputs to float64 if they aren't already
    U = U.astype(np.float64)
    S = S.astype(np.float64)
    V = V.astype(np.float64)
    
    # Initialize output arrays with float64
    dU = np.zeros((3, 3, 3, 3), dtype=np.float64)
    dS = np.zeros((3, 3, 3), dtype=np.float64)
    dV = np.zeros((3, 3, 3, 3), dtype=np.float64)

    tol = 1e-8
    S0, S1, S2 = S

    d01 = S1**2 - S0**2
    d02 = S2**2 - S0**2
    d12 = S2**2 - S1**2

    d01 = 1.0 / (d01 if abs(d01) >= tol else np.inf)
    d02 = 1.0 / (d02 if abs(d02) >= tol else np.inf)
    d12 = 1.0 / (d12 if abs(d12) >= tol else np.inf)

    S_diag = np.diag(S)

    for r in range(3):
        for s in range(3):
            UVT = np.outer(U[r], V[s])  # 3x3
            diag = np.diag(UVT)
            dS[r, s] = diag

            UVT_no_diag = UVT - np.diag(diag)

            tmp = S_diag @ UVT_no_diag + UVT_no_diag.T @ S_diag
            w01 = tmp[0, 1] * d01
            w02 = tmp[0, 2] * d02
            w12 = tmp[1, 2] * d12
            skew = np.array([[0, w01, w02],
                             [-w01, 0, w12],
                             [-w02, -w12, 0]], dtype=np.float64)

            dV[r, s] = V @ skew

            tmp = UVT_no_diag @ S_diag + S_diag @ UVT_no_diag.T
            w01 = tmp[0, 1] * d01
            w02 = tmp[0, 2] * d02
            w12 = tmp[1, 2] * d12
            skew = np.array([[0, w01, w02],
                             [-w01, 0, w12],
                             [-w02, -w12, 0]], dtype=np.float64)

            dU[r, s] = U @ skew

    return dU, dS, dV