from numba import njit
import numpy as np

@njit
def kron_dphidX_eye3(dphidX):
    """
    Manually compute np.kron(dphidX, np.eye(3)) → shape (9, 12)
    """
    B = np.zeros((3 * dphidX.shape[0], 3 * dphidX.shape[1]))
    for i in range(dphidX.shape[0]):
        for j in range(dphidX.shape[1]):
            for k in range(3):
                B[3*i + k, 3*j + k] = dphidX[i, j]
    return B