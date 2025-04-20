import numpy as np
from numpy.linalg import eigh
from numba import njit


@njit
def simple_psd_fix(A: np.ndarray, tol: float = 1e-3) -> np.ndarray:
    """
    Ensure A is positive semi-definite (PSD) by clamping eigenvalues to at least `tol`.
    
    Parameters:
    - A: A square symmetric matrix (NumPy 2D array)
    - tol: Scalar threshold to clamp eigenvalues

    Returns:
    - Modified matrix A that is PSD
    """
    # Compute eigen-decomposition (symmetric/hermitian)
    evals, evecs = eigh(A)  # equivalent to Eigen::SelfAdjointEigenSolver

    # Clamp eigenvalues
    evals_clamped = np.maximum(evals, tol)

    # Reconstruct matrix
    A_fixed = (evecs @ np.diag(evals_clamped)) @ evecs.T
    return A_fixed