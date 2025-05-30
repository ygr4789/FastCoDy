import numpy as np

def eval_at_point(V, T, func, tmp_shape, X=None):
    """
    Evaluate a function per element in a tetrahedral mesh.
    Args:
        V: (N, 3) vertex positions
        T: (M, 4) tetrahedral elements
        func: function to compute per element
        tmp_shape: shape of per-element output (e.g., (4, 3))
        X: optional per-element data (not used here)
    Returns:
        results: (M, flatten(tmp_shape).size) array of evaluated results
    """
    M = T.shape[0]
    tmp_size = np.prod(tmp_shape)
    results = np.zeros((M, tmp_size))

    for i in range(M):
        element = T[i]
        dphi = func(V, element)
        results[i, :] = dphi.T.flatten()

    return results

def linear_tet_dphi_dX(V, element, X=None):
    """
    Compute dphi/dX for a single tetrahedron.
    Args:
        V: (N, 3) ndarray of vertices.
        element: (4,) array of vertex indices for one tetrahedron.
        X: Placeholder (not used in this context).
    Returns:
        dphi: (4, 3) ndarray.
    """
    dphi = np.zeros((4, 3))
    Dm = np.stack([
        V[element[1]] - V[element[0]],
        V[element[2]] - V[element[0]],
        V[element[3]] - V[element[0]]
    ], axis=1)  # Shape (3, 3)

    inv_Dm = np.linalg.inv(Dm)  # Shape (3, 3)
    dphi[1:4, :] = inv_Dm
    dphi[0, :] = -np.sum(inv_Dm, axis=0)
    return dphi

def linear_tetmesh_dphi_dX(V, T):
    """
    Compute dphi/dX for all tetrahedra in a mesh.
    Args:
        V: (N, 3) vertex positions
        T: (M, 4) tetrahedral elements
    Returns:
        dX: (M, 12) array of gradients (each row: 4x3 matrix flattened)
    """
    return eval_at_point(V, T, linear_tet_dphi_dX, tmp_shape=(4, 3))