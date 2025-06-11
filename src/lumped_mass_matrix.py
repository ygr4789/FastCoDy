import numpy as np
import scipy.sparse as sp
import igl

def compute_tet_volume(a, b, c, d):
    return np.abs(np.einsum('ij,ij->i', (a - d), np.cross(b - d, c - d))) / 6.0

def compute_vertex_voronoi_volumes(V, T):
    vol = np.zeros(V.shape[0])
    a, b, c, d = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]], V[T[:, 3]]
    tet_volumes = compute_tet_volume(a, b, c, d)
    
    # Check for negative volumes
    if np.any(tet_volumes <= 0):
        raise ValueError("Negative or zero volume tetrahedra detected")
    
    for i in range(4):
        vol[T[:, i]] += tet_volumes / 4.0

    return vol

def lumped_mass_matrix(V, T):
    """
    Returns:
        M: (3N x 3N) lumped mass matrix (scipy.sparse.csr_matrix)
    """
    N = V.shape[0]
    v_masses = compute_vertex_voronoi_volumes(V, T)

    rows = []
    cols = []
    data = []

    for i in range(N):
        mass = v_masses[i]
        for d in range(3):
            idx = 3 * i + d
            rows.append(idx)
            cols.append(idx)
            data.append(mass)

    M = sp.coo_matrix((data, (rows, cols)), shape=(3 * N, 3 * N)).tocsr()
    return M