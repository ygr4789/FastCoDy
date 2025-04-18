import numpy as np
import scipy.sparse as sp
import igl

def compute_tet_volume(a, b, c, d):
    return np.abs(np.einsum('ij,ij->i', (a - d), np.cross(b - d, c - d))) / 6.0

def compute_vertex_voronoi_volumes(V, T):
    vol = np.zeros(V.shape[0])
    a, b, c, d = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]], V[T[:, 3]]
    tet_volumes = compute_tet_volume(a, b, c, d)

    for i in range(4):
        np.add.at(vol, T[:, i], tet_volumes / 4.0)

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

# def lumped_mass_matrix(V, T):
#     """
#     Create a 3N x 3N block diagonal mass matrix from an N x 3 vertex matrix and a Tetrahedral mesh.
    
#     Args:
#         V: (N, 3) numpy array of vertex positions.
#         T: (M, 4) numpy array of tetrahedron indices.
    
#     Returns:
#         M: scipy.sparse.csr_matrix of shape (3N, 3N)
#     """
#     # Step 1: Compute lumped mass matrix (N x N)
#     Ms = igl.massmatrix(V, T, igl.MASSMATRIX_TYPE_FULL)  # returns scipy sparse

#     # Step 2: Create block-diagonal (3N x 3N)
#     row, col, data = [], [], []

#     Ms = Ms.tocoo()
#     for i, j, v in zip(Ms.row, Ms.col, Ms.data):
#         for d in range(3):
#             idx = 3 * i + d
#             jdx = 3 * j + d
#             row.append(idx)
#             col.append(jdx)
#             data.append(v)

#     M = sp.coo_matrix((data, (row, col)), shape=(3 * V.shape[0], 3 * V.shape[0]))
#     return M.tocsr()