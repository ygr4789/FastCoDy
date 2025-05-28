import numpy as np
import scipy.sparse as sp

def emu_to_lame(E, p):
    lambda_ = (E * p) / ((1 + p) * (1 - 2 * p))
    mu = E / (2 * (1 + p))
    return lambda_, mu

def vectorize(M):
    return M.reshape(-1)

def matrixize(V):
    return V.reshape((3, -1), order='F').T  # Transpose to get (N, 3)

def reorder_xyzxyz_to_xxyyzz(H):
    n_vertices = H.shape[0] // 3

    # Build permutation indices
    idx = np.arange(3 * n_vertices)
    new_idx = np.concatenate([
        idx[0::3],  # x0, x1, x2, ...
        idx[1::3],  # y0, y1, y2, ...
        idx[2::3]   # z0, z1, z2, ...
    ])

    # Apply permutation to both rows and columns
    H_reordered = H[np.ix_(new_idx, new_idx)]
    return H_reordered

def sum_diagonal_blocks(H_reordered):
    n_vertices = H_reordered.shape[0] // 3
    # Extract blocks
    H_xx = H_reordered[0*n_vertices : 1*n_vertices, 0*n_vertices : 1*n_vertices]
    H_yy = H_reordered[1*n_vertices : 2*n_vertices, 1*n_vertices : 2*n_vertices]
    H_zz = H_reordered[2*n_vertices : 3*n_vertices, 2*n_vertices : 3*n_vertices]
    
    # Sum the diagonal blocks
    H_sum = H_xx + H_yy + H_zz
    return H_sum

def lumped_mass_3n_to_n(M_3n):
    n_vertices = M_3n.shape[0] // 3
    lumped_masses = np.zeros(n_vertices)
    for i in range(n_vertices):
        lumped_masses[i] = M_3n[3*i, 3*i]
    M_n = sp.diags(lumped_masses)
    return M_n