import numpy as np
import scipy.sparse as sp
import igl

from src.lumped_mass_matrix import compute_vertex_voronoi_volumes

def create_poisson_mask_matrix(V, T):
    """
    Create a Poisson mask matrix phi (3N x 3N) as a diagonal sparse matrix.

    Args:
        V: (N, 3) vertex positions
        T: (M, 4) tetrahedra

    Returns:
        phi: (3N, 3N) sparse diagonal mask matrix
    """
    # Cotangent Laplacian and mass matrix
    L = igl.cotmatrix(V, T)
    # M = igl.massmatrix(V, T, igl.MASSMATRIX_TYPE_DEFAULT)
    M = compute_vertex_voronoi_volumes(V, T)

    # Identify boundary vertex indices
    F = igl.boundary_facets(T)
    b = np.unique(F.flatten())

    # Solve Poisson system with boundary constraints
    # ones = np.ones(V.shape[0])
    bc = np.zeros(len(b))
    Q = -L

    Aeq = sp.csr_matrix((0, Q.shape[0]))
    Beq = np.array([])  

    # print(Q.shape, l.shape, b.shape, bc.shape, Aeq.shape, Beq.shape)
    # Z = igl.min_quad_with_fixed(Q, M, b, bc, Aeq, Beq, False)

    # Multiply solution by mass matrix
    # Z_mass = M @ Z
    Z_mass = np.identity(V.shape[0])

    # Build diagonal mask (here: it's a zero diagonal matrix)
    ZZ = np.zeros(3 * Z_mass.shape[0])
    
    # Optional: use Z to weight mask if needed
    for i in range(Z_mass.shape[0]):
        if i in b: continue
        ZZ[3 * i + 0] = 1.0
        ZZ[3 * i + 1] = 1.0
        ZZ[3 * i + 2] = 1.0

    phi = sp.diags(ZZ, offsets=0, shape=(3 * V.shape[0], 3 * V.shape[0]), format='csr')
    return phi