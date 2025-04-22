import numpy as np
import scipy.sparse as sp
import igl

if __name__ == "__main__":
    from lumped_mass_matrix import compute_vertex_voronoi_volumes
    from read_data_from_json import read_json_data
else:
    from .lumped_mass_matrix import compute_vertex_voronoi_volumes

def create_mask_matrix(V, T, flag="poisson"):
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
    bc = np.zeros(len(b))
    Q = -L

    Aeq = sp.csc_matrix((0, Q.shape[0]))
    Beq = np.array([])  

    _, Z = igl.min_quad_with_fixed(Q, M, b, bc, Aeq, Beq, False)

    # Multiply solution by mass matrix
    Z_mass = M * Z

    # Build diagonal mask (here: it's a zero diagonal matrix)
    ZZ = np.zeros(3 * Z_mass.shape[0])
    
    # Optional: use Z to weight mask if needed
    for i in range(Z_mass.shape[0]):
        if V[i][2] > 0.06:
            ZZ[3 * i + 0] = 0.0
            ZZ[3 * i + 1] = 0.0
            ZZ[3 * i + 2] = 0.0
        else:
            ZZ[3 * i + 0] = 1.0
            ZZ[3 * i + 1] = 1.0
            ZZ[3 * i + 2] = 1.0
        # ZZ[3 * i + 0] = Z[i]
        # ZZ[3 * i + 1] = Z[i]
        # ZZ[3 * i + 2] = Z[i]

    phi = sp.diags(ZZ, offsets=0, shape=(3 * V.shape[0], 3 * V.shape[0]), format='csr')
    return phi


if __name__ == "__main__":
    import sys
    json_path = sys.argv[1] if len(sys.argv) > 1 else "examples/elephant/elephant.json"

    V, T, F, C, PI, BE, W, TF_list, dt, YM, pr, scale, physic_model = read_json_data(json_path)
    phi = create_mask_matrix(V, T)
    phi_diag = phi.diagonal() 
    Z = phi_diag[::3]
    
    from vedo import Mesh, Plotter, Points
    
    pc = Points(V, r=8)
    pc.pointdata["scalars"] = Z
    pc.cmap("viridis").add_scalarbar(title="weight")

    plotter = Plotter()
    plotter.show(pc, interactive = True)
