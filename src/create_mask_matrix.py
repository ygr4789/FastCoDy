import numpy as np
import scipy.sparse as sp
import igl

if __name__ == "__main__":
    from lumped_mass_matrix import compute_vertex_voronoi_volumes
    from read_data_from_json import read_json_data
else:
    from .lumped_mass_matrix import compute_vertex_voronoi_volumes

def create_mask_matrix(V, T, C=None, BE=None, mask_type="poisson"):
    M = compute_vertex_voronoi_volumes(V, T)
    
    if mask_type == "lin":
        Z = linear_weights(V, T, M)
    elif mask_type == "rig":
        Z = rig_ortho_weights(V, T, C, BE)
    else:
        Z = poisson_weights(V, T, M)
    
    ZZ = np.zeros(3 * Z.shape[0])
    
    for i in range(Z.shape[0]):
        ZZ[3 * i + 0] = Z[i]
        ZZ[3 * i + 1] = Z[i]
        ZZ[3 * i + 2] = Z[i]

    phi = sp.diags(ZZ, offsets=0, shape=(3 * V.shape[0], 3 * V.shape[0]), format='csr')
    return phi
    
def linear_weights(V, T, M):
    Z = np.zeros(V.shape[0])
    for i in range(V.shape[0]):
        top = 0.5
        bot = -0.5
        if V[i][2] > top: Z[i] = 0.0
        elif V[i][2] < bot: Z[i] = 1.0
        # else: Z[i] = 0.0
        else: Z[i] = (top - V[i][2]) / (top - bot)
    return M * Z
    # return Z
    
def rig_ortho_weights(V, T, C, BE):
    pass

def poisson_weights(V, T, M):
    L = igl.cotmatrix(V, T)
    F = igl.boundary_facets(T)
    b = np.unique(F.flatten())
    bc = np.zeros((len(b), 1))

    Q = -L
    Aeq = sp.csc_matrix((0, Q.shape[0]))
    Beq = np.array([])  

    _, Z = igl.min_quad_with_fixed(Q, M, b, bc, Aeq, Beq, False)
    return M * Z

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='calculate cody constrint mask weights')
    parser.add_argument('--input', '-i', type=str, required=False, default='examples/sphere/sphere.json', help='json input path')
    parser.add_argument('--type', '-t', type=str, required=False, default='poisson', help='mask type')
    args = parser.parse_args()
    json_path = args.input
    mask_type = args.type

    V, T, F, C, PI, BE, W, TF_list, dt, YM, pr, scale, physic_model = read_json_data(json_path)
    phi = create_mask_matrix(V, T, mask_type=mask_type)
    phi_diag = phi.diagonal() 
    Z = phi_diag[::3]
    
    from vedo import Mesh, Plotter, Points
    
    pc = Points(V, r=8)
    pc.pointdata["scalars"] = Z
    pc.cmap("viridis").add_scalarbar(title="weight")

    camera_settings = dict(
        pos=(10, 0, 0),           # Camera position
        focalPoint=(0, 0, 0),    # Look-at target
        viewup=(0, 0, 1)         # "Up" direction
    )

    plotter = Plotter()
    plotter.show(pc, camera=camera_settings, interactive = True)
