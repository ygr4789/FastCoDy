import numpy as np
import igl
from scipy.linalg import null_space
from scipy.sparse.linalg import svds, eigsh

from vedo import Points, Plotter
# from vedo.applications import AnimationPlayer

if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
    
from src import *
from sim import *

def reorder_hessian_xyz_to_xxyyzz(H):
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
    M_n = np.diag(lumped_masses)
    return M_n

def solve_generalized_eig(H, M, J, n=10, tol=1e-12):
    null_J = null_space(J.todense())

    # Project H and M into the null space
    H_proj = null_J.T @ H @ null_J
    M_proj = null_J.T @ M @ null_J

    # Solve generalized eigenproblem in reduced space
    eigvals, eigvecs = eigsh(H_proj, k=n, M=M_proj, which='SM')

    # Recover full-space eigenvectors
    full_space_evecs = null_J @ eigvecs
    return full_space_evecs

def create_eigenmode_weights(K, M, J=None, n=10):
    M = lumped_mass_3n_to_n(M)
    KW = sum_diagonal_blocks(reorder_hessian_xyz_to_xxyyzz(K))
    
    if J is not None:
        return solve_generalized_eig(KW, M, J, n)
    else:
        eigvals, eigvecs = eigsh(KW, k=n, M=M, which='SM')
        return eigvecs

def visualize_eigenmodes(V, EMs):
    num_modes = EMs.shape[0]
    rows = int(np.sqrt(num_modes))
    cols = int(np.ceil(num_modes / rows))

    plotter = Plotter(shape=(rows, cols), title="Eigenmodes")

    camera_settings = dict(
        pos=(10, 0, 0),
        focalPoint=(0, 0, 0),
        viewup=(0, 0, 1)
    )
    for i in range(num_modes):
        pc = Points(V, c='green', r=4)
        pc.pointdata["scalars"] = EMs[i]
        pc.cmap("viridis", vmin=-10.0, vmax=10.0).add_scalarbar(title="weight")
        plotter.show(pc, at=i, interactive=False, camera=camera_settings)
    plotter.interactive().close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='generate complementary dynamics secondary motion')
    parser.add_argument('--input', '-i', type=str, required=True, default='examples/sphere/sphere.json', help='json input path')
    
    args = parser.parse_args()
    json_path = args.input
    
    V, T, F, C, PI, BE, W, TF_list, dt, YM, pr, scale, physic_model = read_json_data(json_path)
    lambda_, mu = emu_to_lame(YM, pr)
    params = np.zeros((T.shape[0], 2))
    params[:, 0] = 0.5 * lambda_
    params[:, 1] = mu
    
    dX = linear_tetmesh_dphi_dX(V, T)
    vol = igl.volume(V, T)
    VCol = vectorize(V)
    K = linear_tetmesh_arap_dq2(V, T, VCol, dX, vol, params)
    M = lumped_mass_matrix(V, T)
    J = lbs_matrix_column(V, W)
    phi = create_mask_matrix(V, T, C, BE)
    Jleak = phi @ M @ J
    
    Jw = weight_space_constraint(Jleak, V)
    
    EMs = create_eigenmode_weights(K, M, Jw, n=12).T
    visualize_eigenmodes(V, EMs)
