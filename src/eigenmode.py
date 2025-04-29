import numpy as np
import igl
from scipy.sparse.linalg import eigsh

from vedo import Points
from vedo.applications import AnimationPlayer

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

def lumped_mass_3n_to_n_x_only(M_3n):
    n_vertices = M_3n.shape[0] // 3
    lumped_masses = np.zeros(n_vertices)
    for i in range(n_vertices):
        lumped_masses[i] = M_3n[3*i, 3*i]
    M_n = np.diag(lumped_masses)
    return M_n

def create_eigenmode_weights(K, M, n=10):
    M = lumped_mass_3n_to_n_x_only(M)
    invM = np.linalg.inv(M)
    invM = np.sqrt(invM)

    KW = sum_diagonal_blocks(reorder_hessian_xyz_to_xxyyzz(K))
    eigvals, eigvecs = eigsh(invM * KW, k=n, which='SM')
    return eigvecs

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
    
    EMs = create_eigenmode_weights(K, M, 30).T
    
    pc = Points(V, r=8)
    pc.pointdata["scalars"] = EMs[0]
    pc.cmap("viridis").add_scalarbar(title="weight")

    def update_scene(i: int):
        pc.pointdata["scalars"] = EMs[i]
        plt.render()
        
    camera_settings = dict(
        pos=(10, 0, 0),
        focalPoint=(0, 0, 0),
        viewup=(0, 0, 1)
    )

    plt = AnimationPlayer(update_scene, irange=[0,len(EMs)], loop=True, dt=17)
    plt += [pc]
    plt.set_frame(0)
    plt.show(camera=camera_settings)
    plt.close()
