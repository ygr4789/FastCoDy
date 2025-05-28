import numpy as np
import igl

import scipy.sparse as sp
from scipy.linalg import null_space
from scipy.sparse.linalg import eigsh

from vedo import Points, Plotter, TetMesh, show
# from vedo.applications import AnimationPlayer

if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
    
from src import *
from sim import *

def solve_generalized_eig(H, M, J, n=10, tol=1e-12):
    if J is None:
        return eigsh(H, k=n, M=M, which='SM')
    
    # null_J = null_space(J.todense())
    null_J = null_space(J)

    # Project H and M into the null space
    H_proj = null_J.T @ H @ null_J
    M_proj = null_J.T @ M @ null_J

    # Solve generalized eigenproblem in reduced space
    eigvals, eigvecs = eigsh(H_proj, k=n, M=M_proj, sigma=0.0)

    # Recover full-space eigenvectors
    eigvecs = null_J @ eigvecs
    return eigvals, eigvecs

def create_eigenmode_weights(K, M, J=None, n=10):
    M = lumped_mass_3n_to_n(M)
    KW = sum_diagonal_blocks(reorder_xyzxyz_to_xxyyzz(K))
    return solve_generalized_eig(KW, M, J, n)

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
        scalars = EMs[i]
        pc.pointdata["scalars"] = scalars
        
        max_mag = max(abs(scalars.min()), abs(scalars.max()))
        pc.cmap("coolwarm", vmin=-max_mag, vmax=max_mag).add_scalarbar(title="weight")
        # plotter.show(pc, at=i, interactive=False, camera=camera_settings)
        plotter.show(pc, at=i, interactive=False)
    plotter.interactive().close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='generate eigenmodes')
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
    
    print("calculating arap")
    K = linear_tetmesh_arap_dq2(V, T, VCol, dX, vol, params)
    print("calculating mass")
    M = lumped_mass_matrix(V, T)
    print("calculating lbs")
    J = lbs_matrix_column(V, W)
    
    print("calculating mask")
    phi, leak = create_mask_matrix(V, T, C, BE)
    Jw = W.T @ phi
    
    print("calculating eigenmodes")
    EVs, EMs = create_eigenmode_weights(K, M, Jw, n=50)
    EMs = EMs / (EVs ** 0.5)
    
    print("visualizing eigenmodes")
    visualize_eigenmodes(V, EMs.T)

