import numpy as np
import igl

from scipy.sparse.linalg import eigsh
import scipy.sparse
from vedo import Points, Plotter

if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
    
from src import *
from sim import *

def create_eigenmode_weights(K, M, L, leak=None, n=10):
    M = lumped_mass_3n_to_n(M)
    KW = sum_diagonal_blocks(reorder_xyzxyz_to_xxyyzz(K))
    
    if leak is not None:
        mult = 1 / (1e-8 + leak)
        KW = KW.multiply(mult)
        
    EVs, EMs = eigsh(KW, k=n, M=M, sigma=0.0)
    EMs = explicit_smooth(EMs, L)
    
    if leak is not None:
        EMs = EMs * leak[:, np.newaxis]
    
    return EVs, EMs

def explicit_smooth(f, L, steps=100):
    L = (L + L.T) / 2
    L.setdiag(0)
    row_sums = np.array(L.sum(axis=1)).flatten()
    nonzero_rows = row_sums != 0
    scale = sp.diags(0.499/row_sums[nonzero_rows], 0)
    L = scale @ L
    L.setdiag(0.499)
    
    for _ in range(steps):
        f = L @ f
    return f

def visualize_eigenmodes(V, EMs):
    num_modes = EMs.shape[0]
    rows = int(np.sqrt(num_modes))
    cols = int(np.ceil(num_modes / rows))

    plotter = Plotter(shape=(rows, cols), title="Eigenmodes")

    for i in range(num_modes):
        pc = Points(V, c='green', r=4)
        scalars = EMs[i]
        pc.pointdata["scalars"] = scalars
        
        max_mag = max(abs(scalars.min()), abs(scalars.max()))
        pc.cmap("coolwarm", vmin=-max_mag, vmax=max_mag).add_scalarbar(title="weight")
        plotter.show(pc, at=i, interactive=False)
    plotter.interactive().close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='generate eigenmodes')
    parser.add_argument('--input', '-i', type=str, default='data/wrs/wrs_out/wrs.gltf', help='json input path')
    parser.add_argument('--n_modes', '-n', type=int, default=50, help='number of modes')
    
    args = parser.parse_args()
    gltf_path = args.input
    
    V0, F0, V, T, F, TF_list, C, BE, W, dt = read_gltf_data(gltf_path)
    
    lambda_ = 1
    mu = 1e-5
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
    leak = create_mask_matrix(V, T, F, C, BE, mask_type="rig")
    
    print("calculating eigenmodes")
    L = igl.cotmatrix(V, T)
    EVs, EMs = create_eigenmode_weights(K, M, L, leak, n=10)
    
    print("visualizing eigenmodes")
    visualize_eigenmodes(V, EMs.T)

