import sys
import numpy as np
import igl
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

from scipy.sparse.linalg import minres
from sksparse.cholmod import cholesky

from src import *
from sim import *

from solver import *

from vedo import Mesh
from vedo.applications import AnimationPlayer

import time

def create_cody_animation(json_path, original_motion=False):
    # Load mesh and animation data
    V, T, F, C, PI, BE, W, TF_list, dt, YM, pr, scale, physic_model = read_json_data(json_path)
    # lambda_, mu = emu_to_lame(YM, pr)
    lambda_ = 1
    mu = 5e-5 # lin mic cluster 50 50
    # mu = 1e-5 # poi ele cluster 50 50

    params = np.zeros((T.shape[0], 2))
    params[:, 0] = lambda_
    params[:, 1] = mu

    # === Boundary faces and LBS Matrix ===
    F = igl.boundary_facets(T)
    F = F[:, ::-1]  # reverse each face row to match
    VM = igl.lbs_matrix(V, W)

    # === Volume and differential operator ===
    vol = igl.volume(V, T)
    dX = linear_tetmesh_dphi_dX(V, T)

    # === Mass Matrix ===
    M = lumped_mass_matrix(V, T)

    # === Initial Transformation ===
    TF = TF_list[0]
    Vr = VM @ TF
    U = Vr - V

    VCol = vectorize(V)
    Vn_list = []
    V0n_list = [] = []
    # ---------------------------------------------------
    if not original_motion:
        print(f"compiling njax...")
        # G = linear_tetmesh_arap_dq(V, T, VCol, dX, vol, params)
        K = linear_tetmesh_arap_dq2(V, T, VCol, dX, vol, params)
        # e = linear_tetmesh_arap_q(V, T, VCol, dX, vol, params)
        
        # === Poisson Constraint Mask ===
        phi, leak = create_mask_matrix(V, T, C, BE, 'lin')
        # phi, leak = create_mask_matrix(V, T, C, BE)

        # === Constraint system ===
        J = lbs_matrix_column(V, W)
        # Jleak = M @ phi @ J
        Jw = W.T @ phi
        
        _, EMW = create_eigenmode_weights(K, M, Jw, n=50)
        B = lbs_matrix_column(V, EMW / _ ** 0.5)
        B = leak @ B
        
        G = create_group_matrix(_, EMW, T, vol, n_clusters=50)
        G_exp = create_exploded_group_matrix(G)
    # ---------------------------------------------------
    
    start = time.time()
    
    solver = arap_solver(V, T, J, B, G_exp, params[:, 1], dt*dt)
    z = torch.zeros((B.shape[1], 1), dtype=torch.float64, device=solver.device)
    p = torch.from_numpy(TF.T.flatten()).reshape(-1, 1).to(solver.device)
    st = sim_state(z, p)
    
    for ai, TF in enumerate(TF_list):
        p = torch.from_numpy(TF.T.flatten()).reshape(-1, 1).to(solver.device)
        print(f"frame: {ai}")
        z = solver.step(z, p, st)
        st.update(z, p)
        
        # Store result
        V0Col = J * p.cpu().numpy().flatten()
        UCol = B * z.cpu().numpy().flatten()
        VCol = V0Col + UCol
        
        V0n = matrixize(V0Col)
        Vn = matrixize(VCol)
        print(f"  Average Offset : {torch.mean(torch.abs(z)).item():.2e}")
        
        V0n_list.append(V0n)
        Vn_list.append(Vn)

    # np.save('elephant',Vn_list)
    end = time.time()
    print(f"Total time    : {end - start:.2f} sec")
    print(f"Average time  : {(end - start) / len(TF_list):.2f} sec")

    return Vn_list, V0n_list, F

def render_animation(Vn_list, V0n_list, F):
    mesh = Mesh([Vn_list[0], F])
    mesh0 = Mesh([V0n_list[0], F], alpha=0.1, c='blue')

    def update_scene(i: int):
        # update block and spring position at frame i
        mesh.points = Vn_list[i]
        mesh0.points = V0n_list[i]
        plt.render()
        
    camera_settings = dict(
        pos=(10, 0, 0),           # Camera position
        focalPoint=(0, 0, 0),    # Look-at target
        viewup=(0, 0, 1)         # "Up" direction
    )

    plt = AnimationPlayer(update_scene, irange=[0,len(Vn_list)], loop=True, dt=33)
    plt += [mesh, mesh0]
    plt.set_frame(0)
    # plt.show()
    plt.show(camera=camera_settings)
    # plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='generate complementary dynamics secondary motion')
    parser.add_argument('--input', '-i', type=str, required=False, default='examples/sphere/sphere.json', help='json input path')
    parser.add_argument('--output', '-o', type=str, required=False, help='output path')
    parser.add_argument('--data', '-d', type=str, required=False, help='prebuilt data')
    parser.add_argument('--none', '-n', action="store_true", required=False, default=False, help='no secondary motion')
    
    args = parser.parse_args()
    input_path = args.input
    output_path = args.output
    data = args.data
    original_motion = args.none
    
    if data:
        data = np.load(data)
        Vn_list = data["anim"]
        F = data["faces"]
    elif input_path:
        # Vn_list, F = create_cody_animation(input_path)
        Vn_list, V0n_list, F = create_cody_animation(input_path, original_motion)
        if output_path:
            np.savez(output_path, anim=Vn_list, faces=F)
            
    render_animation(Vn_list, V0n_list, F)
