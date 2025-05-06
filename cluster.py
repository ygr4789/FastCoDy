import sys
import numpy as np
import igl
import scipy.sparse as sp
import scipy.sparse.linalg as spla

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
    lambda_, mu = emu_to_lame(YM, pr)

    params = np.zeros((T.shape[0], 2))
    params[:, 0] = 0.5 * lambda_
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
    M *= 1000

    # === Initial Transformation ===
    TF = TF_list[0]
    Vr = VM @ TF
    U = Vr - V

    VCol = vectorize(V)
    Vn_list = []

    # ---------------------------------------------------
    if not original_motion:
        print(f"compiling njax...")
        # G = linear_tetmesh_arap_dq(V, T, VCol, dX, vol, params)
        K = linear_tetmesh_arap_dq2(V, T, VCol, dX, vol, params)
        # e = linear_tetmesh_arap_q(V, T, VCol, dX, vol, params)
        
        # === Poisson Constraint Mask ===
        phi = create_mask_matrix(V, T, C, BE, 'lin')
        # phi = create_mask_matrix(V, T, C, BE)

        # === Constraint system ===
        EMW = create_eigenmode_weights(K, M, 10)
        A = lbs_matrix_column(V, EMW)
        J = lbs_matrix_column(V, W)
        Aeq = J.T @ M @ phi
        # Aeq = J.T @ M @ phi
        Beq = np.zeros(Aeq.shape[0])
    # ---------------------------------------------------
    
    start = time.time()
    solver = arap_solver(V, T, J, A, Aeq, params[:, 0], dt*dt)
    z = np.zeros(A.shape[1])
    p = TF.T.flatten()
    st = sim_state(z, p)
    
    for ai, TF in enumerate(TF_list):
        p = TF.T.flatten()
        print(f"frame: {ai}")
        st.update(z, p)
        z = solver.step(z, p, st, Beq)
        
        # Store result
        VCol = J * p + A * z
        Vn = matrixize(VCol)
        Vn_list.append(Vn)

    # np.save('elephant',Vn_list)
    end = time.time()
    print(f"Total : {end - start:.5f} sec")

    return Vn_list, F

def render_animation(Vn_list, F):
    mesh = Mesh([Vn_list[0], F])

    def update_scene(i: int):
        # update block and spring position at frame i
        mesh.points = Vn_list[i]
        plt.render()
        
    camera_settings = dict(
        pos=(10, 0, 0),           # Camera position
        focalPoint=(0, 0, 0),    # Look-at target
        viewup=(0, 0, 1)         # "Up" direction
    )

    plt = AnimationPlayer(update_scene, irange=[0,len(Vn_list)], loop=True, dt=33)
    plt += [mesh]
    plt.set_frame(0)
    # plt.show()
    plt.show(camera=camera_settings)
    plt.close()

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
        Vn_list, F = create_cody_animation(input_path, original_motion)
        if output_path:
            np.savez(output_path, anim=Vn_list, faces=F)
            
    render_animation(Vn_list, F)
