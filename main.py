import sys
import numpy as np
import igl
import torch

from src import *
from sim import *

from solver import *

from vedo import Mesh
from vedo.applications import AnimationPlayer

import time

def prepare_cody_simulation(json_path):
    """Initialize and prepare all necessary data for Cody simulation.
    
    Args:
        json_path (str): Path to the JSON file containing mesh and animation data
        
    Returns:
        tuple: (solver, initial_state, V, F, VM, TF_list, dt, J, B) - All necessary components for simulation
    """
    # Load mesh and animation data
    V, T, F, C, PI, BE, W, TF_list, dt, YM, pr, scale, physic_model = read_json_data(json_path)
    # lambda_, mu = emu_to_lame(YM, pr)
    lambda_ = 1
    # mu = 5e-5 # lin mic cluster 50 50
    mu = 1e-5 # poi ele cluster 50 50

    params = np.zeros((T.shape[0], 2))
    params[:, 0] = lambda_
    params[:, 1] = mu

    # === Boundary faces and LBS Matrix ===
    F = igl.boundary_facets(T)
    F = F[:, ::-1]  # reverse each face row to match

    # === Volume and differential operator ===
    vol = igl.volume(V, T)
    dX = linear_tetmesh_dphi_dX(V, T)

    # === Mass Matrix ===
    M = lumped_mass_matrix(V, T)

    # === Initial Transformation ===
    TF = TF_list[0]
    VCol = vectorize(V)
    
    print(f"precomputing matrices...")
    K = linear_tetmesh_arap_dq2(V, T, VCol, dX, vol, params)
    
    # === Poisson Constraint Mask ===
    phi, leak = create_mask_matrix(V, T, C, BE)

    # === Constraint system ===
    J = lbs_matrix_column(V, W)
    Jw = W.T @ phi
    
    _, EMW = create_eigenmode_weights(K, M, Jw, n=50)
    B = lbs_matrix_column(V, EMW)
    B = leak @ B
    
    G = create_group_matrix(_, EMW, T, vol, n_clusters=50)
    G_exp = create_exploded_group_matrix(G)
    
    solver = arap_solver(V, T, J, B, G_exp, params[:, 1], dt*dt)
    z = torch.zeros((B.shape[1], 1), dtype=torch.float64, device=solver.device)
    p = torch.from_numpy(TF.T.flatten()).reshape(-1, 1).to(solver.device)
    initial_state = sim_state(z, p)
    
    return solver, initial_state, F, TF_list, dt, J, B, z

def run_cody_simulation(solver, initial_state, TF_list, J, B, z):
    """Run the Cody simulation steps.
    
    Args:
        solver: The ARAP solver instance
        initial_state: Initial simulation state
        V: Initial vertex positions
        VM: LBS matrix
        TF_list: List of transformation matrices for each frame
        J: LBS matrix column
        B: Basis matrix for deformation
        z: Initial deformation state
    Returns:
        tuple: (Vn_list, V0n_list) - Lists of vertex positions for each frame
    """
    st = initial_state
    Vn_list = []
    V0n_list = []
    
    start = time.time()
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

    end = time.time()
    print(f"Total time    : {end - start:.2f} sec")
    print(f"Average time  : {(end - start) / len(TF_list):.2f} sec")
    
    return Vn_list, V0n_list

def create_cody_animation(json_path):
    """Main function to create Cody animation.
    
    Args:
        json_path (str): Path to the JSON file containing mesh and animation data
        
    Returns:
        tuple: (Vn_list, V0n_list, F) - Lists of vertex positions and faces
    """
    solver, initial_state, F, TF_list, dt, J, B, z = prepare_cody_simulation(json_path)
    Vn_list, V0n_list = run_cody_simulation(solver, initial_state, TF_list, J, B, z)
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
    
    args = parser.parse_args()
    input_path = args.input
    output_path = args.output
    data = args.data
    
    if data:
        data = np.load(data)
        Vn_list = data["anim"]
        F = data["faces"]
    elif input_path:
        Vn_list, V0n_list, F = create_cody_animation(input_path)
        if output_path:
            np.savez(output_path, anim=Vn_list, faces=F)
            
    render_animation(Vn_list, V0n_list, F)
