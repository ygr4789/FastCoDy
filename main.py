import sys
import numpy as np
import igl
import torch

from src import *
from sim import *

from solver import *

from vedo import Mesh, Points, Plotter
from vedo.applications import AnimationPlayer

import time

def prepare_cody_simulation(gltf_path):
    """Initialize and prepare all necessary data for Cody simulation.
    
    Args:
        gltf_path (str): Path to the JSON file containing mesh and animation data
        
    Returns:
        tuple: (solver, initial_state, V, F, VM, TF_list, dt, J, B) - All necessary components for simulation
    """
    # Load mesh and animation data
    V0, F0, V, T, F, TF_list, C, BE, W, dt = read_gltf_data(gltf_path)
    
    print("Loaded gltf data from: ", gltf_path)
    print("# of vertices: ", V.shape[0])
    print("# of tetrahedrons: ", T.shape[0])
    print("# of frames: ", len(TF_list))
    print("# of control points: ", len(C))
    print("# of bone edges: ", len(BE))
    
    lambda_ = 1
    mu = 1e-5
    # mu = 3e-6

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
    leak = create_mask_matrix(V, T, F, C, BE, "rig")
    
    # === Constraint system ===
    J = lbs_matrix_column(V, W)
    L = igl.cotmatrix(V, T)
    EVs, EMs = create_eigenmode_weights(K, M, L, leak, n=50)
    B = lbs_matrix_column(V, EMs)
    
    GEVs, GEMs = create_eigenmode_weights(K, M, L, n=50)
    G = create_group_matrix(GEVs, GEMs, T, vol, n_clusters=50)
    G_exp = create_exploded_group_matrix(G)
    
    solver = arap_solver(V, T, J, B, G_exp, params[:, 1], dt*dt)
    z = torch.zeros((B.shape[1], 1), dtype=torch.float64, device=solver.device)
    p = torch.from_numpy(TF.T.flatten()).reshape(-1, 1).to(solver.device)
    initial_state = sim_state(z, p)
    
    return solver, initial_state, V0, F0, V, T, F, TF_list, dt, J, B, z

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
        tuple: (Vn_list, Vori_list) - Lists of vertex positions for each frame
    """
    st = initial_state
    Vn_list = []
    Vori_list = []
    
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
        
        Vori_list.append(V0n)
        Vn_list.append(Vn)

    end = time.time()
    print(f"Total time    : {end - start:.2f} sec")
    print(f"Average time  : {(end - start) / len(TF_list):.2f} sec")
    
    return Vn_list, Vori_list

def create_cody_animation(gltf):
    """Main function to create Cody animation.
    
    Args:
        gltf (str): Path to the JSON file containing mesh and animation data
        
    Returns:
        tuple: (Vn_list, Vori_list, F) - Lists of vertex positions and faces
    """
    solver, initial_state, V0, F0, V, T, F, TF_list, dt, J, B, z = prepare_cody_simulation(gltf)
    Vn_list, Vori_list = run_cody_simulation(solver, initial_state, TF_list, J, B, z)
    
    V0toV = surface_cast_barycentric(V0, F0, V, T)
    
    Vn_list = [V0toV @ Vn_list[i] for i in range(len(Vn_list))]
    Vori_list = [V0toV @ Vori_list[i] for i in range(len(Vori_list))]
    
    texture_path, uvs = get_texture_info(gltf)
    
    return Vn_list, Vori_list, F0, texture_path, uvs

def render_animation(Vn_list, Vori_list, F0, texture_path, uvs):
    current_frame = 0
    view_original = False
    mesh = Mesh([Vn_list[0], F0]).texture(texture_path, tcoords=uvs)
    mesh.compute_normals()
    
    def update_scene(i: int):
        nonlocal current_frame
        current_frame = i
        mesh.points = Vn_list[i] if not view_original else Vori_list[i]
        mesh.compute_normals()
        plt.render()
        
    def toggle_mesh(obj, ename):
        nonlocal view_original
        view_original = not view_original
        button.switch()
        mesh.points = Vori_list[current_frame] if view_original else Vn_list[current_frame]
        
    plt = AnimationPlayer(update_scene, irange=[0,len(Vn_list)], loop=True, dt=33)
    
    button = plt.add_button(
        toggle_mesh,
        pos=(0.2, 0.92),
        states=["    Deformed    ", "    Original    "],
        c=["w", "w"],
        bc=["red", "blue"],
        size=20,
    )
    
    plt.add(mesh)
    plt.set_frame(current_frame)
    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='generate complementary dynamics secondary motion')
    parser.add_argument('--input', '-i', type=str, required=False, default='data/wrs/wrs_out/wrs.gltf', help='gltf input path')
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
        Vn_list, Vori_list, F0, texture_path, uvs = create_cody_animation(input_path)
        if output_path:
            np.savez(output_path, anim=Vn_list, anim_ori=Vori_list, faces=F0)
            
    render_animation(Vn_list, Vori_list, F0, texture_path, uvs)
