import numpy as np
import scipy.sparse as sp
import igl

if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
    
from src import *
from sim import *

def create_mask_matrix(V, T, F, C, BE, mask_type="rig"):
    M = compute_vertex_voronoi_volumes(V, T)
    
    if mask_type == "rig":
        Z = rig_ortho_weights(V, T, F, C, BE)
    elif mask_type == "poi":
        Z = poisson_weights(V, T, M)
    else:
        raise ValueError(f"Invalid mask type: {mask_type}")
    
    return Z
    
def rig_ortho_weights(V, T, F, C, BE):
    # Find tetrahedrons that intersect with bone segments
    sample_points = []
    # Precompute inverse matrices for all tetrahedra
    tet_matrices = np.zeros((len(T), 4, 4))
    tet_matrices_inv = np.zeros((len(T), 4, 4))
    for i, tet in enumerate(T):
        tet_vertices = V[tet]
        M = np.vstack([tet_vertices.T, np.ones(4)])
        tet_matrices[i] = M
        try:
            tet_matrices_inv[i] = np.linalg.inv(M)
        except:
            tet_matrices_inv[i] = np.nan
            
    # Generate sample points along bone segments
    sample_len = 0.05
    for bone in BE:
        start = C[bone[0]]
        end = C[bone[1]]
        direction = end - start
        length = np.linalg.norm(direction)
        num_samples = int(length / sample_len) + 2
        
        t = np.linspace(0, 1, num_samples)
        points = start + np.outer(t, end - start)
        sample_points.extend(points)
            
    sample_points = np.array(sample_points)
    
    # Batch check if points are in tetrahedra
    in_tet_mask = np.zeros(len(sample_points), dtype=bool)
    points_homog = np.hstack([sample_points, np.ones((len(sample_points), 1))])
    
    # Store which tet each point belongs to
    point_to_tet = -np.ones(len(sample_points), dtype=int)
    
    for i, inv_mat in enumerate(tet_matrices_inv):
        if np.any(np.isnan(inv_mat)):
            continue
        # Compute barycentric coordinates for all points at once
        bary = points_homog @ inv_mat.T
        # Point is inside if all barycentric coords are between 0 and 1
        inside = np.all((bary >= 0) & (bary <= 1), axis=1)
        in_tet_mask |= inside
        # Store tet index for points inside this tet
        point_to_tet[inside] = i
                    
    original_v_num = V.shape[0]
    
    visited_tets = np.zeros(len(T), dtype=bool)
    V_refine = V.copy()
    T_refine = T.copy()
    
    for V_new, ti in zip(sample_points, point_to_tet):
        if ti == -1:
            continue
        if visited_tets[ti]:
            continue
        visited_tets[ti] = True
        
        V_refine = np.vstack([V_refine, V_new])
        i_refine = V_refine.shape[0] - 1
        T1 = [T[ti][0], T[ti][1], T[ti][2], i_refine]
        T2 = [T[ti][0], T[ti][1], T[ti][3], i_refine]
        T3 = [T[ti][2], T[ti][3], T[ti][0], i_refine]
        T4 = [T[ti][1], T[ti][3], T[ti][2], i_refine]
        T_refine = np.vstack([T_refine, T1, T2, T3, T4])
        
    devided_tets = np.unique(point_to_tet[point_to_tet != -1])
    remaining_tets = np.setdiff1d(np.arange(len(T_refine)), devided_tets)
    T_refine = T_refine[remaining_tets]
        
    L = igl.cotmatrix(V_refine, T_refine)
    M = compute_vertex_voronoi_volumes(V_refine, T_refine)
    
    b_bone = np.arange(original_v_num, V_refine.shape[0])
    bc_bone = np.zeros((len(b_bone), 1))

    Q = -L
    l = -M
    Aeq = sp.csc_matrix((0, Q.shape[0]), dtype=Q.dtype)
    Beq = np.array([])
    _, Z = igl.min_quad_with_fixed(Q, l, b_bone, bc_bone, Aeq, Beq, False)
    Z = np.abs(Z)
    if Z.max() > 0: Z = Z / np.max(Z)
    Z = Z[:original_v_num]
    
    user_leak = draw_surface_mask(V, T, F, Z)
    
    b_user_fixed = np.where(user_leak < -0.5)[0]
    b_tmp = np.concatenate([b_bone, b_user_fixed])
    b_tmp = np.unique(b_tmp)
    bc_tmp = np.zeros((len(b_tmp), 1))
    
    surface_ids = igl.boundary_facets(T_refine)
    surface_ids = np.unique(surface_ids.flatten())
    b_non_user_fixed = np.setdiff1d(surface_ids, b_user_fixed)
    b_tmp = np.concatenate([b_tmp, b_non_user_fixed])
    bc_tmp = np.concatenate([bc_tmp, np.ones((len(b_non_user_fixed), 1))])
    
    _, Z = igl.min_quad_with_fixed(Q, np.zeros_like(l), b_tmp, bc_tmp, Aeq, Beq, False)
    Z = np.abs(Z)
    if Z.max() > 0: Z = Z / np.max(Z)
    
    fixed_ids = np.where(Z < 0.01)[0]
    b_fixed = fixed_ids
    bc_fixed = np.zeros((len(b_fixed), 1))
    _, Z = igl.min_quad_with_fixed(Q, l, b_fixed, bc_fixed, Aeq, Beq, False)
    Z = np.abs(Z)
    if Z.max() > 0: Z = Z / np.max(Z)
    Z = Z[:original_v_num]
    
    Z_corrected = gamma_correction(V, T, F, Z)
    Z_corrected[Z_corrected < 0.01] = 0.0
    
    b_surface = np.setdiff1d(surface_ids, b_fixed)
    bc_surface = Z_corrected[b_surface].reshape(-1, 1)
    
    b_final = np.concatenate([b_fixed, b_surface])
    bc_final = np.concatenate([bc_fixed, bc_surface])
    
    _, Z = igl.min_quad_with_fixed(Q, np.zeros_like(l), b_final, bc_final, Aeq, Beq, False)
    Z = np.abs(Z)
    if Z.max() > 0: Z = Z / np.max(Z)
    Z = Z[:original_v_num]
    
    return Z

def poisson_weights(V, T, M):
    L = igl.cotmatrix(V, T)
    F = igl.boundary_facets(T)
    b = np.unique(F.flatten())
    bc = np.zeros((len(b), 1))

    Q = -L
    l = -M
    Aeq = sp.csc_matrix((0, Q.shape[0]), dtype=Q.dtype)
    Beq = np.array([])  

    _, Z = igl.min_quad_with_fixed(Q, l, b, bc, Aeq, Beq, False)
    Z = 1 - Z / np.max(Z)
    
    return Z

def draw_surface_mask(V, T, F, leak):
    from vedo import Mesh, Plotter, build_lut, Text2D
    
    plotter = Plotter()
    paint_radius = 0.1
    leak = leak.copy()
    
    def update_radius(widget, event):
        nonlocal paint_radius
        paint_radius = widget.value
        
    def on_key_press(evt):
        if evt.keypress != "space":
            return
        
        point = evt.picked3d
        if point is None:
            return
        
        dists = np.linalg.norm(V - point, axis=1)
        mask = dists < paint_radius
        leak[mask] = -1.0
        
        mesh.pointdata["scalars"] = leak
        mesh.cmap(lut, vmin=0.0, vmax=1.0)
        
        plotter.render()
    
    def finish_painting(obj, ename):
        plotter.close()
    
    lut = build_lut(
        [ (0.0, 'skyblue'), (1.0, 'pink') ],
        vmin=0.0,
        vmax=1.0,
        below_color='b',
        interpolate=True,
    )
    
    mesh = Mesh([V, F])
    mesh.pointdata["scalars"] = leak
    mesh.cmap(lut).add_scalarbar(pos=(0.85, 0.5), title="weight")
    
    # Add UI controls
    plotter.add_slider(
        update_radius,
        xmin=0.01,
        xmax=0.5,
        value=paint_radius,
        pos="bottom-right",
        title="Brush Radius"
    )
    
    plotter.add_button(
        finish_painting,
        pos=(0.2, 0.08),   # x,y fraction from bottom left corner
        states=["    Apply    "],  # text for each state
        c=["w"],     # font color for each state
        bc=["gray"],  # background color for each state
        size=20,
    )
    
    # Add mouse event handlers
    plotter.add_callback('KeyPress', on_key_press)
    plotter.add(Text2D("Press space to paint fixed surface", pos='top-left'))
    plotter.show(mesh, interactive=True)
    
    return leak

def gamma_correction(V, T, F, leak):
    from vedo import Mesh, Plotter, build_lut
    
    plotter = Plotter()
    gamma_slider = 1.0
    leak = leak.copy()
    
    mesh = Mesh([V, F])
    gamma_corrected = np.power(leak, gamma_slider)
    mesh.pointdata["scalars"] = gamma_corrected
    
    lut = build_lut([ (0.0, 'skyblue'), (1.0, 'red') ], interpolate=True, vmin=0.01, vmax=1.0, below_color='blue')
    mesh.cmap(lut)
    
    def update_gamma(widget, event):
        nonlocal gamma_slider, gamma_corrected
        gamma_slider = widget.value
        gamma_value = gamma_slider
        if gamma_slider > 1.0:
            gamma_value = np.power(gamma_slider, 5.0)
        gamma_corrected = np.power(leak, gamma_value)
        mesh.pointdata["scalars"] = gamma_corrected
        mesh.cmap(lut)
        plotter.render()
    
    def finish_correction(obj, ename):
        plotter.close()
    
    # Add UI controls
    plotter.add_slider(
        update_gamma,
        xmin=0.1,
        xmax=2.0,
        value=gamma_slider,
        pos="bottom-right",
        title="gamma"
    )
    
    plotter.add_button(
        finish_correction,
        pos=(0.2, 0.08),   # x,y fraction from bottom left corner
        states=["    Apply    "],  # text for each state
        c=["w"],     # font color for each state
        bc=["gray"],  # background color for each state
        size=20,
    )
    
    plotter.show(mesh, interactive=True)
    
    return gamma_corrected

def visualize_mask(V, T, F, leak):
    from vedo import Mesh, Plotter, build_lut, Points
    
    plotter = Plotter()
    mesh = Mesh([V, F], alpha=0.2)
    pc = Points(V)
    pc.pointdata["scalars"] = leak
    pc.cmap(build_lut([ (0.0, 'w'), (1.0, 'r') ], interpolate=True), vmin=0.0, vmax=1.0)
    plotter.show(mesh, pc)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='generate mask matrix')
    parser.add_argument('--input', '-i', type=str, default='data/wrs/wrs_out/wrs.gltf', help='json input path')
    parser.add_argument('--type', '-t', type=str, default='rig', help='mask type')
    args = parser.parse_args()
    gltf_path = args.input
    mask_type = args.type

    V0, F0, V, T, F, TF_list, C, BE, W, dt = read_gltf_data(gltf_path)
    
    leak = create_mask_matrix(V, T, F, C, BE, mask_type=mask_type)
    visualize_mask(V, T, F, leak)
