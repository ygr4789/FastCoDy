import igl
import numpy as np
import scipy as sp
from vedo import Mesh, Plotter, Line, Points

import numpy as np
from scipy.spatial import Delaunay

def point_in_tet(p, tet_vertices, tol=1e-8):
    v0, v1, v2, v3 = tet_vertices
    T = np.column_stack((v1 - v0, v2 - v0, v3 - v0))
    try:
        lambdas = np.linalg.solve(T, p - v0)
    except np.linalg.LinAlgError:
        return False  # singular matrix → degenerate tet

    lambdas = np.append(lambdas, 1 - np.sum(lambdas))
    # Check if all in [0, 1] with tolerance
    return np.all(lambdas >= -tol) and np.all(lambdas <= 1 + tol)
    
def refine_tetmesh(V, T, C, BE, tol=1e-8):
    V = V.tolist()
    T = T.tolist()
    b = []
    bc = []

    bone_midpoints = []
    for i in range(BE.shape[0]):
        # Midpoint of bone i
        midpoint = (C[BE[i][0]] + C[BE[i][1]]) / 2.0
        bone_midpoints.append(midpoint)

    for i, p in enumerate(bone_midpoints):
        dists = np.linalg.norm(np.array(V) - p, axis=1)
        if np.min(dists) < tol:
            continue  # already exists

        # Find tetrahedron that contains p
        found = False
        for t_idx, tet in enumerate(T):
            verts = np.array([V[i] for i in tet])
            if point_in_tet(p, verts):
                found = True
                break

        if not found:
            print("Control point not inside mesh:", p)
            continue

        p_idx = len(V)
        b.append(p_idx)
        
        V.append(p.tolist())
        one_hot = np.zeros(len(bone_midpoints))
        one_hot[i] = 1.0
        bc.append(one_hot)

        # Remove old tet
        i0, i1, i2, i3 = T.pop(t_idx)

        # Add 4 new tetrahedra by connecting p to each face of the original tet
        T.extend([
            [p_idx, i1, i2, i3],
            [p_idx, i0, i3, i2],
            [p_idx, i0, i1, i3],
            [p_idx, i0, i2, i1],
        ])

    return np.array(V), np.array(T), np.array(b), np.array(bc)

def create_weights(V, T, C, BE):
    original_v_num = V.shape[0]
    V, T, b, bc = refine_tetmesh(V, T, C, BE)
    # _, b, bc = igl.boundary_conditions(V, T, C, np.array([],dtype=np.int64), BE, np.array([],dtype=np.int64))
    
    bbw_solver = igl.BBW()
    W = bbw_solver.solve(V, T, b, bc)
    return W[:original_v_num]

def visualize_weights(V, F, W, C, BE, flag="surface"):
    num_bones = BE.shape[0]
    rows = int(np.sqrt(num_bones))
    cols = int(np.ceil(num_bones / rows))

    plotter = Plotter(shape=(rows, cols), title="BBW Weights for Each Bone")

    for i in range(num_bones):
        p0 = C[BE[i][0]]
        p1 = C[BE[i][1]]
        highlight = Points([(p0+p1)/2], c='green', r=4)
        bone_line = Line(p0, p1, c="red", lw=2)
        
        if flag=="surface":
            mesh = Mesh([V, F])
            mesh.alpha(0.5)
            mesh.pointdata["weights"] = W[:, i]
            mesh.cmap("viridis").add_scalarbar(title=f"Bone {i}")
            plotter.show(mesh, bone_line, highlight, at=i, interactive=False)
        
        else:
            pc = Points([V], c='green', r=4)
            pc.pointdata["scalars"] = W[:, i]
            pc.cmap("viridis").add_scalarbar(title="weight")
            plotter.show(pc, bone_line, highlight, at=i, interactive=False)
        
    plotter.interactive().close()

def save_as_dmat(output_path, matrix):
    n, m = matrix.shape  # n rows, m columns
    with open(output_path, 'w') as f:
        f.write(f"{m} {n}\n")  # First line: m (cols) then n (rows)
        for val in matrix.flatten(order='F'):  # Column-major order
            f.write(f"{val}\n")

def read_properties(mesh_path, tgf_path):
    V, T, F = igl.read_mesh(mesh_path)
    C, E, PI, BE, CE, PE = igl.read_tgf(tgf_path)
    return V, T, F, C, BE

def create_dmat(mesh_path, tgf_path, output_path):
    V, T, F, C, BE = read_properties(mesh_path, tgf_path)
    W = create_weights(V, T, C, BE)
    save_as_dmat(output_path, W)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='create BBW weights from mesh and tgf')
    parser.add_argument('--mesh', '-m', type=str, required=True, help='mesh file path')
    parser.add_argument('--tgf', '-t', type=str, required=True, help='tgf file path')
    parser.add_argument('--flag', '-f', type=str, required=False, default='surface', help='plot surface/pointcloud')
    
    args = parser.parse_args()
    mesh_path = args.mesh
    tgf_path = args.tgf
    flag = args.flag
    
    V, T, F, C, BE = read_properties(mesh_path, tgf_path)
    W = create_weights(V, T, C, BE)
    visualize_weights(V, F, W, C, BE, flag)