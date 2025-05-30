import numpy as np
import igl

from vedo import Plotter, TetMesh, Points, show
# from vedo.applications import AnimationPlayer

import scipy.sparse as sp
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from sklearn.cluster import KMeans

if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
    
from src import *
from sim import *

def create_group_matrix(eigvals, eigvecs, T, Vol, n_clusters=100):
    EW = eigvals
    EV = eigvecs
    EW2 = EW ** 0.5
    EVdW2 = EV / EW2
    # EVdW2 = EV
    
    n_tets = T.shape[0]
    n_modes = EV.shape[1]
    EVdW2_tet = np.zeros((n_tets, n_modes))
    
    for i in range(n_tets):
        tet_vertices = T[i]
        EVdW2_tet[i] = np.mean(EVdW2[tet_vertices], axis=0)
    
    # Use KMeans for hard clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(EVdW2_tet)
    
    # Convert labels to one-hot encoding matrix G
    G = np.zeros((n_clusters, n_tets))
    G[labels, np.arange(n_tets)] = 1

    return G

def create_exploded_group_matrix(G):
    r, t = G.shape  # r clusters, t tetrahedra
    
    row_idx = np.arange(9) + 9 * np.arange(r)[:, None]    # (r, 9)
    col_idx = np.arange(9) + 9 * np.arange(t)[:, None]    # (t, 9)
    
    rows = np.repeat(row_idx, t, axis=0)                  # (r*t, 9)
    cols = np.tile(col_idx, (r, 1))                       # (r*t, 9)
    
    values = np.repeat(G.flatten(), 9)
    rows = rows.reshape(-1)
    cols = cols.reshape(-1)
    
    mask = values != 0
    rows = rows[mask]
    cols = cols[mask]
    values = values[mask]
    
    G_exp = sp.coo_matrix((values, (rows, cols)), shape=(9*r, 9*t)).tocsr()
    return G_exp

def visualize_groups(V, T, G):
    n_clusters = G.shape[0]
    n_tets = G.shape[1]
    
    colors = plt.cm.tab20(np.linspace(0, 1, n_clusters))
    colors *= 255
    
    dominant_clusters = np.argmax(G, axis=0)
    tetmesh = TetMesh([V, T])
    tetmesh.cellcolors = colors[dominant_clusters]
    
    camera_settings = dict(
        pos=(10, 0, 0),           # Camera position
        focalPoint=(0, 0, 0),    # Look-at target
        viewup=(0, 0, 1)         # "Up" direction
    )
    
    plotter = Plotter(title=f"Tetrahedron Groups (n_clusters={n_clusters})")
    plotter = show(tetmesh, interactive=True, camera=camera_settings)
    # plotter = show(tetmesh, interactive=True)
    
    return plotter

def visualize_groups_pc(V, T, G):
    n_clusters = G.shape[0]
    n_tets = G.shape[1]
    
    # Calculate centroid of each tetrahedron
    tet_centers = np.mean(V[T], axis=1)
    
    # Create a plotter with subplots for each cluster
    rows = int(np.sqrt(n_clusters))
    cols = int(np.ceil(n_clusters / rows))
    plotter = Plotter(shape=(rows, cols), title=f"Cluster Weights (n_clusters={n_clusters})")

    camera_settings = dict(
        pos=(10, 0, 0),
        focalPoint=(0, 0, 0),
        viewup=(0, 0, 1)
    )

    # For each cluster, plot the weights
    for i in range(n_clusters):
        pc = Points(tet_centers, r=2)
        weights = G[i]
        pc.pointdata["weights"] = weights
        
        max_weight = weights.max()
        pc.cmap("YlOrRd", vmin=0, vmax=1).add_scalarbar(title=f"cluster {i}")
        plotter.show(pc, at=i, interactive=False, camera=camera_settings)
        # plotter.show(pc, at=i, interactive=False)
    
    plotter.interactive().close()
    return plotter

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='generate cluster groups')
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
    phi, leak = create_mask_matrix(V, T, C, BE, 'lin')
    Jw = W.T @ phi
    
    _, EMW = create_eigenmode_weights(K, M, Jw, n=50)
    B = lbs_matrix_column(V, EMW / _ ** 0.5)
    B = leak @ B
    
    G = create_group_matrix(_, EMW, T, vol, n_clusters=20)
    # visualize_groups(V, T, G)
    visualize_groups_pc(V, T, G)

