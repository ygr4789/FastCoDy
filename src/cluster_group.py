import numpy as np
import igl

from vedo import Plotter, TetMesh, show
# from vedo.applications import AnimationPlayer

from sklearn.cluster import KMeans
import scipy.sparse as sp
import matplotlib.pyplot as plt


if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
    
from src import *
from sim import *

def create_group_matrix(eigvals, eigvecs, T, Vol, n_clusters=100):
    EW = eigvals
    EV = eigvecs
    EW2 = EW ** 2
    EVdW2 = EV / EW2
    
    n_tets = T.shape[0]
    n_modes = EV.shape[1]
    EVdW2_tet = np.zeros((n_tets, n_modes))
    
    for i in range(n_tets):
        tet_vertices = T[i]
        EVdW2_tet[i] = np.mean(EVdW2[tet_vertices], axis=0)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(EVdW2_tet)
    
    G = np.zeros((n_clusters, n_tets))
    
    for i in range(n_clusters):
        cluster_tets = np.where(cluster_labels == i)[0]
        cluster_vol_sum = np.sum(Vol[cluster_tets])
        G[i, cluster_tets] = Vol[cluster_tets] / cluster_vol_sum
    
    G = G.T
    G = G / np.sum(G, axis=1)[:, None]
    G = G.T
    return G

    # distances = np.zeros((n_tets, n_clusters))
    #     for i in range(n_clusters):
    #         diff = EVdW2_tet - kmeans.cluster_centers_[i]
    #         distances[:, i] = np.sum(diff * diff, axis=1)
        
    # distances = -distances
    # exp_distances = np.exp(distances)
    # G = exp_distances / np.sum(exp_distances, axis=1)[:, None]
    
    # G = G * Vol[:, None]
    # G = G / np.sum(G, axis=1)[:, None]
    # G = G.T
    # return G

def create_exploded_group_matrix(G):
    r, t = G.shape  # r clusters, t tetrahedra
    
    row_idx = np.arange(9)[:,None] + 9*np.arange(r)[:,None,None]  # Shape: (r, 1, 9)
    col_idx = np.arange(9)[:,None] + 9*np.arange(t)[:,None,None]  # Shape: (t, 1, 9)
    
    G_values = np.repeat(G.flatten(), 9)
    
    rows = row_idx.repeat(t, axis=0).reshape(-1)
    cols = col_idx.repeat(r, axis=0).reshape(-1)
    
    mask = G_values != 0
    rows = rows[mask]
    cols = cols[mask]
    values = G_values[mask]
    
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
    phi = create_mask_matrix(V, T, C, BE, 'lin')
    
    Jleak = M @ phi @ J
    Jw = W.T @ lumped_mass_3n_to_n(phi)
    
    _, EMs = create_eigenmode_weights(K, M, Jw, n=20)
    
    # Create and visualize groups
    G = create_group_matrix(_, EMs, T, vol, n_clusters=500)
    # visualize_eigenmodes(V, EMs.T)
    visualize_groups(V, T, G)

