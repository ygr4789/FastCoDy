import numpy as np
from typing import Tuple, Optional
from scipy.sparse import csr_matrix


def compute_barycentric_coordinates_batch(V0: np.ndarray, tet_matrices_inv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute barycentric coordinates using precomputed inverse matrices.
    
    Args:
        v0: Point coordinates (3D)
        tet_matrices_inv: Precomputed inverse matrices for all tetrahedra
        
    Returns:
        Tuple of (barycentric_coordinates, valid_mask)
    """
    # Convert point to homogeneous coordinates
    v0_homog = np.append(V0, np.ones((V0.shape[0], 1)), axis=1) # Shape: (n_V0, 4)
    
    # Compute barycentric coordinates for all tetrahedra at once
    bary_coords = tet_matrices_inv @ v0_homog.T  # Shape: (n_tets, 4, n_V0)
    bary_coords = bary_coords.transpose(2, 0, 1)  # Shape: (n_V0, n_tets, 4)
    
    # Check which tetrahedra are valid (not degenerate)
    valid_mask = ~np.any(np.isnan(tet_matrices_inv), axis=(1, 2))
    
    return bary_coords, valid_mask


def find_best_tetrahedron(V0: np.ndarray, V: np.ndarray, T: np.ndarray, tet_matrices_inv: np.ndarray) -> Tuple[int, np.ndarray]:
    """
    Find the best tetrahedron for a given vertex v0 and return its barycentric coordinates.
    
    Args:
        v0: Surface vertex coordinates (3D)
        V: Tetrahedral mesh vertices (Nx3)
        T: Tetrahedral mesh connectivity (Mx4)
        tet_matrices_inv: Precomputed inverse matrices for all tetrahedra
        
    Returns:
        Tuple of (tetrahedron_index, barycentric_coordinates)
    """
    # Compute barycentric coordinates for all tetrahedra
    bary_coords, valid_mask = compute_barycentric_coordinates_batch(V0, tet_matrices_inv)
    
    min_coords = np.min(bary_coords, axis=2)
    valid_indices = np.where(valid_mask)[0]
    
    valid_min_coords = min_coords[:,valid_indices]
    best_tetrahedron_idx = valid_indices[np.argmax(valid_min_coords, axis=1)]
    best_bary_coords = bary_coords[np.arange(V0.shape[0]), best_tetrahedron_idx]
    
    return best_tetrahedron_idx, best_bary_coords


def precompute_tetrahedron_matrices(V: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Precompute inverse matrices for all tetrahedra for efficient barycentric coordinate computation.
    
    Args:
        V: Tetrahedral mesh vertices (Nx3)
        T: Tetrahedral mesh connectivity (Mx4)
        
    Returns:
        Inverse matrices for all tetrahedra (Mx4x4)
    """
    tet_matrices_inv = np.zeros((len(T), 4, 4))
    
    for i, tet in enumerate(T):
        tet_vertices = V[tet]
        # Create homogeneous coordinate matrix
        M = np.vstack([tet_vertices.T, np.ones(4)])
        try:
            tet_matrices_inv[i] = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            # If tetrahedron is degenerate, fill with NaN
            tet_matrices_inv[i] = np.nan
    
    return tet_matrices_inv


def surface_cast_barycentric(V0: np.ndarray, F0: np.ndarray, V: np.ndarray, T: np.ndarray) -> csr_matrix:
    """
    Find barycentric coordinates for each surface vertex with respect to the tetrahedral mesh.
    
    Args:
        V0: Surface mesh vertices (Nx3)
        F0: Surface mesh faces (Mx3) - not used in current implementation but kept for completeness
        V: Tetrahedral mesh vertices (Px3)
        T: Tetrahedral mesh connectivity (Qx4)
        
    Returns:
        Sparse matrix of shape (N, P) where:
        - N is the number of surface vertices
        - P is the number of tetrahedral vertices
        - Each row contains barycentric coordinates for the corresponding surface vertex
        - Only 4 non-zero entries per row (corresponding to the 4 vertices of the best tetrahedron)
    """
    # Precompute inverse matrices for all tetrahedra
    tet_matrices_inv = precompute_tetrahedron_matrices(V, T)
    
    n_vertices = V0.shape[0]
    n_tet_vertices = V.shape[0]
    
    # Prepare data for sparse matrix construction
    rows = []
    cols = []
    data = []
    
    tetra_idx, bary_coords = find_best_tetrahedron(V0, V, T, tet_matrices_inv)
        
    for i, (tetra_idx, bary_coords) in enumerate(zip(tetra_idx, bary_coords)):
        if tetra_idx >= 0 and bary_coords is not None:
            # Get the vertex indices of the best tetrahedron
            tet_vertices = T[tetra_idx]
            
            # Add entries for each vertex of the tetrahedron
            for j, vertex_idx in enumerate(tet_vertices):
                rows.append(i)
                cols.append(vertex_idx)
                data.append(bary_coords[j])
    
    # Create sparse matrix
    barycentric_matrix = csr_matrix((data, (rows, cols)), shape=(n_vertices, n_tet_vertices))
    
    return barycentric_matrix