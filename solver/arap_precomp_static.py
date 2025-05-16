import igl
import numpy as np
import scipy.sparse as sp

from src import lumped_mass_matrix

class arap_precomp_static:
  def __init__(self, X, T, J, B, G, Aeq, mu, invh2):
    M = lumped_mass_matrix(X, T)
    K, V, Mu = self.deformation_jacobian(X, T, mu)
    C = K.T @ V @ Mu @ K
    A = B.T @ M @ B * invh2 + B.T @ C @ B
    
    AeqB = Aeq @ B
    A = sp.vstack([
      sp.hstack([A, AeqB.T]),
      sp.hstack([AeqB, sp.csr_matrix((AeqB.shape[0], AeqB.shape[0]))])
    ]).tocsr()
    
    self.A = A
    self.GKJ = G @ K @ J
    self.GKB = G @ K @ B
    self.GVKB = G @ V @ Mu @ K @ B
    self.BtCJ = B.T @ C @ J
    
    self.BtMJ = B.T @ M @ J
    self.BtMB = B.T @ M @ B
    
  def deformation_jacobian(self, X, T, mu):
    t = T.shape[0]
    n = X.shape[0]
    
    V = igl.volume(X, T)
    
    K_data = []
    K_rows = []
    K_cols = []
    
    B_t = np.array([
      [-1, -1, -1],
      [ 1,  0,  0],
      [ 0,  1,  0],
      [ 0,  0,  1]
    ])

    for k in range(t):
      x_inds = T[k]  # 4 vertex indices
      X_t = X[x_inds].T  # shape: (3, 4) [A,B,C,D]
      Dm = X_t @ B_t  # 3x3 edge matrix [AB,AC,AD]
      K_t = B_t @ np.linalg.inv(Dm)  # (4, 3)
      K_t_T = K_t.T # (3, 4)
      # K_t_T @ [A;B;C;D] = F (3, 3)

      for r in range(3):  # For each row of K_t_T
        for i in range(4):  # Each vertex of the tet
          x_ind = x_inds[i]
          row_base =  9 * k + 3 * r
          K_data.extend([K_t_T[r, i]] * 3)
          K_rows.extend([row_base + 0, row_base + 1, row_base + 2])
          K_cols.extend([3*x_ind, 3*x_ind + 1, 3*x_ind + 2])

    K = sp.coo_matrix((K_data, (K_rows, K_cols)), shape=(9 * t, 3 * n)).tocsr()

    # Build mass matrix M (diagonal mass per tet)
    diag_vol = np.repeat(V, 9)
    diag_mu = np.repeat(mu, 9)
    row_idx = np.arange(9 * t)
    
    V = sp.coo_matrix((diag_vol, (row_idx, row_idx)), shape=(9 * t, 9 * t)).tocsr()
    Mu = sp.coo_matrix((diag_mu, (row_idx, row_idx)), shape=(9 * t, 9 * t)).tocsr()
    

    return K, V, Mu