import igl
import numpy as np
import scipy.sparse as sp

from src import lumped_mass_matrix

class arap_precomp_static:
  def __init__(self, X, T, J, Aeq, mu, invh2):
    M = lumped_mass_matrix(X, T)
    K, V, Mu= self.deformation_jacobian(X, T, mu)
    C = K.T @ V @ Mu @ K
    
    x = X.flatten()
    A = invh2 * M + C
    # A = invh2 * M
    
    A = sp.vstack([
      sp.hstack([A, Aeq.T]),
      sp.hstack([Aeq, sp.csr_matrix((Aeq.shape[0], Aeq.shape[0]))])
    ]).tocsr()
    
    self.J = J
    
    self.x = x
    self.M = M
    self.A = A
    self.K = K
    self.VK = V @ Mu @ K
    
    self.Kx = K @ x
    self.Cx = C @ x
    self.Mx = M @ x
    self.VKx = V @ Mu @ K @ x
    
    self.CJ = C @ J
    self.KJ = K @ J
    self.MJ = M @ J
    self.VKJ = V @ Mu @ K @ J
    
  def deformation_jacobian(self, X, T, mu):
    t = T.shape[0]
    n = X.shape[0]
    
    print(t)
    V = igl.volume(X, T)
    print(V.shape)
    
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