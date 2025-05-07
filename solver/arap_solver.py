
import numpy as np
import igl
import scipy.sparse.linalg as spla
from scipy.linalg import polar

from solver import arap_precomp_dynamic, arap_precomp_static

class arap_solver:
  def __init__(self, X, T, J, B, Aeq, mu, invh2, max_iter = 30, threshold = 1e-8):
    self.sp = arap_precomp_static(X, T, J, B, Aeq, mu, invh2)
    self.dp = arap_precomp_dynamic(self.sp)
    self.max_iter = max_iter
    self.threshold = threshold
    self.invh2 = invh2
    
    # self.dz = X.shape[0] * 3
    self.dz = B.shape[1]
    
  def step(self, z, p, st, bc, f_ext = 0):
    self.dp.precomp(p, st, bc, f_ext)
    
    z_prev = z
    z_next = z
    iter = 0
    
    while True:
      z_prev = z_next
      r = self.local_step(z_prev)
      z_next = self.global_step(z_prev, r, p)
      res = np.linalg.norm(z_next - z_prev)
      iter += 1
      if iter > self.max_iter: break
      if res < self.threshold: break
      
    print(f"  Iteration: {iter}")
    
    return z_next
    
  def local_step(self, z):
    f = self.sp.KB @ z + self.dp.KJp  # shape: (9t,)
    nt = f.shape[0] // 9 # number of tets

    F_stack = f.reshape(nt * 3, 3) # shape: (3 * t, 3)
    R_stack = np.zeros_like(F_stack)

    for i in range(nt):
        F_i = F_stack[3*i : 3*i+3, :] # shape: (3, 3)
        rot, _ = polar(F_i)
        R_stack[3*i : 3*i+3, :] = rot

    r = R_stack.reshape(-1) # flatten
    return r
    
  def global_step(self, z, r, p):
    arap_grad = self.dp.BtCJp - self.sp.VKB.T * r
    g = self.dp.inertia_grad * self.invh2 + arap_grad + self.dp.f_ext

    rhs = -np.concatenate([g, self.dp.bc], axis=0)
    z_next = spla.spsolve(self.sp.A, rhs)[:self.dz]
    return z_next 