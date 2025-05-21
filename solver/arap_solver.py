import torch
import torch.sparse as sp
import torch.linalg as linalg

from solver import arap_precomp_dynamic, arap_precomp_static

class arap_solver:
  def __init__(self, X, T, J, B, G, mu, invh2, max_iter = 30, threshold = 1e-8, device='cuda'):
    self.device = device
    self.sp = arap_precomp_static(X, T, J, B, G, mu, invh2, device=device)
    self.dp = arap_precomp_dynamic(self.sp)
    self.max_iter = max_iter
    self.threshold = torch.tensor(threshold, device=device)
    self.invh2 = self._to_tensor(invh2)
    
    self.dz = B.shape[1]

  def _to_tensor(self, x):
    if isinstance(x, torch.Tensor):
      return x.to(self.device)
    return torch.tensor(x, device=self.device)
    
  def step(self, z, p, st, f_ext = 0):
    # Convert inputs to torch tensors if they aren't already
    z = self._to_tensor(z)
    p = self._to_tensor(p)
    f_ext = self._to_tensor(f_ext)
    
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    
    start_time.record()
    self.dp.precomp(p, st, f_ext)
    end_time.record()
    torch.cuda.synchronize()
    precomp_time = start_time.elapsed_time(end_time)
    
    z_prev = z
    z_next = z
    iter = 0
    
    local_time = 0
    global_time = 0
    
    while True:
      z_prev = z_next
      
      start_time.record()
      r = self.local_step(z_prev)
      end_time.record()
      torch.cuda.synchronize()
      local_time += start_time.elapsed_time(end_time)
      
      start_time.record() 
      z_next = self.global_step(z_prev, r, p)
      end_time.record()
      torch.cuda.synchronize()
      global_time += start_time.elapsed_time(end_time)
      
      res = torch.norm(z_next - z_prev)
      iter += 1
      if iter >= self.max_iter: break
      if res < self.threshold: break
      
    print(f"  Iteration      : {iter}")
    print(f"  Precomp Time   : {precomp_time:.2f} ms")
    print(f"  Local Time     : {local_time:.2f} ms")
    print(f"  Global Time    : {global_time:.2f} ms")
    print(f"  Total Time     : {precomp_time + local_time + global_time:.2f} ms")
    
    return z_next
  
  def local_step(self, z):
    f = torch.mm(self.sp.GKB, z) + self.dp.GKJp  # shape: (9t,)
    nt = f.shape[0] // 9  # number of tets

    F_stack = f.reshape(nt, 3, 3)  # shape: (t, 3, 3)
    
    # Compute SVD for all matrices at once
    U, S, Vh = torch.linalg.svd(F_stack)
    
    # Compute rotation matrices in parallel
    R_stack = torch.bmm(U, Vh)
    
    r = R_stack.reshape(-1, 1)  # flatten to column vector
    return r
  
  def global_step(self, z, r, p):
    arap_grad = self.dp.BtCJp - torch.mm(self.sp.GVKB.t(), r)
    g = self.dp.inertia_grad * self.invh2 + arap_grad + self.dp.f_ext

    # Use precomputed inverse instead of solving the system
    z_next = torch.mm(self.sp.A_inv, -g)[:self.dz]
    return z_next 