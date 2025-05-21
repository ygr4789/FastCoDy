import torch
import torch.sparse as sp

from solver import arap_precomp_static

class arap_precomp_dynamic:
  def __init__(self, sp: arap_precomp_static) -> None:
    self.sp = sp
    self.device = sp.device
    
  def _to_tensor(self, x):
    if isinstance(x, torch.Tensor):
      return x.to(self.device)
    return torch.tensor(x, device=self.device)
    
  def precomp(self, p, st, f_ext):
    # Convert inputs to torch tensors if they aren't already
    p = self._to_tensor(p)
    f_ext = self._to_tensor(f_ext)
    
    self.f_ext = f_ext

    z_hist = 2.0 * st.z_curr - st.z_prev
    p_hist = 2.0 * st.p_curr - st.p_prev

    self.BtCJp = torch.mm(self.sp.BtCJ, p)
    self.GKJp = torch.mm(self.sp.GKJ, p)

    rig_momentum_terms = torch.mm(self.sp.BtMJ, (p - p_hist))
    self.inertia_grad = rig_momentum_terms - torch.mm(self.sp.BtMB, z_hist)
