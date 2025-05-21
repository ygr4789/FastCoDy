import torch

class sim_state:
  def __init__(self, z, p, device='cuda'):
    self.device = device
    self.z_prev = self._to_tensor(z).clone()
    self.z_curr = self._to_tensor(z).clone() 
    self.p_prev = self._to_tensor(p).clone()
    self.p_curr = self._to_tensor(p).clone()
  
  def _to_tensor(self, x):
    if isinstance(x, torch.Tensor):
      return x.to(self.device)
    return torch.tensor(x, device=self.device)
  
  def update(self, z, p):
    self.z_prev = self.z_curr.clone()
    self.z_curr = self._to_tensor(z).clone()
    self.p_prev = self.p_curr.clone() 
    self.p_curr = self._to_tensor(p).clone()