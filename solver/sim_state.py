class sim_state:
  def __init__(self, z, p):
    self.z_prev = z
    self.z_curr = z
    self.p_prev = p
    self.p_curr = p
  
  def update(self, z, p):
    self.z_prev = self.z_curr
    self.z_curr = z
    self.p_prev = self.p_curr
    self.p_curr = p