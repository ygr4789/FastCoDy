import igl
import numpy as np
import scipy.sparse as sp

from solver import arap_precomp_static

class arap_precomp_dynamic:
  def __init__(self, sp: arap_precomp_static) -> None:
    self.sp = sp
    
  def precomp(self, p, st, bc, f_ext):
    self.f_ext = f_ext
    self.bc = bc

    z_hist = 2.0 * st.z_curr - st.z_prev
    p_hist = 2.0 * st.p_curr - st.p_prev

    self.Cur = self.sp.CJ * p - self.sp.Cx
    self.Mur = self.sp.MJ * p - self.sp.Mx
    self.VKur = self.sp.VKJ * p - self.sp.VKx
    self.Kur = self.sp.KJ * p - self.sp.Kx

    Mp_hist = (self.sp.MJ * p_hist)
    rig_momentum_terms = Mp_hist - self.sp.MJ * p
    
    self.My = self.sp.M * z_hist + rig_momentum_terms
