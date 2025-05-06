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

    self.BtCJp = self.sp.BtCJ * p
    self.KJp = self.sp.KJ * p

    rig_momentum_terms = self.sp.BtMJ * (p - p_hist)
    self.inertia_grad = rig_momentum_terms - self.sp.BtMB * z_hist 
