import scipy.sparse
import igl
import numpy as np
from scipy.sparse import coo_matrix

def lbs_matrix_column(V, W):
    VW = igl.lbs_matrix(V, W)
    n, m = VW.shape
    VWC = np.zeros((3 * n, 3 * m), dtype=VW.dtype)
    for i in range(3):
        VWC[i::3, i*m:(i+1)*m] = VW
    return scipy.sparse.csr_matrix(VWC)