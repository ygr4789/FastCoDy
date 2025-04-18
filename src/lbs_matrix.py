import scipy.sparse
import igl
from scipy.sparse import coo_matrix

def lbs_matrix(V, W):
    Md = igl.lbs_matrix(V, W)
    return scipy.sparse.csr_matrix(Md)

def lbs_matrix_column(V, W):
    n, dim = V.shape
    m = W.shape[1]

    row_idx = []
    col_idx = []
    data = []

    for x in range(dim):
        for j in range(n):
            for i in range(m):
                for c in range(dim + 1):
                    val = W[j, i]
                    if c < dim:
                        val *= V[j, c]
                    row = dim * j + x
                    col = x * m + c * m * dim + i
                    row_idx.append(row)
                    col_idx.append(col)
                    data.append(val)

    M = coo_matrix((data, (row_idx, col_idx)), shape=(n * dim, m * dim * (dim + 1)))
    return M.tocsr()