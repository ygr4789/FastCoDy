import numpy as np
import scipy.sparse as sp

def weight_space_constraint(J, V):
    n = V.shape[0]
    p = J.shape[1]

    # Convert J to sparse if not already
    if not sp.issparse(J):
        J = sp.csr_matrix(J)

    # Identity matrix
    I = sp.eye(n, format='csr')
    Z = sp.csr_matrix((n, n))

    # 3n x n projection matrices using block structure
    Px = sp.vstack([I, Z, Z], format='csr')  # shape: (3n, n)
    Py = sp.vstack([Z, I, Z], format='csr')
    Pz = sp.vstack([Z, Z, I], format='csr')

    # Diagonal sparse matrices from vertex positions
    X = sp.diags(V[:, 0])
    Y = sp.diags(V[:, 1])
    Z_ = sp.diags(V[:, 2])

    # Aij blocks: shape (3n, n)
    A11 = Px @ X; A12 = Px @ Y; A13 = Px @ Z_; A14 = Px
    A21 = Py @ X; A22 = Py @ Y; A23 = Py @ Z_; A24 = Py
    A31 = Pz @ X; A32 = Pz @ Y; A33 = Pz @ Z_; A34 = Pz

    # JTij: shape (p, n)
    def JT(Aij): return J.T @ Aij  # Use sparse matmul

    JT_blocks = [
        JT(A11), JT(A21), JT(A31),
        JT(A12), JT(A22), JT(A32),
        JT(A13), JT(A23), JT(A33),
        JT(A14), JT(A24), JT(A34),
    ]

    Jw = sp.vstack(JT_blocks, format='csr')
    return Jw  # shape: (12p, n)