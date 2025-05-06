import numpy as np

def weight_space_constraint(J, V):
    n = V.shape[0]
    p = J.shape[1]

    I = np.eye(n)
    zeros = np.zeros_like(I)

    # 3n x n projection matrices
    Px = np.vstack([I, zeros, zeros])
    Py = np.vstack([zeros, I, zeros])
    Pz = np.vstack([zeros, zeros, I])

    # Diagonal vertex position matrices
    X = np.diag(V[:, 0])
    Y = np.diag(V[:, 1])
    Z = np.diag(V[:, 2])

    # Compute Aij blocks
    A11 = Px @ X; A12 = Px @ Y; A13 = Px @ Z; A14 = Px
    A21 = Py @ X; A22 = Py @ Y; A23 = Py @ Z; A24 = Py
    A31 = Pz @ X; A32 = Pz @ Y; A33 = Pz @ Z; A34 = Pz

    # Compute JTij
    JT11 = J.T @ A11; JT12 = J.T @ A12; JT13 = J.T @ A13; JT14 = J.T @ A14
    JT21 = J.T @ A21; JT22 = J.T @ A22; JT23 = J.T @ A23; JT24 = J.T @ A24
    JT31 = J.T @ A31; JT32 = J.T @ A32; JT33 = J.T @ A33; JT34 = J.T @ A34

    # Stack vertically
    Jw = np.vstack([
        JT11, JT21, JT31,
        JT12, JT22, JT32,
        JT13, JT23, JT33,
        JT14, JT24, JT34
    ])

    return Jw  # shape: (12p, n)