def emu_to_lame(E, p):
    lambda_ = (E * p) / ((1 + p) * (1 - 2 * p))
    mu = E / (2 * (1 + p))
    return lambda_, mu

def vectorize(M):
    return M.reshape(-1)

def matrixize(V):
    return V.reshape((3, -1), order='F').T  # Transpose to get (N, 3)