import numpy as np

def line_search(f, tmp_g, dUc, Ur, Uc):
    """
    Backtracking line search with Armijo condition.
    
    Args:
        f: callable, takes (Ur, Uc) and returns a scalar energy
        tmp_g: (n,) gradient vector
        dUc: (n,) descent direction
        Ur: fixed variable (used in energy evaluation)
        Uc: current variable (being updated)

    Returns:
        alpha: optimal step size (scalar)
    """
    alpha = 1.0
    p = 0.5
    c = 1e-8
    

    f0 = f(Ur, Uc)
    s = f0 + c * tmp_g.dot(dUc)

    i = 0

    while alpha > c:
        i += 1
        Uc_tmp = Uc + alpha * dUc
        if f(Ur, Uc_tmp) <= s:
            break
        alpha *= p

    return alpha