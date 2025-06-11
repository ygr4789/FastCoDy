import igl
import numpy as np
import scipy.sparse as sp
def create_skinning_weights(V0, F0, weights, joints, V, T, F, C, BE):
    weights_full = np.zeros((V0.shape[0], C.shape[0]))
    for i in range(V0.shape[0]):
        for j in range(weights.shape[1]):
            if weights[i, j] > 0:
                weights_full[i, joints[i, j]] = weights[i, j]
                
    face_indices = np.unique(F.flatten())
    V_face = V[face_indices]
    distances = np.linalg.norm(V0[:, np.newaxis, :] - V_face[np.newaxis, :, :], axis=2)
    VtoV0_map = np.argmin(distances.T, axis=1)
    
    b = face_indices
    bc = weights_full[VtoV0_map]
    bbw_solver = igl.BBW()
    W = bbw_solver.solve(V, T, b, bc)
    W = W / W.sum(axis=1, keepdims=True)
    
    return W