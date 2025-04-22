import sys
import numpy as np
import igl
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from src import *
from sim import *

from vedo import Mesh
from vedo.applications import AnimationPlayer

import time

# import sys
# sys.path.append("./pybind/build")
# import pyBartels

def create_animation(json_path):
    # Load mesh and animation data
    V, T, F, C, PI, BE, W, TF_list, dt, YM, pr, scale, physic_model = read_json_data(json_path)
    lambda_, mu = emu_to_lame(YM, pr)

    params = np.zeros((T.shape[0], 2))
    params[:, 0] = 0.5 * lambda_
    params[:, 1] = mu

    # === Boundary faces and LBS Matrix ===
    F = igl.boundary_facets(T)
    F = F[:, ::-1]  # reverse each face row to match
    VM = igl.lbs_matrix(V, W)

    # === Volume and differential operator ===
    vol = igl.volume(V, T)
    # dX = pyBartels.linear_tetmesh_dphi_dX(V, T)
    dX = linear_tetmesh_dphi_dX(V, T)

    # === Mass Matrix ===
    M = lumped_mass_matrix(V, T)
    M *= 1000
    # M *= 10000

    # === Linear Blend Skinning A matrix ===
    A = lbs_matrix_column(V, W)

    # === Poisson Constraint Mask ===
    phi = create_mask_matrix(V, T)

    # === Constraint system ===
    Aeq = A.T @ M @ phi
    Beq = np.zeros(Aeq.shape[0])

    # === Initial Transformation ===
    TF = TF_list[0]
    Vr = VM @ TF
    U = Vr - V

    nc = V.shape[0] * V.shape[1]

    UCol = vectorize(U)
    VCol = vectorize(V)

    UdCol = np.zeros(nc)
    UcCol = np.zeros(nc)

    max_iter = 20
    Vn_list = []

    for ai, TF in enumerate(TF_list):
        print(f"frame: {ai}")
        
        # Store previous state
        UCol0 = UCol.copy()
        UdCol0 = UdCol.copy()
        UcCol0 = UcCol.copy()

        # Compute reference state
        Vr = VM @ TF
        Ur = Vr - V
        UrCol = vectorize(Ur)        
        
        start = time.time()
        # Newton iterations
        for i in range(max_iter):
            # Current state
            # print(f"  iter {i}")
            q = VCol + UrCol + UcCol
            
            # Compute gradient and hessian
            # G = pyBartels.linear_tetmesh_arap_dq(V, T, q, dX, vol, params)
            G = linear_tetmesh_arap_dq(V, T, q, dX, vol, params)
            # flag = time.time()
            # print(f"    G-compute : {flag - start:.5f} sec")
            # start = flag
            
            # K = pyBartels.linear_tetmesh_arap_dq2(V, T, q, dX, vol, params)
            K = linear_tetmesh_arap_dq2(V, T, q, dX, vol, params)
            # flag = time.time()
            # print(f"    K-compute : {flag - start:.5f} sec")
            # start = flag
            
            # Build system matrix and RHS
            tmp_g = (M/(dt*dt)) @ (UrCol + UcCol) - \
                    M @ (UCol0/(dt*dt) + UdCol0/dt) + G
            tmp_H = M/(dt*dt) + K
            tmp_H = 0.5 * (tmp_H + tmp_H.T)  # Ensure symmetry
            
            # Build KKT system
            AA = sp.vstack([
                sp.hstack([tmp_H, Aeq.T]),
                sp.hstack([Aeq, sp.csr_matrix((Aeq.shape[0], Aeq.shape[0]))])
            ]).tocsr()
            
            b = np.concatenate([-tmp_g, Beq])
            
            # Solve system
            dUc = spla.spsolve(AA, b)[:tmp_g.shape[0]]
            
            # Check convergence
            if tmp_g @ dUc > -1e-6:
                break
            
            # flag = time.time()
            # print(f"    M-compute : {flag - start:.5f} sec")
            # start = flag
            
            # Line search
            def f(UrColi, UcColi):
                e = 0.5*(UrColi + UcColi - UCol0 - dt*UdCol0).T @ \
                    (M/(dt*dt)) @ (UrColi + UcColi - UCol0 - dt*UdCol0)
                # return e + pyBartels.linear_tetmesh_arap_q(V, T, VCol + UrColi + UcColi, 
                return e + linear_tetmesh_arap_q(V, T, VCol + UrColi + UcColi, 
                                               dX, vol, params)
            
            # Update position
            alpha = line_search(f, tmp_g, dUc, UrCol, UcCol)
            UcCol = UcCol + alpha * dUc
            
            # flag = time.time()
            # print(f"    L-compute : {flag - start:.5f} sec")
            # start = flag
        
        # Update velocities and positions for next frame
        UCol = UrCol + UcCol
        UdCol = (UCol - UCol0) / dt
        
        # Store result
        Vn = V + matrixize(UCol)
        Vn_list.append(Vn)

    # np.save('elephant',Vn_list)
    return Vn_list, F

def render_animation(Vn_list, F):
    mesh = Mesh([Vn_list[0], F])

    def update_scene(i: int):
        # update block and spring position at frame i
        mesh.points = Vn_list[i]
        plt.render()

    plt = AnimationPlayer(update_scene, irange=[0,len(Vn_list)], loop=True)
    plt += [mesh]
    plt.set_frame(0)
    plt.show()
    plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='generate complementary dynamics secondary motion')
    parser.add_argument('--input', '-i', type=str, required=False, help='json input path')
    parser.add_argument('--output', '-o', type=str, required=False, help='output path')
    parser.add_argument('--data', '-d', type=str, required=False, help='prebuilt data')
    
    args = parser.parse_args()
    input = args.input
    output = args.output
    data = args.data
    
    if input is None and data is None:
        print("input json file or prebuilt data required")
        exit()
    
    if data:
        Vn_list = np.load(data)
    elif input:
        Vn_list, F = create_animation(input)
        if output:
            np.save(output, Vn_list)
            
    render_animation(Vn_list, F)
        
    

