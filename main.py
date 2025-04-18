import sys
import numpy as np
import igl
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from scipy.optimize import minimize, LinearConstraint

from src.util import *
from src.lbs_matrix import *
from src.read_data_from_json import *
from src.lumped_mass_matrix import *
from src.create_mask_matrix import *
from src.line_search import *

from sim.dphi import linear_tetmesh_dphi_dX
from sim.arap_dq import linear_tetmesh_arap_dq
from sim.arap_dq2 import linear_tetmesh_arap_dq2
from sim.arap_q import linear_tetmesh_arap_q

from vedo import Plotter, Mesh
from vedo.applications import AnimationPlayer

# ---- Assume custom functions are already defined elsewhere ----
# from my_module import read_json_data, emu_to_lame, lumped_mass_matrix
# from my_module import lbs_matrix_column, create_poisson_mask_matrix
# from my_module import vectorize, linear_tetmesh_dphi_dX

# === Initialization ==

def main():
    json_path = sys.argv[1] if len(sys.argv) > 1 else "examples/sphere/sphere.json"

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
    dX = linear_tetmesh_dphi_dX(V, T)

    # === Mass Matrix ===
    M = lumped_mass_matrix(V, T)
    M *= 200

    # === Linear Blend Skinning A matrix ===

    A = lbs_matrix_column(V, W)

    # === Poisson Constraint Mask ===

    phi = create_poisson_mask_matrix(V, T)

    # === Constraint system ===

    Aeq = A.T @ M @ phi
    # Aeq = np.zeros((24, Aeq.shape[1]))
    # Aeq[:, :24] = np.eye(24)
    
    
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
        
        # def objective_func(UcCol):
        #     e = 0.5*(UrCol + UcCol - UCol0 - dt*UdCol0).T @ \
        #             (M/(dt*dt)) @ (UrCol + UcCol - UCol0 - dt*UdCol0)
        #     return e + linear_tetmesh_arap_q(V, T, VCol + UrCol + UcCol, dX, vol, params)
        # def jac_func(UcCol):
        #     q = VCol + UrCol + UcCol
        #     jac = (M/(dt*dt)) @ (UrCol + UcCol - UCol0 - dt*UdCol0)
        #     return jac + linear_tetmesh_arap_dq(V, T, q, dX, vol, params)
        
        # comp_constraint = LinearConstraint(Aeq, 0.0, 0.0)

        # res = minimize(
        #     objective_func,
        #     UcCol,
        #     jac=jac_func,
        #     constraints=[comp_constraint]
        # )
        # UcCol = res.x
        # UCol = UrCol + UcCol
        # UdCol = (UCol - UCol0) / dt
        
        # # Store result
        # Vn = V + matrixize(UCol)
        # Vn_list.append(Vn)
        
        # continue
        
        # Newton iterations
        for i in range(max_iter):
            # Current state
            q = VCol + UrCol + UcCol
            
            # Compute gradient and hessian
            G = linear_tetmesh_arap_dq(V, T, q, dX, vol, params)
            K = linear_tetmesh_arap_dq2(V, T, q, dX, vol, params)
            
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
            
            # Line search
            def f(UrColi, UcColi):
                e = 0.5*(UrColi + UcColi - UCol0 - dt*UdCol0).T @ \
                    (M/(dt*dt)) @ (UrColi + UcColi - UCol0 - dt*UdCol0)
                return e + linear_tetmesh_arap_q(V, T, VCol + UrColi + UcColi, 
                                               dX, vol, params)
            
            # Update position
            alpha = line_search(f, tmp_g, dUc, UrCol, UcCol)
            UcCol = UcCol + alpha * dUc
        
        # Update velocities and positions for next frame
        UCol = UrCol + UcCol
        UdCol = (UCol - UCol0) / dt
        
        # Store result
        Vn = V + matrixize(UCol)
        Vn_list.append(Vn)

    # np.save('elephant',Vn_list)
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
    main()

