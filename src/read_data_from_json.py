import os
import numpy as np
import json
import igl
from scipy.sparse import coo_matrix
from scipy.spatial.transform import Rotation as R

def read_bone_anim(anim_file, C, BE, P, center, scale):
    dim = 3
    T_list = []

    with open(anim_file, 'r') as file:
        num_bone, num_frame = map(int, file.readline().split())

        rot_list = [None] * num_bone
        tran_list = [None] * num_bone
        rest_list = []

        # === Read root rest transform ===
        root_rest_tran = np.array(file.readline().strip().split(), dtype=float)
        root_rest_tran = (root_rest_tran - center) / scale

        root_euler = np.array(file.readline().strip().split(), dtype=float)
        root_affine_rot = euler_to_quat_from_affine(root_euler).as_matrix()

        # === Read rest pose bone rotations ===
        for _ in range(num_bone):
            euler = np.array(file.readline().strip().split(), dtype=float)
            q = euler_to_quat_from_affine(euler)
            rest_list.append(q)

        for _ in range(num_frame):
            root_rot = np.array(file.readline().strip().split(), dtype=float)
            root_tran = np.array(file.readline().strip().split(), dtype=float)
            root_tran = (root_tran - center) / scale
            root_q = euler_to_quat_from_affine(root_rot, root_affine_rot)

            for i in range(num_bone):
                euler = np.array(file.readline().strip().split(), dtype=float)
                q = euler_to_quat_from_affine(euler, rest_list[i].as_matrix())

                if P[i] == -1:
                    root_count = np.sum(P == -1)
                    if root_count > 1:
                        q = root_q * q  # rotation order reversed
                    tran = root_tran - root_rest_tran
                else:
                    tran = np.zeros(3)

                rot_list[i] = q
                tran_list[i] = tran

            # Convert Rotation objects to quaternions (w,x,y,z format)
            vQ = np.array([q.as_quat() for q in rot_list], dtype=np.float64)  # This will be (n_bones, 4)
            vT = np.array(tran_list, dtype=np.float64)  # This will be (n_bones, 3)
            
            # Forward kinematics with quaternions
            vQ_out, vT_out = igl.forward_kinematics(C, BE, P, vQ, vT)

            # Build transformation matrix for each bone
            T = np.zeros((num_bone * 4, dim))
            for i in range(num_bone):
                a = np.eye(4)
                # Convert quaternion output back to rotation matrix
                a[:dim, :dim] = R.from_quat(vQ_out[i]).as_matrix()
                a[:dim, dim] = vT_out[i]
                T[i * (dim+1):(i + 1) * (dim+1), :] = a[:dim, :dim+1].T
                # T[i * 4:(i + 1) * 4, :] = a[:4, :dim]

            T_list.append(T)

    return T_list

def read_pnt_anim(anim_file, C, center, scale):
    dim = 3
    T_list = []

    with open(anim_file, 'r') as file:
        num_bone, num_frame = map(int, file.readline().split())

        # === Read rest state ===
        rest_tran_list = []
        rest_affine_list = []

        for _ in range(num_bone):
            rest_tran = np.array(file.readline().split(), dtype=float)
            rest_tran = (rest_tran - center) / scale
            rest_tran_list.append(rest_tran)

            euler = np.array(file.readline().split(), dtype=float)
            q = euler_to_quat_from_affine(euler)
            a = np.eye(4)
            a[:3, :3] = q.as_matrix()
            rest_affine_list.append(a)

        for _ in range(num_frame):
            tran_list = []
            rot_list = []

            for i in range(num_bone):
                tran = np.array(file.readline().split(), dtype=float)
                tran = (tran - center) / scale

                euler = np.array(file.readline().split(), dtype=float)
                q = euler_to_quat_from_affine(euler, rest_affine_list[i])

                # Apply relative to center offset
                q_matrix = q.as_matrix()
                cn = C[i]  # row i of C (shape: (3,))
                if cn.ndim == 1:
                    cn = cn.reshape(3,)

                tran_corrected = tran - rest_tran_list[i] + cn - q.apply(cn)
                tran_list.append(tran_corrected)
                rot_list.append(q)

            # === Compose full transformation matrix ===
            T = np.zeros((num_bone * (dim + 1), dim))
            
            for i in range(num_bone):
                a = np.eye(4)
                a[:3, :3] = rot_list[i].as_matrix()
                a[:3, 3] = tran_list[i]
                T[i * (dim + 1):(i + 1) * (dim + 1), :] = a[:dim, :dim+1].T

            T_list.append(T)

    return T_list

def convert_vector_mask_to_sparsematrix(dim, V):
    n = len(V)
    row = []
    col = []
    data = []
    for k in range(n):
        for d in range(dim):
            idx = dim * k + d
            row.append(idx)
            col.append(idx)
            data.append(V[k])
    return coo_matrix((data, (row, col)), shape=(n * dim, n * dim))

def normalize_obj(V, center, scale):
    V = V - center
    V = V / scale
    return V

def read_mesh_files_from_directory(directory):
    mesh_list = []
    for filename in os.listdir(directory):
        if filename.endswith('.mesh'):
            name = os.path.splitext(filename)[0]
            mesh_list.append(name)
    return mesh_list

def euler_to_quat(euler, frame_axes=None):
    if frame_axes is not None:
        # frame_axes: (3, 3) matrix with column axes
        r = R.from_euler('xyz', euler)
        quat_matrix = r.as_matrix()
        rotated = frame_axes @ quat_matrix
        return R.from_matrix(rotated)
    else:
        return R.from_euler('xyz', euler)

def euler_to_quat_from_affine(euler_deg, frame_affine=None):
    euler_rad = np.deg2rad(euler_deg)
    if frame_affine is not None:
        # rotate with frame basis
        rot = R.from_euler('xyz', euler_rad)
        rot_mat = frame_affine[:3, :3] @ rot.as_matrix() @ frame_affine[:3, :3].T
        # rot_mat = rot.as_matrix()
        return R.from_matrix(rot_mat)
    else:
        return R.from_euler('xyz', euler_rad)

def read_blendshape_anim(file_path):
    w_list = []
    with open(file_path, 'r') as f:
        num_bs, num_frames = map(int, f.readline().split())
        for _ in range(num_frames):
            weights = np.array(list(map(float, f.readline().split())))
            w_list.append(weights)
    return w_list

def read_json_data(filename):
    with open(filename, 'r') as f:
        j = json.load(f)

    dir_path = os.path.dirname(filename) + os.sep

    # Required fields
    physic_model = j.get("model", None)
    if physic_model is None:
        raise ValueError("No model specified in JSON.")

    # Initialize containers
    V = T = F = C = PI = BE = W = None
    TF_list = []
    dt = YM = pr = scale = 0.0

    if "mesh_file" in j:
        mesh_filename = os.path.join(dir_path, j["mesh_file"])
        V, T, F = igl.read_mesh(mesh_filename)  # assumes pyigl overloads .read_mesh

    if "handle_file" in j:
        handle_filename = os.path.join(dir_path, j["handle_file"])
        C, E, PI, BE, CE, PE = igl.read_tgf(handle_filename)

    if "weight_file" in j:
        weight_filename = os.path.join(dir_path, j["weight_file"])
        W = igl.read_dmat(weight_filename)

    # Read scalar params
    dt = j.get("dt", 0.0)
    YM = j.get("YM", 0.0)
    pr = j.get("pr", 0.0)
    scale = j.get("scale", 1.0)

    # Normalize V
    V_reshaped = V.reshape(-1, 3)  # shape (N, 3)
    Vmin = V_reshaped.min(axis=0)      # min x, y, z
    Vmax = V_reshaped.max(axis=0)      # max x, y, z
    Vcenter = (Vmax + Vmin) / 2.0      # center point
    V_shifted = V_reshaped - Vcenter   # centered vertices
    Vbound = np.linalg.norm(V_shifted, axis=1).max()
    V_normalized = V_shifted / Vbound
    V = V_normalized

    # Normalize handles
    C = C - Vcenter
    C = C / Vbound

    # Animation
    if "anim_file" in j:
        anim_filename = os.path.join(dir_path, j["anim_file"])
        if BE.shape[0] > 0:
            P = igl.directed_edge_parents(BE)
            TF_list = read_bone_anim(anim_filename, C, BE, P, Vcenter.T, Vbound)
        else:
            TF_list = read_pnt_anim(anim_filename, C, Vcenter.T, Vbound)

    return V, T, F, C, PI, BE, W, TF_list, dt, YM, pr, scale, physic_model