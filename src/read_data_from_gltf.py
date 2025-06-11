import os
import numpy as np
from pygltflib import GLTF2
from scipy.spatial.transform import Rotation as R
import igl
import wildmeshing
import pickle

if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from src.create_vol_weights import create_skinning_weights
from src.surface_cast import surface_cast_barycentric

def read_accessor(gltf, accessor_idx, base_dir):
    """Read data from a GLTF accessor."""
    accessor = gltf.accessors[accessor_idx]
    view = gltf.bufferViews[accessor.bufferView]
    buffer = gltf.buffers[view.buffer]
    buffer_path = os.path.join(base_dir, buffer.uri)
    with open(buffer_path, "rb") as f:
        raw = f.read()

    offset = (view.byteOffset or 0) + (accessor.byteOffset or 0)
    length = accessor.count
    type_count = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}
    count = type_count[accessor.type]

    dtype_map = {
        5120: np.int8, 5121: np.uint8,
        5122: np.int16, 5123: np.uint16,
        5125: np.uint32, 5126: np.float32
    }

    dtype = dtype_map[accessor.componentType]
    arr = np.frombuffer(raw, dtype=dtype, count=length * count, offset=offset)
    return arr.reshape(length, count)

def make_transform_matrix(translation, rotation, scale):
    """Create a 4x4 transform matrix from translation, rotation, and scale."""
    t = np.eye(4)
    t[:3, 3] = np.array(translation) if translation is not None else np.zeros(3)
    r = np.eye(4)
    r[:3, :3] = R.from_quat(np.array(rotation) if rotation is not None else [0, 0, 0, 1]).as_matrix()
    s = np.eye(4)
    s[:3, :3] = np.diag(np.array(scale) if scale is not None else np.ones(3))
    return t @ r @ s

def build_parent_map(nodes):
    """Build a mapping of child node indices to their parent indices."""
    parent_map = {}
    for idx, node in enumerate(nodes):
        if node.children:
            for child in node.children:
                parent_map[child] = idx
    return parent_map

def load_gltf(gltf_path):
    """Load and prepare GLTF data."""
    base_dir = os.path.dirname(gltf_path)
    gltf = GLTF2().load(gltf_path)
    return gltf, base_dir

def get_frame_transform(node_idx, nodes, parent_map, frame_idx, times, joint_transforms, cache=None, center=None, scale=None):
    """Get transform matrix for a specific frame."""
    if cache is None:
        cache = {}
    if (node_idx, frame_idx) in cache:
        return cache[(node_idx, frame_idx)]
    
    node = nodes[node_idx]
    transforms = joint_transforms.get(node_idx, {}) if joint_transforms else {}
    
    if frame_idx == -1:  # Rest pose
        t = node.translation if node.translation else [0, 0, 0]
        r = node.rotation if node.rotation else [0, 0, 0, 1]
        s = node.scale if node.scale else [1, 1, 1]
    else:  # Animated pose
        default_t = node.translation if node.translation else [0, 0, 0]
        default_r = node.rotation if node.rotation else [0, 0, 0, 1]
        default_s = node.scale if node.scale else [1, 1, 1]
        
        t_data = transforms.get('translation', [default_t])
        r_data = transforms.get('rotation', [default_r])
        s_data = transforms.get('scale', [default_s])
        
        frame_idx = min(frame_idx, len(times) - 1) if times is not None else 0
        
        t = t_data[frame_idx] if frame_idx < len(t_data) else default_t
        r = r_data[frame_idx] if frame_idx < len(r_data) else default_r
        s = s_data[frame_idx] if frame_idx < len(s_data) else default_s
    
    local_matrix = make_transform_matrix(t, r, s)
    
    parent_idx = parent_map.get(node_idx)
    if parent_idx is not None:
        parent_matrix = get_frame_transform(parent_idx, nodes, parent_map, frame_idx, times, joint_transforms, cache)
        global_matrix = parent_matrix @ local_matrix
    else:
        global_matrix = local_matrix
    
    cache[(node_idx, frame_idx)] = global_matrix
    return global_matrix

def load_animation_data(gltf, base_dir):
    """Load animation data from GLTF."""
    if not gltf.animations:
        return None, None
    
    animation = gltf.animations[0]
    times = None
    joint_transforms = {}
    
    for channel in animation.channels:
        sampler = animation.samplers[channel.sampler]
        node_idx = channel.target.node
        
        if times is None:
            times = read_accessor(gltf, sampler.input, base_dir)
        
        transform_data = read_accessor(gltf, sampler.output, base_dir)
        
        if node_idx not in joint_transforms:
            joint_transforms[node_idx] = {
                'translation': [],
                'rotation': [],
                'scale': []
            }
        
        if channel.target.path == 'translation':
            joint_transforms[node_idx]['translation'] = transform_data
        elif channel.target.path == 'rotation':
            joint_transforms[node_idx]['rotation'] = transform_data
        elif channel.target.path == 'scale':
            joint_transforms[node_idx]['scale'] = transform_data
    
    return times, joint_transforms

def read_gltf_data(gltf_path):
    """Read vertex positions, face indices, joint transformations, and control point data from GLTF file.
    
    Args:
        gltf_path: Path to the GLTF file
        
    Returns:
        V: numpy array of shape (n, 3) containing vertex positions
        T: numpy array of shape (t, 4) containing tetrahedron indices
        F: numpy array of shape (f, 3) containing surface face indices
        TF_list: list of length k (number of frames) where each element is a (4*j)*3 matrix
                 containing joint transformation matrices (j is number of joints)
        C: numpy array of shape (n, 3) containing control point positions in rest pose
        BE: numpy array of shape (k, 2) containing bone edge connections (parent-child pairs)
        W: numpy array containing weights
    """
    cache_path = gltf_path.replace(".gltf", "_cache.pkl")
    if os.path.exists(cache_path):
        print("Found cache file: ", cache_path)
        with open(cache_path, "rb") as f:
            return pickle.load(f)
        
    print("No cache file found, generating data...")
    
    gltf, base_dir = load_gltf(gltf_path)
    
    # Get mesh data
    primitive = gltf.meshes[0].primitives[0]
    V0 = read_accessor(gltf, primitive.attributes.POSITION, base_dir).astype(np.float64)
    F0 = read_accessor(gltf, primitive.indices, base_dir).reshape(-1, 3).astype(np.int32)
    
    if not igl.is_edge_manifold(F0):
        raise ValueError("The mesh is not edge-manifold.")
    
    # Get skinning data
    skin = gltf.skins[0]
    nodes = gltf.nodes
    
    parent_map = build_parent_map(nodes)
    joints = skin.joints
    
    for joint_id in joints:
        joint_node = nodes[joint_id]
        if joint_node.children:
            for child_id in joint_node.children:
                if child_id not in joints:
                    joints.append(child_id)
        
    num_joints = len(joints)
    cache = {}
    
    # Find the mesh node and apply its rest pose transform
    mesh_node_idx = None
    for idx, node in enumerate(nodes):
        if node.mesh is not None and node.mesh == 0:  # Find the node that references our mesh
            mesh_node_idx = idx
            break
    
    if mesh_node_idx is not None:
        # Get the mesh node's transform matrix
        mesh_transform = get_frame_transform(mesh_node_idx, nodes, parent_map, -1, None, None, cache)
        # Apply mesh transform to all vertices
        vertices_homogeneous = np.hstack([V0, np.ones((len(V0), 1))])
        V0 = (mesh_transform @ vertices_homogeneous.T).T[:, :3]
    
    # Vertex normalization
    Vmin = V0.min(axis=0)      # min x, y, z
    Vmax = V0.max(axis=0)      # max x, y, z
    Vcenter = (Vmax + Vmin) / 2.0      # center point
    V_shifted = V0 - Vcenter   # centered vertices
    Vbound = np.linalg.norm(V_shifted, axis=1).max()
    V0 = V_shifted / Vbound
    
    norm_transform = np.eye(4)
    norm_transform[:3, 3] = - Vcenter / Vbound
    norm_transform[:3, :3] = np.diag([1/Vbound]*3)
    
    # Get control point positions (C) in rest pose
    C = np.zeros((num_joints, 3))
    for i, joint_idx in enumerate(joints):
        transform = get_frame_transform(joint_idx, nodes, parent_map, -1, None, None, cache)
        C[i] = transform[:3, 3]
    
    C = (norm_transform @ np.hstack([C, np.ones((len(C), 1))]).T).T[:, :3]
    
    # Create bone edge connections (BE)
    BE = []
    for child_idx, parent_idx in parent_map.items():
        if child_idx in joints and parent_idx in joints:
            child_pos = joints.index(child_idx)
            parent_pos = joints.index(parent_idx)
            BE.append([parent_pos, child_pos])
    BE = np.array(BE) if BE else np.zeros((0, 2), dtype=int)
    
    # Load animation data
    times, joint_transforms = load_animation_data(gltf, base_dir)
    TF_list = []
    
    if times is not None:
        num_frames = len(times)
        
        # Pre-compute all rest pose transforms once
        rest_pose_transforms = np.array([
            norm_transform @ get_frame_transform(joint_idx, nodes, parent_map, -1, None, None, cache)
            for joint_idx in skin.joints
        ])
        
        rest_pose_inv = np.linalg.inv(rest_pose_transforms)
        
        # Process each frame
        for frame_idx in range(num_frames):
            # Get current frame transforms
            current_frame_transforms = np.array([
                norm_transform @ get_frame_transform(joint_idx, nodes, parent_map, frame_idx, times, joint_transforms, cache)
                for joint_idx in skin.joints
            ])
            
            # Compute relative transforms
            relative_transforms = np.matmul(current_frame_transforms, rest_pose_inv)
            TF = np.zeros((4 * num_joints, 3))
            for i in range(num_joints):
                TF[i*4:(i+1)*4, :] = relative_transforms[i, :3, :4].T
            
            TF_list.append(TF)
    else:
        # Single frame with identity transformations
        TF = np.zeros((4 * num_joints, 3))
        for i in range(num_joints):
            TF[i*4:(i+1)*4, :] = np.eye(4)[:4, :3].T
        TF_list.append(TF)
    
    tetra = wildmeshing.Tetrahedralizer(stop_quality=1000, epsilon = 0.001, edge_length_r = 0.02)
    tetra.set_mesh(V0, F0)
    tetra.tetrahedralize()
    V, T = tetra.get_tet_mesh()
    visualize_mesh(V, T)
    
    V = V.astype(np.float64)
    T = T.astype(np.int32)
    F = igl.boundary_facets(T)
    F = F[:, ::-1]
    
    # Create weights
    weights = read_accessor(gltf, primitive.attributes.WEIGHTS_0, base_dir) if primitive.attributes.WEIGHTS_0 is not None else None
    joints = read_accessor(gltf, primitive.attributes.JOINTS_0, base_dir) if primitive.attributes.JOINTS_0 is not None else None
    
    W = create_skinning_weights(V0, F0, weights, joints, V, T, F, C, BE)
    dt = 0.03
    
    with open(cache_path, "wb") as f:
        pickle.dump((V0, F0, V, T, F, TF_list, C, BE, W, dt), f)
    
    return V0, F0, V, T, F, TF_list, C, BE, W, dt

def visualize_mesh(V, T):
    from vedo import Mesh, Plotter, Points
    F = igl.boundary_facets(T)
    mesh = Mesh([V, F], alpha=0.2)
    surface_idx = np.unique(F.flatten())
    inner_idx = np.setdiff1d(np.arange(V.shape[0]), surface_idx)
    pc = Points(V[inner_idx], c='red', r=3)
    
    plotter = Plotter()
    plotter.add(mesh)
    plotter.add(pc)
    plotter.show()

def get_texture_info(gltf_path):
    """Get the diffuse texture path and UV coordinates from the GLTF file.
    
    Args:
        gltf_path: Path to the GLTF file
        
    Returns:
        texture_path: Path to the diffuse texture file (or None if not found)
        uvs: UV coordinates for texture mapping (or None if not found)
    """
    base_dir = os.path.dirname(gltf_path)
    gltf = GLTF2().load(gltf_path)
    
    # Get the first mesh and primitive
    if not gltf.meshes:
        return None, None
    
    primitive = gltf.meshes[0].primitives[0]
    
    # Get UV coordinates if available
    uvs = None
    if hasattr(primitive.attributes, 'TEXCOORD_0') and primitive.attributes.TEXCOORD_0 is not None:
        uvs = read_accessor(gltf, primitive.attributes.TEXCOORD_0, base_dir)
    
    uvs = uvs.copy()  # Create a copy to make it writable
    uvs[:, 1] = 1.0 - uvs[:, 1]    
    
    # Get material information
    if primitive.material is None:
        return None, uvs
    
    material = gltf.materials[primitive.material]
    
    # Check for PBR metallic roughness material (most common)
    if hasattr(material, 'pbrMetallicRoughness') and material.pbrMetallicRoughness:
        pbr = material.pbrMetallicRoughness
        if hasattr(pbr, 'baseColorTexture') and pbr.baseColorTexture:
            texture_idx = pbr.baseColorTexture.index
            texture = gltf.textures[texture_idx]
            image_idx = texture.source
            image = gltf.images[image_idx]
            
            if hasattr(image, 'uri') and image.uri:
                texture_path = os.path.join(base_dir, image.uri)
                return texture_path, uvs
    
    # Check for legacy diffuse texture
    if hasattr(material, 'diffuseTexture') and material.diffuseTexture:
        texture_idx = material.diffuseTexture.index
        texture = gltf.textures[texture_idx]
        image_idx = texture.source
        image = gltf.images[image_idx]
        
        if hasattr(image, 'uri') and image.uri:
            texture_path = os.path.join(base_dir, image.uri)
            return texture_path, uvs
    
    # If no specific diffuse texture found, try the first available texture
    if gltf.textures:
        texture = gltf.textures[0]
        image_idx = texture.source
        image = gltf.images[image_idx]
        
        if hasattr(image, 'uri') and image.uri:
            texture_path = os.path.join(base_dir, image.uri)
            return texture_path, uvs
    
    return None, uvs

if __name__ == "__main__":
    import argparse
    from src.lbs_matrix import lbs_matrix_column
    from src.util import matrixize
    from vedo import Mesh, Points, Lines, Plotter
    from vedo.applications import AnimationPlayer
    
    parser = argparse.ArgumentParser(description='read data from gltf')
    parser.add_argument('--input', '-i', type=str, required=True, help='gltf input path')
    
    args = parser.parse_args()
    input_path = args.input
    
    V0, F0, V, T, F, TF_list, C, BE, W, dt = read_gltf_data(input_path)
    J = lbs_matrix_column(V, W)
    
    V0n_list = [V]
    
    for ai, TF in enumerate(TF_list):
        p = TF.T.flatten()
        V0Col = J * p
        V0n = matrixize(V0Col)
        V0n_list.append(V0n)
    
    surface_idx = np.unique(F.flatten())
    inner_idx = np.setdiff1d(np.arange(V.shape[0]), surface_idx)
    
    texture_path, uvs = get_texture_info(input_path)
    V0toV = surface_cast_barycentric(V0, F0, V, T)
    V0_cast = V0toV @ V
    
    mesh0 = Mesh([V0_cast, F0]).texture(texture_path, tcoords=uvs)
    plt0 = Plotter()
    plt0 += [mesh0]
    plt0.show()
    
    mesh = Mesh([V0toV @ V0n_list[0], F0]).texture(texture_path, tcoords=uvs)
    
    cp = Points(C, r=5, c='red')
    bones = Lines(C[BE[:, 0]], C[BE[:, 1]], c='green')

    def update_scene(i: int):
        # update block and spring position at frame i
        if i == 0: plt.add(cp, bones)
        else: plt.remove(cp, bones)
        mesh.points = V0toV @ V0n_list[i]
        plt.render()
        
    plt = AnimationPlayer(update_scene, irange=[0,len(V0n_list)], loop=True, dt=33)
    plt += [mesh]  # Add control points to the scene
    plt.set_frame(0)
    plt.show()