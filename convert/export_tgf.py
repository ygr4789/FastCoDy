import bpy
import os

def export_tgf(file_path, export_path, armature_name = "Armature"):
    bpy.ops.wm.open_mainfile(filepath=file_path)

    # === Config ===
    armature_name = "Armature"
    armature = bpy.data.objects[armature_name]
    bones = armature.data.bones

    # === Setup ===
    joint_positions = []
    joint_indices = {}  # (bone_name, 'head' or 'tail') → index
    head_added = {b.name: False for b in bones}
    tail_added = {b.name: False for b in bones}
    index_counter = 1

    # === Detect bones with no children (root bones) ===
    root_bone = bones[0]
    for b in bones:
        if not b.parent:
            root_bone = b
            continue

    # === Step 1: Add head of root bone first
    root_head_world = armature.matrix_world @ root_bone.head_local
    joint_positions.append((index_counter, root_head_world))
    joint_indices[(root_bone.name, 'head')] = index_counter
    head_added[root_bone.name] = True
    index_counter += 1

    # === Step 2: Add heads of remaining bones if not skipped by parent’s tail
    for bone in bones:
        if bone == root_bone: 
            continue

        parent = bone.parent.name if bone.parent else None
        if parent and tail_added[parent]:
            joint_indices[(bone.name, 'head')] = joint_indices[(parent.name, 'tail')]
            head_added[bone.name] = True
            continue  # Skip adding head if parent's tail is already added

        head_world = armature.matrix_world @ bone.head_local
        joint_positions.append((index_counter, head_world))
        joint_indices[(bone.name, 'head')] = index_counter
        head_added[bone.name] = True
        if parent:
            joint_indices[(parent, 'tail')] = index_counter
            tail_added[parent] = True
        index_counter += 1

    # === Step 3: Add all tails not already added
    for bone in bones:
        if tail_added[bone.name]:
            continue

        tail_world = armature.matrix_world @ bone.tail_local
        joint_positions.append((index_counter, tail_world))
        joint_indices[(bone.name, 'tail')] = index_counter
        tail_added[bone.name] = True
        index_counter += 1

    # === Step 4: Write edges (bones)
    edges = []

    root_head_idx = joint_indices.get((root_bone.name, 'head'))
    root_tail_idx = joint_indices.get((root_bone.name, 'tail'))
    edges.append((root_head_idx, root_tail_idx))

    for bone in bones:
        if bone == root_bone: continue
        head_idx = joint_indices.get((bone.name, 'head'))
        tail_idx = joint_indices.get((bone.name, 'tail'))
        edges.append((head_idx, tail_idx))

    # === Step 5: Write TGF file
    with open(export_path, 'w') as f:
        # Joint positions
        for idx, pos in joint_positions:
            f.write(f"{idx}\t{pos.x:.9f}\t{pos.y:.9f}\t{pos.z:.9f}\n")

        f.write("#\n")

        # Bone edges
        for src, dst in edges:
            f.write(f"{src}\t{dst}\t1\t0\t0\t0\n")

        f.write("#\n")

    print(f"✔ Exported .tgf to {export_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='export tgf from blender')
    parser.add_argument('--input', '-i', type=str, required=True, help='input blender file path')
    parser.add_argument('--output', '-o', type=str, required=False, default='output.tgf', help='output tgf export path')
    parser.add_argument('--armature', '-a', type=str, required=False, default='Armature', help='target armature to export')
    
    args = parser.parse_args()
    file_path = args.input
    export_path = args.output
    armature = args.armature
    export_tgf(file_path, export_path, armature)