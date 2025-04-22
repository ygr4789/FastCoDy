import bpy
import numpy as np
import math
import os
from mathutils import Euler

def export_anim(file_path, export_path, armature_name = "Armature"):
    bpy.ops.wm.open_mainfile(filepath=file_path)

    # === Configuration ===
    armature = bpy.data.objects[armature_name]
    bones = armature.pose.bones
    n_bones = len(bones)

    # === Get number of frames in animation
    scene = bpy.context.scene
    start_frame = scene.frame_start
    end_frame = scene.frame_end
    n_frames = end_frame - start_frame + 1

    # === Get bone hierarchy (needed for decoding)
    bone_names = [bone.name for bone in bones]
    bone_parents = [-1 if bone.parent is None else bone_names.index(bone.parent.name) for bone in bones]

    # === Helper: format vector
    def format_vec(vec):
        return '\t'.join(f"{v:.9f}" for v in vec)

    # === Helper: to degrees
    def radians_to_degrees(euler):
        return [math.degrees(a) for a in euler]

    with open(export_path, 'w') as f:

        # === 1. Write header: num_bones and num_frames
        f.write(f"{n_bones}\t{n_frames}\n")

        # === 2. Root rest translation (object-level location)
        rest_loc = armature.location
        f.write(format_vec(rest_loc) + "\n")

        # === 3. Root rest rotation (object-level rotation in Euler)
        rest_rot_euler = armature.rotation_euler  # assumes Euler XYZ mode
        f.write(format_vec(radians_to_degrees(rest_rot_euler)) + "\n")

        # === 4. Per-bone rest Euler angles
        for bone in bones:
            rest_rot = bone.bone.matrix_local.to_euler('XYZ')  # rest pose in local space
            f.write(format_vec(radians_to_degrees(rest_rot)) + "\n")

        # === 5. Per-frame animation
        for frame in range(start_frame, end_frame + 1):
            scene.frame_set(frame)

            # 5.1 Root rotation (object)
            rot = armature.rotation_euler
            f.write(format_vec(radians_to_degrees(rot)) + "\n")

            # 5.2 Root translation (object)
            loc = armature.location
            f.write(format_vec(loc) + "\n")

            # 5.3 Bone local Euler rotations (animated)
            for bone in bones:
                local_rot = bone.rotation_euler
                f.write(format_vec(radians_to_degrees(local_rot)) + "\n")

    print(f"✔ Exported anim.txt to {export_path}")
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='export animation from blender')
    parser.add_argument('--input', '-i', type=str, required=True, help='input blender file path')
    parser.add_argument('--output', '-o', type=str, required=False, default='output_anim.txt', help='output anim export path')
    parser.add_argument('--armature', '-a', type=str, required=False, default='Armature', help='target armature to export')
    
    args = parser.parse_args()
    file_path = args.input
    export_path = args.output
    armature = args.armature
    export_anim(file_path, export_path, armature)