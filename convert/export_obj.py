import bpy
import os

def export_obj(file_path, export_path):
    bpy.ops.wm.open_mainfile(filepath=file_path)
    
    bpy.ops.wm.obj_export(
        filepath=export_path,
        export_materials=False,    # Prevent MTL file export
        export_normals=True,
        export_uv=False,
        export_colors=False,
        forward_axis='Y',
        up_axis='Z'
    )
    
    print(f"✔ Exported .obj to {export_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='export obj from blender')
    parser.add_argument('--input', '-i', type=str, required=True, help='input blender file path')
    parser.add_argument('--output', '-o', type=str, required=False, default='output.obj', help='output obj export path')
    
    args = parser.parse_args()
    file_path = args.input
    export_path = args.output
    export_obj(file_path, export_path)