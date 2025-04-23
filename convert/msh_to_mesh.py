import numpy as np
import meshio
from collections import defaultdict

def extract_surface_faces(tetrahedrons):
    face_count = defaultdict(int)
    face_combinations = [
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 3],
        [1, 2, 3],
    ]
    
    for tet in tetrahedrons:
        for combo in face_combinations:
            face = tuple(sorted([tet[i] for i in combo]))  # sort so duplicates match
            face_count[face] += 1
    
    # A face that appears only once is on the surface
    surface_faces = [face for face, count in face_count.items() if count == 1]
    return np.array(surface_faces)

def convert_msh_to_mesh(msh_path, mesh_path):
    # Read tetrahedral mesh using meshio
    mesh = meshio.read(msh_path) # Read Gmsh .msh file
    points = mesh.points # (n_points, 3)
    cells = mesh.cells_dict
    tets = cells.get("tetra", [])
    surface_faces = extract_surface_faces(tets)
    
    # enforce MeshVersionFormatted to be 1
    points = points.astype(np.float32)
    tets = tets.astype(np.int32)
    surface_faces = surface_faces.astype(np.int32)

    mesh = meshio.Mesh(
        points=points,
        cells=[
            ("triangle", surface_faces),
            ("tetra", tets)
        ]
    )
    mesh.write(mesh_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='convert Gmsh .msh to meshio .mesh')
    parser.add_argument('--input', '-i', type=str, required=True, help='input msh file path')
    parser.add_argument('--output', '-o', type=str, required=False, default='output.mesh', help='output mesh export path')
    
    args = parser.parse_args()
    input = args.input
    output = args.output
    convert_msh_to_mesh(input, output)