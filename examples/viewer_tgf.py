import vedo
import numpy as np

def read_tgf(filepath):
    nodes = {}
    edges = []
    reading_edges = False

    with open(filepath, 'r') as file:
        for line in file:
            line = line.strip()
            if line == "#":
                reading_edges = True
                continue
            if not line:
                continue

            if reading_edges:
                parts = line.split()
                if len(parts) >= 2:
                    edges.append((int(parts[0]), int(parts[1])))
            else:
                parts = line.split()
                if len(parts) >= 4:
                    node_id = int(parts[0])
                    x, y, z = map(float, parts[1:4])
                    nodes[node_id] = np.array([x, y, z])

    return nodes, edges

def visualize_with_mesh(nodes, edges, mesh_path, mesh_alpha=0.3):
    plt = vedo.Plotter()

    actors = []

    # Draw joints
    for nid, coord in nodes.items():
        sp = vedo.Sphere(pos=coord, r=0.5, c='red')
        actors.append(sp)
        actors.append(vedo.Text3D(str(nid), pos=coord + 1.0, s=0.4))

    plt = vedo.Plotter()

    actors = []
    for nid, coord in nodes.items():
        sp = vedo.Sphere(pos=coord, r=0., c='red')
        actors.append(sp)
        txt = vedo.Text3D(str(nid), pos=coord + 1.5, s=0.8)
        actors.append(txt)

    for id1, id2 in edges:
        if id1 in nodes and id2 in nodes:
            line = vedo.Line(nodes[id1], nodes[id2], c='gray')
            actors.append(line)

    # Load and add mesh
    mesh = vedo.load(mesh_path)
    mesh.alpha(mesh_alpha)
    mesh.c("tan")  # Or any color you like
    actors.append(mesh)

    plt.show(*actors, axes=1)

if __name__ == "__main__":
    tgf_file = "human.tgf"
    mesh_file = "human.obj"
    nodes, edges = read_tgf(tgf_file)
    visualize_with_mesh(nodes, edges, mesh_file)