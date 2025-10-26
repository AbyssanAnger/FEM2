import os
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Patch
from matplotlib.collections import PatchCollection


def read_mesh_file(file_path: str) -> (torch.Tensor, torch.Tensor):
    """
    Read node coordinates and connectivity list from mesh file. Element nodes need to be sorted according to FEM
    convention. Meshes created with Gmsh or Abaqus will be sorted correctly.
    :param file_path: Path to mesh file.
    :return: Node coordinates [no. nodes, 2], connectivity list [no. elements, no. element nodes].
    """
    # Read lines in mesh file
    with open(file_path, "r") as file:
        lines = file.readlines()
    nodes, elements = [], []
    reading_nodes, reading_elements = False, False
    for line in lines:
        if line.startswith("*") or not line.strip():
            if line.lower().startswith("*node"):
                reading_nodes = True
            elif line.lower().startswith("*element"):
                reading_nodes = False
                reading_elements = True
            else:
                reading_nodes = False
                reading_elements = False
            continue
        if reading_nodes:
            node = [float(i) for i in line.split(",")][1:3]
            nodes.append(node)
        elif reading_elements:
            element = [int(i) - 1 for i in line.split(",")][1:]
            # Transform Q9 to Q8 serendipity elements
            if len(element) == 9:
                del element[8]
            elements.append(element)
    # Convert to torch tensors, sort connectivity list
    x = torch.tensor(nodes)
    conn = sort_element_nodes(x, torch.tensor(elements))
    return x, conn


def sort_element_nodes(x: torch.Tensor, conn: torch.Tensor) -> torch.Tensor:
    """
    Sort element nodes in connectivity list counterclockwise.
    :param x: Node coordinates [no. nodes, 2].
    :param conn: Connectivity list indicating which nodes belong to which element [no. elements, no. element nodes].
    :return: New connectivity list in which element nodes are sorted counterclockwise [no. elements, no. element nodes].
    """
    elements = x[conn]  # Node coordinates grouped by elements
    centroids = torch.mean(elements, dim=1)  # Centroid coordinates for each element
    angles = torch.atan2(
        elements[:, :, 1] - centroids[:, 1].unsqueeze(1),
        elements[:, :, 0] - centroids[:, 0].unsqueeze(1),
    )  # Angles between element nodes and centroids
    sorted_indices = torch.argsort(angles, dim=1)  # Sorted indices of element nodes
    ordered_conn = conn.gather(1, sorted_indices)  # Ordered connectivity list
    return ordered_conn


def plot_mesh(
    x: torch.Tensor,
    conn: torch.Tensor,
    idx2: torch.Tensor = None,
    directory: str = None,
):
    """
    Plot mesh as elements and nodes, distinguishing by materials.
    :param x: Node coordinates [no. nodes, 2].
    :param conn: Connectivity list [no. elements, no. element nodes].
    :param idx2: Indices of elements belonging to second material [no. elements]. Can be set to None if a homogeneous
    material is used.
    :param directory: Directory to save plot to. If None, plot is not saved.
    """
    # Make patch collection from elements, different colors for each material
    patches1, patches2 = [], []
    for element in conn[~idx2]:
        patches1.append(Polygon(x[element]))
    patches1 = PatchCollection(
        patches1, edgecolor="black", alpha=0.3, linewidths=0.5, facecolors="blue"
    )
    for element in conn[idx2]:
        patches2.append(Polygon(x[element]))
    patches2 = PatchCollection(
        patches2, edgecolor="black", alpha=0.3, linewidths=0.5, facecolors="black"
    )
    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.add_collection(patches1)
    ax.add_collection(patches2)
    ax.scatter(x[:, 0], x[:, 1], s=5, color="red", label="Nodes")
    ax.set_aspect("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Mesh")
    legend_elements = [
        Patch(facecolor="blue", alpha=0.3, edgecolor="black", label="Hydrogel"),
        Patch(facecolor="black", alpha=0.3, edgecolor="black", label="Calcium"),
    ]
    ax.legend(handles=legend_elements)
    if directory is not None:
        if not os.path.exists(f"{directory}"):
            os.makedirs(f"{directory}")
        plt.savefig(f"{directory}/mesh.png", bbox_inches="tight")
    plt.show()
