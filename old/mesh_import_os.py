import os
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Patch
from matplotlib.collections import PatchCollection


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


def read_mesh_file(
    file_path: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    Read node coordinates and connectivity list from Gmsh .msh file (v4.1).
    Supports 2D quad4 (type 3) or tri3 (type 2) elements.
    Also extracts material indices (physical groups) for elements.
    Assumes physical tags: e.g., 1 for material 1 (Hydrogel), 2 for material 2 (Calcium).
    :param file_path: Path to .msh file.
    :return: Node coordinates [no. nodes, 2], connectivity list [no. elements, no. element nodes],
             idx2: indices of elements with physical tag 2 (or None if no tag 2).
    """
    with open(file_path, "r") as file:
        content = (
            file.read().replace("\n", " ").split()
        )  # Flatten to tokens for easier parsing
        i = 0

    def skip_section(end_marker):
        nonlocal i
        while i < len(content) and content[i] != end_marker:
            i += 1
        if i < len(content) and content[i] == end_marker:
            i += 1
        else:
            raise ValueError(f"Missing section end marker: {end_marker}")

    # Skip MeshFormat
    if content[i] != "$MeshFormat":
        raise ValueError("File does not start with $MeshFormat")
    i += 1
    skip_section("$EndMeshFormat")

    # Skip Entities
    if content[i] != "$Entities":
        raise ValueError("Missing $Entities section")
    i += 1
    skip_section("$EndEntities")

    # Read Nodes
    if content[i] != "$Nodes":
        raise ValueError(f"Missing $Nodes section; found {content[i]} instead")
    i += 1
    if i >= len(content):
        raise ValueError("Unexpected end of file after $Nodes")
    num_nodes = int(content[i])
    i += 1
    nodes = []
    node_counts = 0
    for _ in range(num_nodes):
        if i >= len(content):
            raise ValueError("Unexpected end of file in nodes section")
        node_tag = int(content[i])
        i += 1
        x_coord = float(content[i])
        i += 1
        y_coord = float(content[i])
        i += 1
        z_coord = float(content[i])
        i += 1
        # Parametric coordinates (num_param)
        if i >= len(content):
            raise ValueError("Unexpected end of file in node params")
        num_param = int(content[i])
        i += 1
        for __ in range(num_param):
            if i >= len(content):
                raise ValueError("Unexpected end of file in parametric coords")
            param = float(content[i])
            i += 1
        nodes.append([x_coord, y_coord])
        node_counts += 1
    skip_section("$EndNodes")
    print(f"Parsed {node_counts} nodes (expected {num_nodes})")  # Debug

    # Read Elements
    if content[i] != "$Elements":
        raise ValueError(f"Missing $Elements section; found {content[i]} instead")
    i += 1
    if i >= len(content):
        raise ValueError("Unexpected end of file after $Elements")
    num_elements = int(content[i])
    i += 1
    elements = []
    physical_tags = []
    el_node_counts = {
        2: 3,
        3: 4,
    }  # tri3:3, quad4:4; add more if needed (e.g., 5:8 for quad8)
    for _ in range(num_elements):
        if i >= len(content):
            raise ValueError("Unexpected end of file in elements section")
        el_tag = int(content[i])
        i += 1
        el_type = int(content[i])
        i += 1
        if el_type not in el_node_counts:
            raise ValueError(
                f"Unsupported element type {el_type}; add to el_node_counts"
            )
        num_nodes_el = el_node_counts[el_type]
        num_tags = int(content[i])
        i += 1
        tags = [int(content[i + j]) for j in range(num_tags)]
        i += num_tags
        node_tags = [int(content[i + j]) for j in range(num_nodes_el)]
        i += num_nodes_el
        # 1-based to 0-based
        el_nodes = [nt - 1 for nt in node_tags]
        elements.append(el_nodes)
        # Physical tag: assume tags[1] if exists (after entity tag), else 0 (material 1)
        phys_tag = tags[1] if len(tags) > 1 else (tags[0] if tags else 0)
        physical_tags.append(phys_tag)
    skip_section("$EndElements")

    x = torch.tensor(nodes, dtype=torch.float32)
    conn = torch.tensor(elements, dtype=torch.long)
    conn = sort_element_nodes(x, conn)

    # idx2: indices where physical_tag == 2 (Calcium)
    phys_tensor = torch.tensor(physical_tags)
    idx2 = (
        torch.nonzero(phys_tensor == 2).squeeze(-1)
        if (phys_tensor == 2).any()
        else None
    )

    print(
        f"Parsed {len(elements)} elements with physical tags: {set(physical_tags)}"
    )  # Debug
    return x, conn, idx2


def plot_mesh(
    x: torch.Tensor,
    conn: torch.Tensor,
    idx2: torch.Tensor | None = None,
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
    fig, ax = plt.subplots(figsize=(10, 10))

    if idx2 is not None and len(idx2) > 0:
        # Two materials
        mask2 = torch.zeros(len(conn), dtype=torch.bool)
        mask2[idx2] = True

        patches1, patches2 = [], []
        for element in conn[~mask2]:
            patches1.append(Polygon(x[element].numpy()))  # .numpy() for matplotlib
        if patches1:
            patches1 = PatchCollection(
                patches1,
                edgecolor="black",
                alpha=0.3,
                linewidths=0.5,
                facecolors="blue",
            )
            ax.add_collection(patches1)

        for element in conn[mask2]:
            patches2.append(Polygon(x[element].numpy()))
        if patches2:
            patches2 = PatchCollection(
                patches2,
                edgecolor="black",
                alpha=0.3,
                linewidths=0.5,
                facecolors="black",
            )
            ax.add_collection(patches2)

        legend_elements = [
            Patch(facecolor="blue", alpha=0.3, edgecolor="black", label="Hydrogel"),
            Patch(facecolor="black", alpha=0.3, edgecolor="black", label="Calcium"),
        ]
        ax.legend(handles=legend_elements)
    else:
        # Homogeneous material (all blue)
        patches1 = []
        for element in conn:
            patches1.append(Polygon(x[element].numpy()))
        if patches1:
            patches1 = PatchCollection(
                patches1,
                edgecolor="black",
                alpha=0.3,
                linewidths=0.5,
                facecolors="blue",
            )
            ax.add_collection(patches1)

        legend_elements = [
            Patch(facecolor="blue", alpha=0.3, edgecolor="black", label="Hydrogel"),
        ]
        ax.legend(handles=legend_elements)

    # Plot nodes
    ax.scatter(x[:, 0], x[:, 1], s=5, color="red", label="Nodes")
    ax.set_aspect("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Mesh")

    if directory is not None:
        os.makedirs(directory, exist_ok=True)
        plt.savefig(f"{directory}/mesh.png", bbox_inches="tight", dpi=150)
    plt.show()


# Main execution
path = "C:/Users/Sebastian/Desktop/Coding/FEM2/old/FEM2_Balken_HA1.msh"  # Fixed double-slash
safe_path = "C:/Users/Sebastian/Desktop/Coding/FEM2/old/"

x, conn, idx2 = read_mesh_file(path)
plot_mesh(x, conn, idx2, safe_path)
