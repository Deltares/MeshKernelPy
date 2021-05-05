from meshkernel.py_structures import Mesh2d


def plot_edges(mesh2d: Mesh2d):
    """Plots the edges of the given Mesh2d.

    Args:
        mesh2d (Mesh2d): The Mesh2d for which to plot the edges.
    """
    import matplotlib.pyplot as plt

    for edge_index in range(0, mesh2d.edge_nodes.size, 2):
        first_edge_node_index = mesh2d.edge_nodes[edge_index]
        second_edge_node_index = mesh2d.edge_nodes[edge_index + 1]

        edge_x = [
            mesh2d.node_x[first_edge_node_index],
            mesh2d.node_x[second_edge_node_index],
        ]
        edge_y = [
            mesh2d.node_y[first_edge_node_index],
            mesh2d.node_y[second_edge_node_index],
        ]

        plt.plot(edge_x, edge_y, "b-")

    plt.show()


def plot_nodes(mesh2d: Mesh2d):
    """Plots the nodes of the given Mesh2d.

    Args:
        mesh2d (Mesh2d): The Mesh2d for which to plot the nodes.
    """
    import matplotlib.pyplot as plt

    # Plot white points only for scaling the plot
    plt.plot(mesh2d.node_x, mesh2d.node_y, "ow")

    # Numbering the nodes
    for i in range(mesh2d.node_x.size):
        plt.annotate(
            int(i),
            xy=(mesh2d.node_x[i], mesh2d.node_y[i]),
            ha="center",
            va="center",
            fontsize=12,
            color="blue",
        )

    plt.show()


def plot_faces(mesh2d: Mesh2d):
    """Plots the faces of the given Mesh2d.

    Args:
        mesh2d (Mesh2d): The Mesh2d for which to plot the faces.
    """
    import matplotlib.pyplot as plt

    node_position = 0
    for face_index, num_nodes in enumerate(mesh2d.nodes_per_face):
        # Calculate values to draw
        face_nodes = mesh2d.face_nodes[node_position : (node_position + num_nodes)]
        face_nodes_x = mesh2d.node_x[face_nodes]
        face_nodes_y = mesh2d.node_y[face_nodes]
        face_x = mesh2d.face_x[face_index]
        face_y = mesh2d.face_y[face_index]
        node_position += num_nodes

        # Draw polygon
        plt.fill(face_nodes_x, face_nodes_y)
        # Draw face index at its center
        plt.text(face_x, face_y, face_index, ha="center", va="center", fontsize=22)

    plt.show()
