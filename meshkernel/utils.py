import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.collections import LineCollection


def plot_edges(node_x, node_y, edge_nodes, ax, *args, **kwargs):
    """Plots the edges at a given axes.
    `args` and `kwargs` will be used as parameters of the `plot` method.


    Args:
        node_x (ndarray): A 1D double array describing the x-coordinates of the nodes.
        node_y (ndarray): A 1D double array describing the y-coordinates of the nodes.
        edge_nodes (ndarray, optional): A 1D integer array describing the nodes composing each mesh 2d edge.
        ax (matplotlib.axes.Axes): The axes where to plot the edges
    """
    n_edge = int(edge_nodes.size / 2)
    edge_coords = np.empty((n_edge, 2, 2), dtype=np.float64)
    node_0 = edge_nodes[0::2]
    node_1 = edge_nodes[1::2]
    edge_coords[:, 0, 0] = node_x[node_0]
    edge_coords[:, 0, 1] = node_y[node_0]
    edge_coords[:, 1, 0] = node_x[node_1]
    edge_coords[:, 1, 1] = node_y[node_1]

    if "colors" not in kwargs:
        kwargs["colors"] = [
            mcolors.to_rgba(c)
            for c in plt.rcParams["axes.prop_cycle"].by_key()["color"]
        ]
    line_segments = LineCollection(edge_coords, *args, **kwargs)
    ax.add_collection(line_segments)
    ax.autoscale(enable=True)
