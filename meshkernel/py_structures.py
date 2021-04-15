from dataclasses import dataclass
import numpy as np


@dataclass
class Mesh2d:
    """This class is used for getting and setting two-dimensional mesh data

    Attributes:
        # Read and write
        edge_nodes (np.ndarray(int)): The nodes composing each mesh 2d edge.
        node_x (np.ndarray(double)): The x-coordinates of the nodes.
        node_y (np.ndarray(double)): The y-coordinates of the nodes.
        # Read-only
        face_nodes (np.ndarray(int)): The nodes composing each mesh 2d face.
        nodes_per_face (np.ndarray(int)): The nodes composing each mesh 2d face.
        edge_x (np.ndarray(double)): The x-coordinates of the mesh edges middle points.
        edge_y (np.ndarray(double)): The x-coordinates of the mesh edges middle points.
        face_x (np.ndarray(double)): The x-coordinates of the mesh faces mass centers.
        face_y (np.ndarray(double)): The y-coordinates of the mesh faces mass centers.

    """

    edge_nodes: np.ndarray
    node_x: np.ndarray
    node_y: np.ndarray
    face_nodes: np.ndarray = np.empty(0, dtype=int)
    nodes_per_face: np.ndarray = np.empty(0, dtype=int)
    edge_x: np.ndarray = np.empty(0, dtype=np.double)
    edge_y: np.ndarray = np.empty(0, dtype=np.double)
    face_x: np.ndarray = np.empty(0, dtype=np.double)
    face_y: np.ndarray = np.empty(0, dtype=np.double)
