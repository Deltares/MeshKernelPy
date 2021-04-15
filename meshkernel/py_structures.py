from dataclasses import dataclass
import numpy as np


@dataclass
class Mesh2d:
    """This is a test class for dataclasses.

    This is the body of the docstring description.

    Attributes:
        edge_nodes (np.ndarray(int)): The nodes composing each mesh 2d edge.
        face_nodes (np.ndarray(int)): The nodes composing each mesh 2d face.
        nodes_per_face (np.ndarray(int)): The nodes composing each mesh 2d face.
        node_x (np.ndarray(double)): The x-coordinates of the nodes.
        node_y (np.ndarray(double)): The y-coordinates of the nodes.
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
