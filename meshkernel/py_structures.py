from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Mesh2d:
    """This class is used for getting and setting two-dimensional mesh data

    Attributes:
        node_x (np.ndarray(double)): The x-coordinates of the nodes.
        node_y (np.ndarray(double)): The y-coordinates of the nodes.
        edge_nodes (np.ndarray(int), optional): The nodes composing each mesh 2d edge.
        face_nodes (np.ndarray(int), optional): The nodes composing each mesh 2d face.
        nodes_per_face (np.ndarray(int), optional): The nodes composing each mesh 2d face.
        edge_x (np.ndarray(double), optional): The x-coordinates of the mesh edges' middle points.
        edge_y (np.ndarray(double), optional): The x-coordinates of the mesh edges' middle points.
        face_x (np.ndarray(double), optional): The x-coordinates of the mesh faces' mass centers.
        face_y (np.ndarray(double), optional): The y-coordinates of the mesh faces' mass centers.

    """

    node_x: np.ndarray
    node_y: np.ndarray
    edge_nodes: np.ndarray
    face_nodes: np.ndarray = np.empty(0, dtype=np.int32)
    nodes_per_face: np.ndarray = np.empty(0, dtype=np.int32)
    edge_x: np.ndarray = np.empty(0, dtype=np.double)
    edge_y: np.ndarray = np.empty(0, dtype=np.double)
    face_x: np.ndarray = np.empty(0, dtype=np.double)
    face_y: np.ndarray = np.empty(0, dtype=np.double)


@dataclass
class GeometryList:
    """A class to describe a list of geometries.

    Attributes:
        x_coordinates (np.ndarray(np.double)): The x coordinates.
        y_coordinates (np.ndarray(np.double)): The y coordinates.
        geometry_separator (float, optional): The value used as a separator in the coordinates. Default is `-999.0`
        inner_outer_separator (float, optional): The value used to separate the inner part of a polygon from its outer
                                                 part. Default is `-998.0`
    """

    x_coordinates: np.ndarray
    y_coordinates: np.ndarray
    geometry_separator: float = -999.0
    inner_outer_separator: float = -998.0
