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

    @staticmethod
    def create_rectilinear_mesh(
        rows: int,
        columns: int,
        origin_x: float = 0.0,
        origin_y: float = 0.0,
        spacing_x: float = 1.0,
        spacing_y: float = 1.0,
    ) -> Mesh2d:
        """Create a Mesh2d instance describing a rectilinear mesh

        Args:
            rows (int): The number of rows
            columns (int): The number of columns
            origin_x (float, optional): The x-coordinate of the origin. Defaults to 0.0.
            origin_y (float, optional): The y-coordinate of the origin. Defaults to 0.0.
            spacing_x (float, optional): The spacing between the columns. Defaults to 1.0.
            spacing_y (float, optional): The spacing between the rows. Defaults to 1.0.

        Returns:
            Mesh2d: The calculated rectilinear mesh
        """

        assert spacing_x > 0, "spacing_x needs to be positive"
        assert spacing_y > 0, "spacing_y needs to be positive"

        # Initialize helper objects
        num_nodes = rows * columns
        indices_values = np.empty((rows, columns))

        # Allocate memory for mesh arrays
        node_x = np.empty(num_nodes, dtype=np.double)
        node_y = np.empty(num_nodes, dtype=np.double)
        edge_nodes = np.empty(2 * (2 * num_nodes - rows - columns), dtype=np.int32)

        # Calculate node positions
        node_index = 0
        for row_index in range(rows):
            for column_index in range(columns):
                node_x[node_index] = column_index * spacing_x + origin_x
                node_y[node_index] = row_index * spacing_y + origin_y
                indices_values[row_index, column_index] = (
                    row_index * columns + column_index
                )
                node_index += 1

        # Calculate edge indices
        edge_index = 0
        for row_index in range(rows - 1):
            for column_index in range(columns):
                edge_nodes[edge_index] = indices_values[row_index, column_index]
                edge_index += 1
                edge_nodes[edge_index] = indices_values[row_index + 1, column_index]
                edge_index += 1
        for row_index in range(rows):
            for column_index in range(columns - 1):
                edge_nodes[edge_index] = indices_values[row_index, column_index + 1]
                edge_index += 1
                edge_nodes[edge_index] = indices_values[row_index, column_index]
                edge_index += 1

        return Mesh2d(node_x, node_y, edge_nodes)
