import numpy as np

from meshkernel.errors import InputError
from meshkernel.py_structures import Mesh2d

class Mesh2dFactory:
    @staticmethod
    def create(
            rows: int,
            columns: int,
            origin_x: float = 0.0,
            origin_y: float = 0.0,
            spacing_x: float = 1.0,
            spacing_y: float = 1.0,
    ) -> Mesh2d:
        """Create a Mesh2d instance describing a rectilinear mesh.

        Args:
            rows (int): The number of rows.
            columns (int): The number of columns.
            origin_x (float, optional): The x-coordinate of the origin. Defaults to 0.0.
            origin_y (float, optional): The y-coordinate of the origin. Defaults to 0.0.
            spacing_x (float, optional): The spacing between the columns. Defaults to 1.0.
            spacing_y (float, optional): The spacing between the rows. Defaults to 1.0.

        Returns:
            Mesh2d: The calculated rectilinear mesh.
        """

        # Validate input
        if rows < 1:
            raise InputError("There needs to be at least one row.")
        if not isinstance(rows, int):
            raise InputError("`rows` needs to be an integer.")
        if columns < 1:
            raise InputError("There needs to be at least one column.")
        if not isinstance(columns, int):
            raise InputError("`columns` needs to be an integer.")
        if spacing_x <= 0:
            raise InputError("`spacing_x` needs to be positive.")
        if spacing_y <= 0:
            raise InputError("`spacing_y` needs to be positive.")

        # Convert to node rows and columns
        node_rows = rows + 1
        node_columns = columns + 1

        # Initialize helper objects
        num_nodes = node_rows * node_columns
        indices_values = np.empty((node_rows, node_columns))

        # Allocate memory for mesh arrays
        node_x = np.empty(num_nodes, dtype=np.double)
        node_y = np.empty(num_nodes, dtype=np.double)
        edge_nodes = np.empty(
            2 * (2 * num_nodes - node_rows - node_columns), dtype=np.int32
        )

        # Calculate node positions
        node_index = 0
        for row_index in range(node_rows):
            for column_index in range(node_columns):
                node_x[node_index] = column_index * spacing_x + origin_x
                node_y[node_index] = row_index * spacing_y + origin_y
                indices_values[row_index, column_index] = (
                        row_index * node_columns + column_index
                )
                node_index += 1

        # Calculate edge indices
        edge_index = 0
        for row_index in range(node_rows - 1):
            for column_index in range(node_columns):
                edge_nodes[edge_index] = indices_values[row_index, column_index]
                edge_index += 1
                edge_nodes[edge_index] = indices_values[row_index + 1, column_index]
                edge_index += 1
        for row_index in range(node_rows):
            for column_index in range(node_columns - 1):
                edge_nodes[edge_index] = indices_values[row_index, column_index + 1]
                edge_index += 1
                edge_nodes[edge_index] = indices_values[row_index, column_index]
                edge_index += 1

        return Mesh2d(node_x, node_y, edge_nodes)
