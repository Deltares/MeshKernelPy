import numpy as np
import pytest
from numpy.testing import assert_array_equal

from meshkernel import InputError, Mesh2d, MeshKernel, __version__


def read_asc_file(file_path):
    """Reads asc file and returns headers and data as numpy array
    Args:
            file_path (str): The file path
    Returns:
            header: The ascii header with
            data: The ascii data as numpy array
        """

    header = {}
    data = []

    with open(file_path, 'r') as file:
        # Read header information
        for _ in range(6):
            line = file.readline().strip().split()
            header[line[0]] = float(line[1])

        # Read data values
        for line in file:
            data_row = [float(value) for value in line.strip().split()]
            data.insert(0, data_row)  # Insert row at the beginning

        # Flatten the
        data = np.array(data).flatten().astype(np.double)

    return header, data
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


def test_create_rectilinear_mesh_simple():
    """Test create_rectilinear_mesh``by creating a simple 2x2 mesh

    6---7---8
    |   |   |
    3---4---5
    |   |   |
    0---1---2

    """
    mesh2d = Mesh2dFactory.create(2, 2)

    # Assert node positions
    assert_array_equal(
        mesh2d.node_x, np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0])
    )
    assert_array_equal(
        mesh2d.node_y, np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
    )

    # Assert indices of edge nodes
    assert_array_equal(
        mesh2d.edge_nodes,
        np.array(
            [0, 3, 1, 4, 2, 5, 3, 6, 4, 7, 5, 8, 1, 0, 2, 1, 4, 3, 5, 4, 7, 6, 8, 7]
        ),
    )


def test_create_rectilinear_mesh_extensive():
    """Test create_rectilinear_mesh``by creating a 2x3 mesh.
    Also set
        - origin_x to -1.0 and spacing_x to 2.0 and
        - origin_y to 1.0 and spacing_y to 3.0.


    8---9---10---11
    |   |   |   |
    4---5---6---7
    |   |   |   |
    0---1---2---3

    """
    mesh2d = Mesh2dFactory.create(
        2, 3, origin_x=-1.0, origin_y=1.0, spacing_x=2.0, spacing_y=3.0
    )

    # Assert node positions
    assert_array_equal(
        mesh2d.node_x,
        np.array([-1.0, 1.0, 3.0, 5.0, -1.0, 1.0, 3.0, 5.0, -1.0, 1.0, 3.0, 5.0]),
    )
    assert_array_equal(
        mesh2d.node_y,
        np.array([1.0, 1.0, 1.0, 1.0, 4.0, 4.0, 4.0, 4.0, 7.0, 7.0, 7.0, 7.0]),
    )

    # Assert indices of edge nodes
    assert_array_equal(
        mesh2d.edge_nodes,
        np.array(
            [
                0,
                4,
                1,
                5,
                2,
                6,
                3,
                7,
                4,
                8,
                5,
                9,
                6,
                10,
                7,
                11,
                1,
                0,
                2,
                1,
                3,
                2,
                5,
                4,
                6,
                5,
                7,
                6,
                9,
                8,
                10,
                9,
                11,
                10,
            ]
        ),
    )


def test_create_rectilinear_mesh_reject_negative_spacing():
    """Tests if `create_rectilinear_mesh` rejects negative spacing."""
    with pytest.raises(InputError):
        Mesh2dFactory.create(2, 2, spacing_x=-1.0)

    with pytest.raises(InputError):
        Mesh2dFactory.create(2, 2, spacing_y=-1.0)


def test_create_rectilinear_mesh_reject_negative_rows_columns():
    """Tests if `create_rectilinear_mesh` rejects negative spacing."""
    with pytest.raises(InputError):
        Mesh2dFactory.create(-1, 2)

    with pytest.raises(InputError):
        Mesh2dFactory.create(2, -1)


def test_get_meshkernel_version():
    """Tests if we can get the version of MeshKernel through the API"""
    mk = MeshKernel()
    meshkernel_version = mk.get_meshkernel_version()
    assert len(meshkernel_version) > 0


def test_get_meshkernelpy_version():
    """Tests if we can get the version of MeshKernelPy through the API"""
    mk = MeshKernel()
    meshkernelpy_version = mk.get_meshkernelpy_version()
    assert meshkernelpy_version == __version__
