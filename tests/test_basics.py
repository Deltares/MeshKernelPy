import math

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from meshkernel import (
    DeleteMeshOption,
    GeometryList,
    InputError,
    Mesh2d,
    Mesh2dFactory,
    MeshKernel,
    MeshKernelError,
)

cases_constructor = [(True), (False)]


@pytest.mark.parametrize("is_geometric", cases_constructor)
def test_constructor(is_geometric: bool):
    """Test if the constructor works"""
    MeshKernel(is_geometric)


def test_different_id():
    """Test if the meshkernelid of two instances differs"""
    meshlib_1 = MeshKernel(False)
    meshlib_2 = MeshKernel(False)

    assert meshlib_1._meshkernelid != meshlib_2._meshkernelid


def test_set_mesh():
    """Test to set a simple mesh and then get it again with new parameters

    3---2
    |   |
    0---1
    """
    meshlib = MeshKernel(False)

    edge_nodes = np.array([0, 1, 1, 2, 2, 3, 3, 0], dtype=np.int32)
    node_x = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.double)
    node_y = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.double)

    input_mesh2d = Mesh2d(node_x, node_y, edge_nodes)
    meshlib.set_mesh2d(input_mesh2d)

    output_mesh2d = meshlib.get_mesh2d()

    # Test if the input and output differs
    assert_array_equal(output_mesh2d.edge_nodes, input_mesh2d.edge_nodes)
    assert_array_equal(output_mesh2d.node_x, input_mesh2d.node_x)
    assert_array_equal(output_mesh2d.node_y, input_mesh2d.node_y)

    # Test if faces are correctly calculated
    assert_array_equal(output_mesh2d.face_nodes, np.array([0, 1, 2, 3]))
    assert_array_equal(output_mesh2d.nodes_per_face, np.array([4]))
    assert_array_equal(output_mesh2d.face_x, np.array([0.5]))
    assert_array_equal(output_mesh2d.face_y, np.array([0.5]))

    # Test if edges are correctly calculated
    assert_array_equal(output_mesh2d.edge_x, np.array([0.5, 1.0, 0.5, 0.0]))
    assert_array_equal(output_mesh2d.edge_y, np.array([0.0, 0.5, 1.0, 0.5]))


cases_delete_node_mesh2d = [
    (0, 0.0, 0.0),
    (1, 1.0, 0.0),
    (2, 2.0, 0.0),
    (3, 0.0, 1.0),
    (4, 1.0, 1.0),
    (5, 2.0, 1.0),
    (6, 0.0, 2.0),
    (7, 1.0, 2.0),
    (8, 2.0, 2.0),
]


@pytest.mark.parametrize("node_index, deleted_x, deleted_y", cases_delete_node_mesh2d)
def test_delete_node_mesh2d(node_index: int, deleted_y: float, deleted_x: float):
    """Test `delete_node_mesh2d` by deleting a node from a 3x3 Mesh2d.

    6---7---8
    |   |   |
    3---4---5
    |   |   |
    0---1---2

    """
    mesh2d = Mesh2dFactory.create_rectilinear_mesh(3, 3)
    meshkernel = MeshKernel(False)

    meshkernel.set_mesh2d(mesh2d)

    meshkernel.delete_node_mesh2d(node_index)

    mesh2d = meshkernel.get_mesh2d()

    assert mesh2d.node_x.size == 8

    for x, y in zip(mesh2d.node_x, mesh2d.node_y):
        assert x != deleted_x or y != deleted_y


def test_delete_node_mesh2d_invalid_node_index():
    """Test `delete_node_mesh2d` by passing a negative `node_index`.

    6---7---8
    |   |   |
    3---4---5
    |   |   |
    0---1---2

    """
    mesh2d = Mesh2dFactory.create_rectilinear_mesh(3, 3)
    meshkernel = MeshKernel(False)

    meshkernel.set_mesh2d(mesh2d)

    meshkernel.delete_node_mesh2d(0)

    with pytest.raises(InputError):
        meshkernel.delete_node_mesh2d(-1)


def test_delete_mesh2d():
    """Test `delete_mesh2d` by deleting a polygon from a 5x5 mesh2d.

    20--21--22--23--24
    |   |   |   |   |
    15--16--17--18--19
    |   |   |   |   |
    10--11--12--13--14
    |   |   |   |   |
    5---6---7---8---9
    |   |   |   |   |
    0---1---2---3---4

    """
    mesh2d = Mesh2dFactory.create_rectilinear_mesh(5, 5)
    meshkernel = MeshKernel(False)

    meshkernel.set_mesh2d(mesh2d)

    # Polygon for indices 6, 8, 18, 16
    x_coordinates = np.array([1.0, 3.0, 3.0, 1.0], dtype=np.double)
    y_coordinates = np.array([1.0, 1.0, 3.0, 3.0], dtype=np.double)

    geometry_list = GeometryList(x_coordinates, y_coordinates)
    delete_option = DeleteMeshOption.ALLNODES
    invert_deletion = False

    meshkernel.delete_mesh2d(geometry_list, delete_option, invert_deletion)
    mesh2d = meshkernel.get_mesh2d()

    # assert mesh2d.node_x.size == 16
    # assert mesh2d.edge_x.size == 16
    # assert mesh2d.face_x.size == 1


cases_count_hanging_edges_mesh2d = [
    (
        np.array([0.0, 1.0, 1.0, 0.0], dtype=np.double),  # node_x
        np.array([0.0, 0.0, 1.0, 1.0], dtype=np.double),  # node_y
        np.array([0, 1, 1, 3, 2, 3, 2, 0], dtype=np.int32),  # edge_nodes
        0,
    ),
    (
        np.array([0.0, 1.0, 1.0, 0.0, 0.0], dtype=np.double),  # node_x
        np.array([0.0, 0.0, 1.0, 1.0, 2.0], dtype=np.double),  # node_y
        np.array([0, 1, 1, 3, 2, 3, 2, 0, 3, 4], dtype=np.int32),  # edge_nodes
        1,
    ),
    (
        np.array([0.0, 1.0, 1.0, 0.0, 0.0, 2.0], dtype=np.double),  # node_x
        np.array([0.0, 0.0, 1.0, 1.0, 2.0, 1.0], dtype=np.double),  # node_y
        np.array([0, 1, 1, 3, 2, 3, 2, 0, 3, 4, 2, 5], dtype=np.int32),  # edge_nodes
        2,
    ),
]


@pytest.mark.parametrize(
    "node_x, node_y, edge_nodes, expected", cases_count_hanging_edges_mesh2d
)
def test_count_hanging_edges_mesh2d(
    node_x: np.array, node_y: np.array, edge_nodes: np.array, expected: int
):
    """Test to count the hanging edges in a simple Mesh2d
    5*
    |
    2---3---4*
    |   |
    0---1
    """

    meshlib = MeshKernel(False)

    mesh2d = Mesh2d(node_x, node_y, edge_nodes)

    meshlib.set_mesh2d(mesh2d)

    result = meshlib.count_hanging_edges_mesh2d()

    assert result == expected
