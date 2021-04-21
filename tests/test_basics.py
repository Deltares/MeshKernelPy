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
    meshkernel_1 = MeshKernel(False)
    meshkernel_2 = MeshKernel(False)

    assert meshkernel_1._meshkernelid != meshkernel_2._meshkernelid


def test_set_mesh():
    """Test to set a simple mesh and then get it again with new parameters

    3---2
    |   |
    0---1
    """
    meshkernel = MeshKernel(False)

    edge_nodes = np.array([0, 1, 1, 2, 2, 3, 3, 0], dtype=np.int32)
    node_x = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.double)
    node_y = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.double)

    input_mesh2d = Mesh2d(node_x, node_y, edge_nodes)
    meshkernel.set_mesh2d(input_mesh2d)

    output_mesh2d = meshkernel.get_mesh2d()

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


def test_insert_node_mesh2d():
    """Test `insert_node_mesh2d` with a 2x2 Mesh2d.

    3---2
    |   |
    0---1
    """

    meshkernel = _get_meshkernel_with_mesh(2, 2)

    node_index = meshkernel.insert_node_mesh2d(0.5, 0.0)

    mesh2d = meshkernel.get_mesh2d()

    # assert node_index == 4 TODO

    assert mesh2d.node_x.size == 5


@pytest.mark.parametrize("node_index, deleted_x, deleted_y", cases_delete_node_mesh2d)
def test_delete_node_mesh2d(node_index: int, deleted_x: float, deleted_y: float):
    """Test `delete_node_mesh2d` by deleting a node from a 3x3 Mesh2d.

    6---7---8
    |   |   |
    3---4---5
    |   |   |
    0---1---2

    """
    meshkernel = _get_meshkernel_with_mesh(3, 3)

    meshkernel.delete_node_mesh2d(node_index)

    mesh2d = meshkernel.get_mesh2d()

    assert mesh2d.node_x.size == 8

    for x, y in zip(mesh2d.node_x, mesh2d.node_y):
        assert x != deleted_x or y != deleted_y


def test_delete_node_mesh2d_invalid_node_index():
    """Test `delete_node_mesh2d` by passing a negative `node_index`."""

    meshkernel = _get_meshkernel_with_mesh(2, 2)

    with pytest.raises(InputError):
        meshkernel.delete_node_mesh2d(-1)


cases_move_node_mesh2d = [
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


@pytest.mark.parametrize("node_index, moved_x, moved_y", cases_move_node_mesh2d)
def test_move_node_mesh2d(node_index: int, moved_x: float, moved_y: float):
    """Test to move a node in a simple Mesh2d to new location.

    6---7---8
    |   |   |
    3---4---5
    |   |   |
    0---1---2

    """

    meshkernel = _get_meshkernel_with_mesh(3, 3)

    x_coordinates = np.array([5.0], dtype=np.double)
    y_coordinates = np.array([7.0], dtype=np.double)

    geometry_list = GeometryList(x_coordinates, y_coordinates)
    meshkernel.move_node_mesh2d(geometry_list, node_index)

    mesh2d = meshkernel.get_mesh2d()

    assert mesh2d.node_x[node_index] == 5.0
    assert mesh2d.node_y[node_index] == 7.0

    for x, y in zip(mesh2d.node_x, mesh2d.node_y):
        assert x != moved_x or y != moved_y


def test_move_node_mesh2d_invalid_node_index():
    """Test `move_node_mesh2d` by passing a negative `node_index`."""

    meshkernel = _get_meshkernel_with_mesh(2, 2)

    x_coordinates = np.array([5.0], dtype=np.double)
    y_coordinates = np.array([7.0], dtype=np.double)

    geometry_list = GeometryList(x_coordinates, y_coordinates)

    with pytest.raises(InputError):
        meshkernel.move_node_mesh2d(geometry_list, -1)


cases_delete_edge_mesh2d = [
    (0.5, 0.0),
    (1.5, 0.0),
    (0.0, 0.5),
    (1.0, 0.5),
    (2.0, 0.5),
    (0.5, 1.0),
    (1.5, 1.0),
    (0.0, 1.5),
    (1.0, 1.5),
    (2.0, 1.5),
    (0.5, 2.0),
    (1.5, 2.0),
]


@pytest.mark.parametrize("delete_x, delete_y", cases_delete_edge_mesh2d)
def test_delete_edge_mesh2d(delete_x: float, delete_y: float):
    """Test `delete_edge_mesh2d` by deleting an edge from a 3x3 Mesh2d.

    6---7---8
    |   |   |
    3---4---5
    |   |   |
    0---1---2

    """
    meshkernel = _get_meshkernel_with_mesh(3, 3)

    x_coordinate = np.array([delete_x], dtype=np.double)
    y_coordinate = np.array([delete_y], dtype=np.double)
    geometry_list = GeometryList(x_coordinate, y_coordinate)

    meshkernel.delete_edge_mesh2d(geometry_list)

    mesh2d = meshkernel.get_mesh2d()

    assert mesh2d.node_x.size == 9
    assert mesh2d.edge_x.size == 11
    assert mesh2d.face_x.size == 3

    for x, y in zip(mesh2d.edge_x, mesh2d.edge_y):
        assert x != delete_x or y != delete_y


cases_find_edge_mesh2d = [
    (0.5, 0.0, 2),
    (1.0, 0.5, 1),
    (0.5, 1.0, 3),
    (0.0, 0.5, 0),
]


@pytest.mark.parametrize("x, y, exp_index", cases_find_edge_mesh2d)
def test_find_edge_mesh2d(x: float, y: float, exp_index: int):
    """Test `find_edge_mesh2d` on a 2x2 Mesh2d.

        (3)
       2---3
    (0)|   |(1)
       0---1
        (2)

    """

    meshkernel = _get_meshkernel_with_mesh(2, 2)

    x_coordinate = np.array([x], dtype=np.double)
    y_coordinate = np.array([y], dtype=np.double)
    geometry_list = GeometryList(x_coordinate, y_coordinate)

    edge_index = meshkernel.find_edge_mesh2d(geometry_list)

    assert edge_index == exp_index


cases_delete_mesh2d_small_polygon = [
    (True, DeleteMeshOption.ALLNODES, 4, 4, 1),
    (True, DeleteMeshOption.ALLFACECIRCUMCENTERS, 16, 24, 9),
    (True, DeleteMeshOption.ALLCOMPLETEFACES, 4, 4, 1),
    (False, DeleteMeshOption.ALLNODES, 32, 48, 16),
    (False, DeleteMeshOption.ALLFACECIRCUMCENTERS, 32, 48, 16),
    (False, DeleteMeshOption.ALLCOMPLETEFACES, 36, 60, 25),
]


@pytest.mark.parametrize(
    "invert_deletion, delete_option, exp_nodes, exp_edges, exp_faces",
    cases_delete_mesh2d_small_polygon,
)
def test_delete_mesh2d_small_polygon(
    invert_deletion: bool,
    delete_option: DeleteMeshOption,
    exp_nodes: int,
    exp_edges: int,
    exp_faces: int,
):
    """Test `delete_mesh2d` by deleting a polygon from a 6x6 mesh2d.

    30--31--32--33--34--35
    |   |   |   |   |   |
    24--25--26--27--28--29
    |   | * |   | * |   |
    18--19--20--21--22--23
    |   |   |   |   |   |
    12--13--14--15--16--17
    |   | * |   | * |   |
    6---7---8---9---10--11
    |   |   |   |   |   |
    0---1---2---3---4---5

    """
    meshkernel = _get_meshkernel_with_mesh(6, 6)

    # Polygon around nodes 14, 15, 21 & 20 (through the face circum centers)
    x_coordinates = np.array([1.5, 3.5, 3.5, 1.5, 1.5], dtype=np.double)
    y_coordinates = np.array([1.5, 1.5, 3.5, 3.5, 1.5], dtype=np.double)

    geometry_list = GeometryList(x_coordinates, y_coordinates)

    meshkernel.delete_mesh2d(geometry_list, delete_option, invert_deletion)
    mesh2d = meshkernel.get_mesh2d()

    assert mesh2d.node_x.size == exp_nodes
    assert mesh2d.edge_x.size == exp_edges
    assert mesh2d.face_x.size == exp_faces


cases_delete_mesh2d_empty_polygon = [(False, 0, 0, 0), (True, 25, 40, 16)]


@pytest.mark.parametrize(
    "invert_deletion, exp_nodes, exp_edges, exp_faces",
    cases_delete_mesh2d_empty_polygon,
)
def test_delete_mesh2d_empty_polygon(
    invert_deletion: bool, exp_nodes: int, exp_edges: int, exp_faces: int
):
    """Test `delete_mesh2d` by deleting a an empty polygon from a 5x5 mesh2d.

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
    meshkernel = _get_meshkernel_with_mesh(5, 5)

    x_coordinates = np.empty(0, dtype=np.double)
    y_coordinates = np.empty(0, dtype=np.double)

    geometry_list = GeometryList(x_coordinates, y_coordinates)
    delete_option = DeleteMeshOption.ALLNODES

    meshkernel.delete_mesh2d(geometry_list, delete_option, invert_deletion)
    mesh2d = meshkernel.get_mesh2d()

    assert mesh2d.node_x.size == exp_nodes
    assert mesh2d.edge_x.size == exp_edges
    assert mesh2d.face_x.size == exp_faces


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

    meshkernel = MeshKernel(False)

    mesh2d = Mesh2d(node_x, node_y, edge_nodes)

    meshkernel.set_mesh2d(mesh2d)

    result = meshkernel.count_hanging_edges_mesh2d()

    assert result == expected


def _get_meshkernel_with_mesh(rows: int, columns: int) -> MeshKernel:
    """Creates a new instance of 'meshkernel' and sets a Mesh2d with the specified dimensions.

    Args:
        rows (int): Number of node rows
        columns (int): Number of node columns

    Returns:
        MeshKernel: The created instance of `meshkernel`
    """

    mesh2d = Mesh2dFactory.create_rectilinear_mesh(rows, columns)
    meshkernel = MeshKernel(False)

    meshkernel.set_mesh2d(mesh2d)

    return meshkernel
