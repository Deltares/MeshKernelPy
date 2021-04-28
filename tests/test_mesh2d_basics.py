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


@pytest.fixture(scope="function")
def meshkernel_with_mesh2d() -> MeshKernel:
    """Creates a new instance of 'meshkernel' and sets a Mesh2d with the specified dimensions.

    Args:
        rows (int): Number of node rows
        columns (int): Number of node columns

    Returns:
        MeshKernel: The created instance of `meshkernel`
    """

    def _create(rows: int, columns: int):
        mesh2d = Mesh2dFactory.create_rectilinear_mesh(rows, columns)
        mk = MeshKernel(False)

        mk.set_mesh2d(mesh2d)

        return mk

    return _create


cases_is_geometric_constructor = [(True), (False)]


@pytest.mark.parametrize("is_geometric", cases_is_geometric_constructor)
def test_constructor(is_geometric: bool):
    """Test if the constructor works"""
    MeshKernel(is_geometric)


def test_different_instances_have_different_ids():
    """Test if the meshkernelid of two instances differs"""
    mk_1 = MeshKernel(False)
    mk_2 = MeshKernel(False)

    assert mk_1._meshkernelid != mk_2._meshkernelid


def test_set_mesh_and_get_mesh():
    """Test to set a simple mesh and then get it again with new parameters

    3---2
    |   |
    0---1
    """
    mk = MeshKernel(False)

    edge_nodes = np.array([0, 1, 1, 2, 2, 3, 3, 0], dtype=np.int32)
    node_x = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.double)
    node_y = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.double)

    input_mesh2d = Mesh2d(node_x, node_y, edge_nodes)
    mk.set_mesh2d(input_mesh2d)

    output_mesh2d = mk.get_mesh2d()

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


def test_insert_edge_mesh2d(meshkernel_with_mesh2d: MeshKernel):
    """Test `insert_edge_mesh2d` by inserting one edge within a 2x2 Mesh2d.

    2---3
    |   |
    0---1
    """

    mk = meshkernel_with_mesh2d(2, 2)

    edge_index = mk.insert_edge_mesh2d(0, 3)

    mesh2d = mk.get_mesh2d()

    assert edge_index == 4
    assert mesh2d.node_x.size == 4
    assert mesh2d.edge_x.size == 5
    assert mesh2d.face_x.size == 2


def test_insert_node_mesh2d(meshkernel_with_mesh2d: MeshKernel):
    """Test `insert_node_mesh2d` with a 2x2 Mesh2d.

    2---3
    |   |
    0---1
    """

    mk = meshkernel_with_mesh2d(2, 2)

    node_index = mk.insert_node_mesh2d(1.5, 0.5)
    edge_index = mk.insert_edge_mesh2d(3, node_index)

    mesh2d = mk.get_mesh2d()

    assert node_index == 4
    assert mesh2d.node_x.size == 5
    assert edge_index == 4
    assert mesh2d.edge_x.size == 5


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
def test_delete_node_mesh2d(
    meshkernel_with_mesh2d: MeshKernel,
    node_index: int,
    deleted_x: float,
    deleted_y: float,
):
    """Test `delete_node_mesh2d` by deleting a node from a 3x3 Mesh2d.

    6---7---8
    |   |   |
    3---4---5
    |   |   |
    0---1---2

    """
    mk = meshkernel_with_mesh2d(3, 3)

    mk.delete_node_mesh2d(node_index)

    mesh2d = mk.get_mesh2d()

    assert mesh2d.node_x.size == 8

    for x, y in zip(mesh2d.node_x, mesh2d.node_y):
        assert x != deleted_x or y != deleted_y


def test_delete_node_mesh2d_invalid_node_index(meshkernel_with_mesh2d: MeshKernel):
    """Test `delete_node_mesh2d` by passing a negative `node_index`."""

    mk = meshkernel_with_mesh2d(2, 2)

    with pytest.raises(InputError):
        mk.delete_node_mesh2d(-1)


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
def test_move_node_mesh2d(
    meshkernel_with_mesh2d: MeshKernel, node_index: int, moved_x: float, moved_y: float
):
    """Test to move a node in a simple Mesh2d to new location.

    6---7---8
    |   |   |
    3---4---5
    |   |   |
    0---1---2

    """

    mk = meshkernel_with_mesh2d(3, 3)

    x_coordinates = np.array([5.0], dtype=np.double)
    y_coordinates = np.array([7.0], dtype=np.double)

    geometry_list = GeometryList(x_coordinates, y_coordinates)
    mk.move_node_mesh2d(geometry_list, node_index)

    mesh2d = mk.get_mesh2d()

    assert mesh2d.node_x[node_index] == 5.0
    assert mesh2d.node_y[node_index] == 7.0

    for x, y in zip(mesh2d.node_x, mesh2d.node_y):
        assert x != moved_x or y != moved_y


def test_move_node_mesh2d_invalid_node_index(meshkernel_with_mesh2d: MeshKernel):
    """Test `move_node_mesh2d` by passing a negative `node_index`."""

    mk = meshkernel_with_mesh2d(2, 2)

    x_coordinates = np.array([5.0], dtype=np.double)
    y_coordinates = np.array([7.0], dtype=np.double)

    geometry_list = GeometryList(x_coordinates, y_coordinates)

    with pytest.raises(InputError):
        mk.move_node_mesh2d(geometry_list, -1)


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
def test_delete_edge_mesh2d(
    meshkernel_with_mesh2d: MeshKernel, delete_x: float, delete_y: float
):
    """Test `delete_edge_mesh2d` by deleting an edge from a 3x3 Mesh2d.

    6---7---8
    |   |   |
    3---4---5
    |   |   |
    0---1---2

    """
    mk = meshkernel_with_mesh2d(3, 3)

    x_coordinate = np.array([delete_x], dtype=np.double)
    y_coordinate = np.array([delete_y], dtype=np.double)
    geometry_list = GeometryList(x_coordinate, y_coordinate)

    mk.delete_edge_mesh2d(geometry_list)

    mesh2d = mk.get_mesh2d()

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
def test_find_edge_mesh2d(
    meshkernel_with_mesh2d: MeshKernel, x: float, y: float, exp_index: int
):
    """Test `find_edge_mesh2d` on a 2x2 Mesh2d.

        (3)
       2---3
    (0)|   |(1)
       0---1
        (2)

    """

    mk = meshkernel_with_mesh2d(2, 2)

    x_coordinate = np.array([x], dtype=np.double)
    y_coordinate = np.array([y], dtype=np.double)
    geometry_list = GeometryList(x_coordinate, y_coordinate)

    edge_index = mk.find_edge_mesh2d(geometry_list)

    assert edge_index == exp_index


cases_get_node_index_mesh2d = [
    (0.0, 0.0, 0),
    (0.4, 0.0, 0),
    (0.0, 0.4, 0),
    (1.0, 0.0, 1),
    (0.6, 0.0, 1),
    (1.0, 0.4, 1),
    (0.0, 1.0, 2),
    (0.4, 1.0, 2),
    (0.0, 0.6, 2),
    (1.0, 1.0, 3),
    (0.6, 1.0, 3),
    (1.0, 0.6, 3),
]


@pytest.mark.parametrize("x, y, exp_index", cases_get_node_index_mesh2d)
def test_get_node_index_mesh2d(
    meshkernel_with_mesh2d: MeshKernel, x: float, y: float, exp_index: int
):
    """Test `find_edge_mesh2d` on a 2x2 Mesh2d.

    2---3
    |   |
    0---1

    """

    mk = meshkernel_with_mesh2d(2, 2)

    x_coordinate = np.array([x], dtype=np.double)
    y_coordinate = np.array([y], dtype=np.double)
    geometry_list = GeometryList(x_coordinate, y_coordinate)

    edge_index = mk.get_node_index_mesh2d(geometry_list, 0.5)

    assert edge_index == exp_index


def test_get_node_index_mesh2d_no_node_in_search_radius(
    meshkernel_with_mesh2d: MeshKernel,
):
    """Test `find_edge_mesh2d` when there is no node within the search radius."""

    mk = meshkernel_with_mesh2d(2, 2)

    x_coordinate = np.array([0.5], dtype=np.double)
    y_coordinate = np.array([0.5], dtype=np.double)
    geometry_list = GeometryList(x_coordinate, y_coordinate)

    with pytest.raises(MeshKernelError):
        mk.get_node_index_mesh2d(geometry_list, 0.4)


cases_delete_mesh2d_small_polygon = [
    (True, DeleteMeshOption.ALL_NODES, 4, 4, 1),
    (True, DeleteMeshOption.ALL_FACE_CIRCUMCENTERS, 16, 24, 9),
    (True, DeleteMeshOption.ALL_COMPLETE_FACES, 4, 4, 1),
    (False, DeleteMeshOption.ALL_NODES, 32, 48, 16),
    (False, DeleteMeshOption.ALL_FACE_CIRCUMCENTERS, 32, 48, 16),
    (False, DeleteMeshOption.ALL_COMPLETE_FACES, 36, 60, 25),
]


@pytest.mark.parametrize(
    "invert_deletion, delete_option, exp_nodes, exp_edges, exp_faces",
    cases_delete_mesh2d_small_polygon,
)
def test_delete_mesh2d_small_polygon(
    meshkernel_with_mesh2d: MeshKernel,
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
    mk = meshkernel_with_mesh2d(6, 6)

    # Polygon around nodes 14, 15, 21 & 20 (through the face circum centers)
    x_coordinates = np.array([1.5, 3.5, 3.5, 1.5, 1.5], dtype=np.double)
    y_coordinates = np.array([1.5, 1.5, 3.5, 3.5, 1.5], dtype=np.double)

    geometry_list = GeometryList(x_coordinates, y_coordinates)

    mk.delete_mesh2d(geometry_list, delete_option, invert_deletion)
    mesh2d = mk.get_mesh2d()

    assert mesh2d.node_x.size == exp_nodes
    assert mesh2d.edge_x.size == exp_edges
    assert mesh2d.face_x.size == exp_faces


cases_delete_mesh2d_empty_polygon = [(False, 0, 0, 0), (True, 25, 40, 16)]


@pytest.mark.parametrize(
    "invert_deletion, exp_nodes, exp_edges, exp_faces",
    cases_delete_mesh2d_empty_polygon,
)
def test_delete_mesh2d_empty_polygon(
    meshkernel_with_mesh2d: MeshKernel,
    invert_deletion: bool,
    exp_nodes: int,
    exp_edges: int,
    exp_faces: int,
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
    mk = meshkernel_with_mesh2d(5, 5)

    x_coordinates = np.empty(0, dtype=np.double)
    y_coordinates = np.empty(0, dtype=np.double)

    geometry_list = GeometryList(x_coordinates, y_coordinates)
    delete_option = DeleteMeshOption.ALL_NODES

    mk.delete_mesh2d(geometry_list, delete_option, invert_deletion)
    mesh2d = mk.get_mesh2d()

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
    """Tests `count_hanging_edges_mesh2d` by counting the hanging edges in a simple Mesh2d
    4*
    |
    3---2---5*
    |   |
    0---1
    """

    mk = MeshKernel(False)

    mesh2d = Mesh2d(node_x, node_y, edge_nodes)

    mk.set_mesh2d(mesh2d)

    result = mk.count_hanging_edges_mesh2d()

    assert result == expected


def test_delete_hanging_edges_mesh2d():
    """Tests `delete_hanging_edges_mesh2d` by deleting 2 hanging edges in a simple Mesh2d
    4*
    |
    3---2---5*
    |   |
    0---1
    """

    mk = MeshKernel(False)

    node_x = np.array([0.0, 1.0, 1.0, 0.0, 0.0, 2.0], dtype=np.double)
    node_y = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 1.0], dtype=np.double)
    edge_nodes = np.array([0, 1, 1, 2, 2, 3, 3, 0, 3, 4, 2, 5], dtype=np.int32)

    mesh2d = Mesh2d(node_x, node_y, edge_nodes)

    mk.set_mesh2d(mesh2d)

    mk.delete_hanging_edges_mesh2d()

    mesh2d = mk.get_mesh2d()

    assert mesh2d.node_x.size == 4
    assert mesh2d.edge_x.size == 4
    assert mesh2d.face_x.size == 1