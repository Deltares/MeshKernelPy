import numpy as np
import pytest
from numpy import ndarray
from numpy.testing import assert_array_equal

from meshkernel import GeometryList, Mesh1d, Mesh2dFactory, MeshKernel


def test_mesh1d_get():
    r"""Tests `mesh1d_set` and `mesh1d_get` to set and get a simple mesh.

      1   3
     / \ /
    0   2
    """
    mk = MeshKernel()

    node_x = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.double)
    node_y = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.double)
    edge_nodes = np.array([0, 1, 1, 2, 2, 3], dtype=np.int32)
    input_mesh1d = Mesh1d(node_x, node_y, edge_nodes)

    mk.mesh1d_set(input_mesh1d)

    output_mesh1d = mk.mesh1d_get()

    # Test if the input and output differs
    assert_array_equal(output_mesh1d.edge_nodes, input_mesh1d.edge_nodes)
    assert_array_equal(output_mesh1d.node_x, input_mesh1d.node_x)
    assert_array_equal(output_mesh1d.node_y, input_mesh1d.node_y)


def test_contacts_compute_single():
    """Tests `contacts_compute_single` with a 5x5 Mesh2d and a Mesh1d with 5 nodes.

    30--31--32--33--34--35
    |   |   |   |   |  /|
    24--25--26--27--28--29
    |   |   |   |  /|   |
    18--19--20--21--22--23
    |   |   |  /|   |   |
    12--13--14--15--16--17
    |   |  /|   |   |   |
    6---7---8---9---10--11
    |  /|   |   |   |   |
    0---1---2---3---4---5
    """

    mk = MeshKernel()

    mesh2d = Mesh2dFactory.create_rectilinear_mesh(5, 5)

    node_x = np.array([0.75, 1.75, 2.75, 3.75, 4.75], dtype=np.double)
    node_y = np.array([0.25, 1.25, 2.25, 3.25, 4.25], dtype=np.double)
    edge_nodes = np.array([0, 1, 1, 2, 2, 3, 3, 4], dtype=np.int32)
    mesh1d = Mesh1d(node_x, node_y, edge_nodes)

    mk.mesh2d_set(mesh2d)
    mk.mesh1d_set(mesh1d)

    node_mask = np.full(node_x.size, True)

    polygon_x = np.array([-1.0, 6.0, 6.0, -1.0, -1.0], dtype=np.double)
    polygon_y = np.array([-1.0, -1.0, 6.0, 6.0, -1.0], dtype=np.double)
    polygon = GeometryList(polygon_x, polygon_y)

    mk.contacts_compute_single(node_mask, polygon)

    contacts = mk.contacts_get()

    assert contacts.mesh1d_indices.size == 3
    assert contacts.mesh2d_indices.size == 3

    assert contacts.mesh1d_indices[0] == 1
    assert contacts.mesh1d_indices[1] == 2
    assert contacts.mesh1d_indices[2] == 3

    assert contacts.mesh2d_indices[0] == 6
    assert contacts.mesh2d_indices[1] == 12
    assert contacts.mesh2d_indices[2] == 18


def test_contacts_compute_multiple():
    """Tests `contacts_compute_multiple` with a 5x5 Mesh2d and a Mesh1d with 5 nodes.

    30--31--32--33--34--35
    |   |   |   |   |  /|
    24--25--26--27--28--29
    |   |   |   |  /|   |
    18--19--20--21--22--23
    |   |   |  /|   |   |
    12--13--14--15--16--17
    |   |  /|   |   |   |
    6---7---8---9---10--11
    |  /|   |   |   |   |
    0---1---2---3---4---5
    """

    mk = MeshKernel()

    mesh2d = Mesh2dFactory.create_rectilinear_mesh(5, 5)

    node_x = np.array([0.75, 1.75, 2.75, 3.75, 4.75], dtype=np.double)
    node_y = np.array([0.25, 1.25, 2.25, 3.25, 4.25], dtype=np.double)
    edge_nodes = np.array([0, 1, 1, 2, 2, 3, 3, 4], dtype=np.int32)
    mesh1d = Mesh1d(node_x, node_y, edge_nodes)

    mk.mesh2d_set(mesh2d)
    mk.mesh1d_set(mesh1d)

    node_mask = np.full(node_x.size, True)

    mk.contacts_compute_multiple(node_mask)

    contacts = mk.contacts_get()

    assert contacts.mesh1d_indices.size == 9
    assert contacts.mesh2d_indices.size == 9

    assert_array_equal(contacts.mesh1d_indices, [0, 0, 1, 1, 2, 2, 3, 3, 4])
    assert_array_equal(contacts.mesh2d_indices, [0, 1, 6, 7, 12, 13, 18, 19, 24])


def test_contacts_compute_with_polygons():
    """Tests `contacts_compute_with_polygons` with a 5x5 Mesh2d and a Mesh1d with 5 nodes.

    30--31--32--33--34--35
    |   |   |   |   | / |
    24--25--26--27--28--29
    |   |   |   | / |   |
    18--19--20--21--22--23
    |   |   | / |   |   |
    12--13--14--15--16--17
    |   | / |   |   |   |
    6---7---8---9---10--11
    | / |   |   |   |   |
    0---1---2---3---4---5
    """

    mk = MeshKernel()

    mesh2d = Mesh2dFactory.create_rectilinear_mesh(5, 5)

    node_x = np.array([0.5, 1.5, 2.5, 3.5, 4.5], dtype=np.double)
    node_y = np.array([0.5, 1.5, 2.5, 3.5, 4.5], dtype=np.double)
    edge_nodes = np.array([0, 1, 1, 2, 2, 3, 3, 4], dtype=np.int32)
    mesh1d = Mesh1d(node_x, node_y, edge_nodes)

    mk.mesh2d_set(mesh2d)
    mk.mesh1d_set(mesh1d)

    node_mask = np.full(node_x.size, True)

    # Two polygons around Mesh2d nodes 4, 5, 23, 22 and 12, 13, 31, 30
    separator = -999.0
    polygon_x = np.array(
        [-0.1, 1.1, 1.1, -0.1, -0.1, separator, 3.9, 5.1, 5.1, 3.9, 3.9],
        dtype=np.double,
    )
    polygon_y = np.array(
        [1.9, 1.9, 5.1, 5.1, 1.9, separator, -0.1, -0.1, 3.1, 3.1, -0.1],
        dtype=np.double,
    )
    polygon = GeometryList(polygon_x, polygon_y)

    mk.contacts_compute_with_polygons(node_mask, polygon)

    contacts = mk.contacts_get()

    assert contacts.mesh1d_indices.size == 2
    assert contacts.mesh2d_indices.size == 2

    assert contacts.mesh1d_indices[0] == 1
    assert contacts.mesh1d_indices[1] == 3

    assert contacts.mesh2d_indices[0] == 10
    assert contacts.mesh2d_indices[1] == 14


def test_contacts_compute_with_points():
    """Tests `contacts_compute_with_points` with a 5x5 Mesh2d and a Mesh1d with 5 nodes.

    30--31--32--33--34--35
    |   |   |   |   | / |
    24--25--26--27--28--29
    |   |   |   | / |   |
    18--19--20--21--22--23
    |   |   | / |   |   |
    12--13--14--15--16--17
    |   | / |   |   |   |
    6---7---8---9---10--11
    | / |   |   |   |   |
    0---1---2---3---4---5
    """

    mk = MeshKernel()

    mesh2d = Mesh2dFactory.create_rectilinear_mesh(5, 5)

    node_x = np.array([0.5, 1.5, 2.5, 3.5, 4.5], dtype=np.double)
    node_y = np.array([0.5, 1.5, 2.5, 3.5, 4.5], dtype=np.double)
    edge_nodes = np.array([0, 1, 1, 2, 2, 3, 3, 4], dtype=np.int32)
    mesh1d = Mesh1d(node_x, node_y, edge_nodes)

    mk.mesh2d_set(mesh2d)
    mk.mesh1d_set(mesh1d)

    node_mask = np.full(node_x.size, True)

    # Three points in Mesh2d faces 10, 8, 14
    points_x = np.array([0.5, 3.5, 4.5], dtype=np.double)
    points_y = np.array([2.5, 1.5, 2.5], dtype=np.double)
    points = GeometryList(points_x, points_y)

    mk.contacts_compute_with_points(node_mask, points)

    contacts = mk.contacts_get()

    assert contacts.mesh1d_indices.size == 3
    assert contacts.mesh2d_indices.size == 3

    assert contacts.mesh1d_indices[0] == 1
    assert contacts.mesh1d_indices[1] == 2
    assert contacts.mesh1d_indices[2] == 3

    assert contacts.mesh2d_indices[0] == 10
    assert contacts.mesh2d_indices[1] == 8
    assert contacts.mesh2d_indices[2] == 14


cases_contacts_compute_boundary = [
    (
        np.array([True, True, True, True, True]),  # node_mask
        np.array([0, 2, 4], dtype=np.int32),  # exp_mesh1d_indices
        np.array([0, 2, 3], dtype=np.int32),  # exp_mesh2d_indices
    ),
    (
        np.array([True, False, False, False, True], dtype=np.int32),  # node_mask
        np.array([0, 0, 4], dtype=np.int32),  # exp_mesh1d_indices
        np.array([0, 2, 3], dtype=np.int32),  # exp_mesh2d_indices
    ),
    (
        np.array([False, False, True, True, True], dtype=np.int32),  # node_mask
        np.array([2, 4], dtype=np.int32),  # exp_mesh1d_indices
        np.array([2, 3], dtype=np.int32),  # exp_mesh2d_indices
    ),
    (
        np.array([True, False, False, True, True], dtype=np.int32),  # node_mask
        np.array([0, 3, 4], dtype=np.int32),  # exp_mesh1d_indices
        np.array([0, 2, 3], dtype=np.int32),  # exp_mesh2d_indices
    ),
    (
        np.array([False, False, True, False, False], dtype=np.int32),  # node_mask
        np.array([2], dtype=np.int32),  # exp_mesh1d_indices
        np.array([2], dtype=np.int32),  # exp_mesh2d_indices
    ),
    (
        np.array([False, False, False, False, False], dtype=np.int32),  # node_mask
        np.array([], dtype=np.int32),  # exp_mesh1d_indices
        np.array([], dtype=np.int32),  # exp_mesh2d_indices
    ),
]


@pytest.mark.parametrize(
    "node_mask, exp_mesh1d_indices, exp_mesh2d_indices",
    cases_contacts_compute_boundary,
)
def test_contacts_compute_boundary(
    node_mask: ndarray, exp_mesh1d_indices: ndarray, exp_mesh2d_indices: ndarray
):
    """Tests `contacts_compute_boundary` with a 2x2 Mesh2d and a Mesh1d with 5 nodes.


       ---3---4
     2
    |   6---7---8
    1   |   |   |
    |   3---4---5
    0   |   |   |
        0---1---2
    """

    mk = MeshKernel()

    mesh2d = Mesh2dFactory.create_rectilinear_mesh(2, 2)

    node_x = np.array([-1.0, -1.0, -0.5, 0.5, 1.5], dtype=np.double)
    node_y = np.array([0.5, 1.5, 2.5, 3.0, 3.0], dtype=np.double)
    edge_nodes = np.array([0, 1, 1, 2, 2, 3, 3, 4], dtype=np.int32)
    mesh1d = Mesh1d(node_x, node_y, edge_nodes)

    mk.mesh2d_set(mesh2d)
    mk.mesh1d_set(mesh1d)

    polygon_x = np.array([-1.1, 3.1, 3.1, -1.1, -1.1], dtype=np.double)
    polygon_y = np.array([-0.1, -0.1, 3.1, 3.1, -0.1], dtype=np.double)
    polygon = GeometryList(polygon_x, polygon_y)

    mk.contacts_compute_boundary(node_mask, polygon, 2.0)

    contacts = mk.contacts_get()

    assert_array_equal(contacts.mesh1d_indices, exp_mesh1d_indices)
    assert_array_equal(contacts.mesh2d_indices, exp_mesh2d_indices)
