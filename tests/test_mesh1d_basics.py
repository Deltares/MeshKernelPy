import numpy as np
import pytest
from mesh2d_factory import Mesh2dFactory
from numpy import ndarray
from numpy.testing import assert_array_equal
import matplotlib.pyplot as plt

from meshkernel import (
    Contacts,
    DeleteMeshOption,
    GeometryList,
    Mesh1d,
    MeshKernel,
    MakeGridParameters,
)


def sort_contacts_by_mesh2d_indices(contacts):
    """Sort the contacts by the mesh2d indices to get consistent results.
    The contacts computed by meshkernel can be in any order
    """
    if len(contacts.mesh1d_indices) == 0 or len(contacts.mesh2d_indices) == 0:
        return contacts

    indices = np.argsort(contacts.mesh2d_indices)
    contacts.mesh1d_indices = contacts.mesh1d_indices[indices]
    contacts.mesh2d_indices = contacts.mesh2d_indices[indices]


def test_mesh1d_set_and_mesh1d_get():
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


def test_mesh1d_add():
    r"""Tests `mesh1d_add`."""
    mk = MeshKernel()

    node_x = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.double)
    node_y = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.double)
    edge_nodes = np.array([0, 1, 1, 2, 2, 3], dtype=np.int32)

    input_mesh1d_1 = Mesh1d(node_x, node_y, edge_nodes)
    mk.mesh1d_set(input_mesh1d_1)

    input_mesh1d_2 = Mesh1d(node_x + 4, node_y, edge_nodes)
    mk.mesh1d_add(input_mesh1d_2)

    output_mesh1d = mk.mesh1d_get()

    assert_array_equal(
        output_mesh1d.node_x,
        np.concatenate(
            (input_mesh1d_1.node_x, input_mesh1d_2.node_x),
            axis=None,
        ),
    )

    assert_array_equal(
        output_mesh1d.node_y,
        np.concatenate(
            (input_mesh1d_1.node_y, input_mesh1d_2.node_y),
            axis=None,
        ),
    )


def test_contacts_set_and_get():
    """Tests `contacts_set` and `contacts_get`."""

    mk = MeshKernel()

    mesh1d_indices = np.array([1, 2, 3, 4], dtype=np.int32)
    mesh2d_indices = np.array([5, 6, 7, 8], dtype=np.int32)
    mk.contacts_set(Contacts(mesh1d_indices, mesh2d_indices))
    contacts = mk.contacts_get()
    assert_array_equal(mesh1d_indices, contacts.mesh1d_indices)
    assert_array_equal(mesh2d_indices, contacts.mesh2d_indices)


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

    mesh2d = Mesh2dFactory.create(5, 5)

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
    projection_factor = 0.0

    mk.contacts_compute_single(node_mask, polygon, projection_factor)

    contacts = mk.contacts_get()
    sort_contacts_by_mesh2d_indices(contacts=contacts)

    assert contacts.mesh1d_indices.size == 5
    assert contacts.mesh2d_indices.size == 5

    assert contacts.mesh1d_indices[0] == 0
    assert contacts.mesh1d_indices[1] == 1
    assert contacts.mesh1d_indices[2] == 2
    assert contacts.mesh1d_indices[3] == 3
    assert contacts.mesh1d_indices[4] == 4

    assert contacts.mesh2d_indices[0] == 0
    assert contacts.mesh2d_indices[1] == 6
    assert contacts.mesh2d_indices[2] == 12
    assert contacts.mesh2d_indices[3] == 18
    assert contacts.mesh2d_indices[4] == 24


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

    mesh2d = Mesh2dFactory.create(5, 5)

    node_x = np.array([0.7, 1.5, 2.6, 3.9, 4.8], dtype=np.double)
    node_y = np.array([0.3, 1.4, 2.6, 3.2, 4.2], dtype=np.double)
    edge_nodes = np.array([0, 1, 1, 2, 2, 3, 3, 4], dtype=np.int32)
    mesh1d = Mesh1d(node_x, node_y, edge_nodes)

    mk.mesh2d_set(mesh2d)
    mk.mesh1d_set(mesh1d)

    node_mask = np.full(node_x.size, True)

    mk.contacts_compute_multiple(node_mask)

    contacts = mk.contacts_get()
    sort_contacts_by_mesh2d_indices(contacts=contacts)

    assert contacts.mesh1d_indices.size == 9
    assert contacts.mesh2d_indices.size == 9

    assert_array_equal(contacts.mesh1d_indices, [0, 0, 1, 1, 2, 3, 3, 3, 4])
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

    mesh2d = Mesh2dFactory.create(5, 5)

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
    sort_contacts_by_mesh2d_indices(contacts=contacts)

    assert contacts.mesh1d_indices.size == 2
    assert contacts.mesh2d_indices.size == 2

    assert contacts.mesh1d_indices[0] == 1
    assert contacts.mesh1d_indices[1] == 3

    assert contacts.mesh2d_indices[0] == 10
    assert contacts.mesh2d_indices[1] == 14


def test_contacts_compute_with_points_node_mask_true():
    """Tests `contacts_compute_with_points` with a 5x5 Mesh2d and a Mesh1d with 5 nodes.
    the node_mask contains only true value

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

    mesh2d = Mesh2dFactory.create(5, 5)

    node_x = np.array([0.5, 1.5, 2.5, 3.5, 4.5], dtype=np.double)
    node_y = np.array([0.5, 1.5, 2.5, 3.5, 4.5], dtype=np.double)
    edge_nodes = np.array([0, 1, 1, 2, 2, 3, 3, 4], dtype=np.int32)
    mesh1d = Mesh1d(node_x, node_y, edge_nodes)

    mk.mesh2d_set(mesh2d=mesh2d)
    mk.mesh1d_set(mesh1d=mesh1d)

    node_mask = np.full(node_x.size, True)

    # Three points in Mesh2d faces 10, 8, 14
    points_x = np.array([0.5, 3.5, 4.5], dtype=np.double)
    points_y = np.array([2.5, 1.5, 2.5], dtype=np.double)
    points = GeometryList(points_x, points_y)

    mk.contacts_compute_with_points(node_mask=node_mask, polygons=points)

    contacts = mk.contacts_get()
    sort_contacts_by_mesh2d_indices(contacts=contacts)

    assert contacts.mesh1d_indices.size == 3
    assert contacts.mesh2d_indices.size == 3

    assert contacts.mesh1d_indices[0] == 2
    assert contacts.mesh1d_indices[1] == 1
    assert contacts.mesh1d_indices[2] == 3

    assert contacts.mesh2d_indices[0] == 8
    assert contacts.mesh2d_indices[1] == 10
    assert contacts.mesh2d_indices[2] == 14


def test_contacts_compute_with_points_node_mask_false():
    """Tests `contacts_compute_with_points` with a 5x5 Mesh2d and a Mesh1d with 5 nodes.
    the node_mask contains only false value

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

    mesh2d = Mesh2dFactory.create(5, 5)

    node_x = np.array([0.5, 1.5, 2.5, 3.5, 4.5], dtype=np.double)
    node_y = np.array([0.5, 1.5, 2.5, 3.5, 4.5], dtype=np.double)
    edge_nodes = np.array([0, 1, 1, 2, 2, 3, 3, 4], dtype=np.int32)
    mesh1d = Mesh1d(node_x, node_y, edge_nodes)

    mk.mesh2d_set(mesh2d)
    mk.mesh1d_set(mesh1d)

    node_mask = np.full(node_x.size, False)

    # Three points in Mesh2d faces 10, 8, 14
    points_x = np.array([0.5, 3.5, 4.5], dtype=np.double)
    points_y = np.array([2.5, 1.5, 2.5], dtype=np.double)
    points = GeometryList(points_x, points_y)

    mk.contacts_compute_with_points(node_mask=node_mask, polygons=points)

    contacts = mk.contacts_get()
    sort_contacts_by_mesh2d_indices(contacts=contacts)

    assert contacts.mesh1d_indices.size == 0
    assert contacts.mesh2d_indices.size == 0


def test_contacts_compute_with_points_node_mask_mixed():
    """Tests `contacts_compute_with_points` with a 5x5 Mesh2d and a Mesh1d with 5 nodes.
    the node_mask contains both true and false value

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

    mesh2d = Mesh2dFactory.create(5, 5)

    node_x = np.array([0.5, 1.5, 2.5, 3.5, 4.5], dtype=np.double)
    node_y = np.array([0.5, 1.5, 2.5, 3.5, 4.5], dtype=np.double)
    edge_nodes = np.array([0, 1, 1, 2, 2, 3, 3, 4], dtype=np.int32)
    mesh1d = Mesh1d(node_x, node_y, edge_nodes)

    mk.mesh2d_set(mesh2d)
    mk.mesh1d_set(mesh1d)

    node_mask = np.full(node_x.size, True)
    node_mask[1] = False

    # Three points in Mesh2d faces 10, 8, 14
    points_x = np.array([0.5, 3.5, 4.5], dtype=np.double)
    points_y = np.array([2.5, 1.5, 2.5], dtype=np.double)
    points = GeometryList(points_x, points_y)

    mk.contacts_compute_with_points(node_mask=node_mask, polygons=points)

    contacts = mk.contacts_get()
    sort_contacts_by_mesh2d_indices(contacts=contacts)

    assert contacts.mesh1d_indices.size == 2
    assert contacts.mesh2d_indices.size == 2

    assert contacts.mesh1d_indices[0] == 2
    assert contacts.mesh1d_indices[1] == 3

    assert contacts.mesh2d_indices[0] == 8
    assert contacts.mesh2d_indices[1] == 14


cases_contacts_compute_boundary = [
    (
        np.array([True, True, True, True, True]),  # node_mask
        np.array([1, 2, 4], dtype=np.int32),  # exp_mesh1d_indices
        np.array([0, 2, 3], dtype=np.int32),  # exp_mesh2d_indices
    ),
    (
        np.array([True, False, False, False, True], dtype=np.int32),  # node_mask
        np.array([4, 4], dtype=np.int32),  # exp_mesh1d_indices
        np.array([2, 3], dtype=np.int32),  # exp_mesh2d_indices
    ),
    (
        np.array([False, False, True, True, True], dtype=np.int32),  # node_mask
        np.array([2, 4], dtype=np.int32),  # exp_mesh1d_indices
        np.array([2, 3], dtype=np.int32),  # exp_mesh2d_indices
    ),
    (
        np.array([True, False, False, True, True], dtype=np.int32),  # node_mask
        np.array([3, 4], dtype=np.int32),  # exp_mesh1d_indices
        np.array([2, 3], dtype=np.int32),  # exp_mesh2d_indices
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

    mesh2d = Mesh2dFactory.create(2, 2)

    node_x = np.array([-1.0, -1.0, -0.5, 0.5, 1.5], dtype=np.double)
    node_y = np.array([-1.0, 1.5, 2.5, 3.0, 3.0], dtype=np.double)
    edge_nodes = np.array([0, 1, 1, 2, 2, 3, 3, 4], dtype=np.int32)
    mesh1d = Mesh1d(node_x, node_y, edge_nodes)

    mk.mesh2d_set(mesh2d)
    mk.mesh1d_set(mesh1d)

    polygon_x = np.array([-1.1, 3.1, 3.1, -1.1, -1.1], dtype=np.double)
    polygon_y = np.array([-0.1, -0.1, 3.1, 3.1, -0.1], dtype=np.double)
    polygon = GeometryList(polygon_x, polygon_y)

    mk.contacts_compute_boundary(node_mask, 2.0, polygon)

    contacts = mk.contacts_get()
    sort_contacts_by_mesh2d_indices(contacts=contacts)

    assert_array_equal(contacts.mesh1d_indices, exp_mesh1d_indices)
    assert_array_equal(contacts.mesh2d_indices, exp_mesh2d_indices)


def test_contacts_compute_boundary_with_no_polygon():
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

    mesh2d = Mesh2dFactory.create(2, 2)

    node_x = np.array([-1.0, -1.0, -0.5, 0.5, 1.5], dtype=np.double)
    node_y = np.array([0.5, 1.5, 2.5, 3.0, 3.0], dtype=np.double)
    edge_nodes = np.array([0, 1, 1, 2, 2, 3, 3, 4], dtype=np.int32)
    mesh1d = Mesh1d(node_x, node_y, edge_nodes)

    mk.mesh2d_set(mesh2d)
    mk.mesh1d_set(mesh1d)

    node_mask = np.array([True, True, True, True, True])
    mk.contacts_compute_boundary(node_mask, 2.0)

    contacts = mk.contacts_get()

    sort_contacts_by_mesh2d_indices(contacts=contacts)

    exp_mesh1d_indices = np.array([0, 2, 4], dtype=np.int32)
    exp_mesh2d_indices = np.array([0, 2, 3], dtype=np.int32)

    assert_array_equal(contacts.mesh1d_indices, exp_mesh1d_indices)
    assert_array_equal(contacts.mesh2d_indices, exp_mesh2d_indices)


def test_contacts_compute_with_points_after_deletion(
    meshkernel_with_mesh2d: MeshKernel,
):
    """Tests contacts_compute_with_points and mesh2d_delete to ensure the correct indices are retrieved
    in the gap-free array.
    """

    mk = MeshKernel()

    mesh2d = Mesh2dFactory.create(3, 3)

    node_x = np.array([0.75, 2.1], dtype=np.double)
    node_y = np.array([0.75, 2.1], dtype=np.double)
    edge_nodes = np.array([0, 1], dtype=np.int32)
    mesh1d = Mesh1d(node_x, node_y, edge_nodes)

    mk.mesh1d_set(mesh1d)
    mk.mesh2d_set(mesh2d)

    node_mask = np.full(node_x.size, True)

    # The points indicating the faces to connect
    points_x = np.array([0.75, 2.1], dtype=np.double)
    points_y = np.array([0.75, 2.1], dtype=np.double)
    points = GeometryList(points_x, points_y)

    mk.contacts_compute_with_points(node_mask=node_mask, polygons=points)

    contacts = mk.contacts_get()
    sort_contacts_by_mesh2d_indices(contacts=contacts)

    assert contacts.mesh1d_indices.size == 2
    assert contacts.mesh2d_indices.size == 2

    assert contacts.mesh1d_indices[0] == 0
    assert contacts.mesh1d_indices[1] == 1

    assert contacts.mesh2d_indices[0] == 0
    assert contacts.mesh2d_indices[1] == 8

    x_coordinates = np.array([-1.0, 1.5, 1.5, -1.0, -1.0])
    y_coordinates = np.array([1.5, 1.5, 3.5, 3.5, 1.5])

    polygon = GeometryList(x_coordinates=x_coordinates, y_coordinates=y_coordinates)
    mk.mesh2d_delete(
        geometry_list=polygon,
        delete_option=DeleteMeshOption.INSIDE_NOT_INTERSECTED,
        invert_deletion=False,
    )

    mk.contacts_compute_with_points(node_mask=node_mask, polygons=points)

    contacts = mk.contacts_get()
    sort_contacts_by_mesh2d_indices(contacts=contacts)

    assert contacts.mesh1d_indices.size == 2
    assert contacts.mesh2d_indices.size == 2

    assert contacts.mesh1d_indices[0] == 0
    assert contacts.mesh1d_indices[1] == 1

    assert contacts.mesh2d_indices[0] == 0
    assert contacts.mesh2d_indices[1] == 7


def get_circle_gl(r, detail=100):
    t = np.r_[np.linspace(0, 2 * np.pi, detail), 0]
    polygon = GeometryList(np.cos(t) * r, np.sin(t) * r)
    return polygon


def test_contacts_compute_single_circle():
    # Define line (spiral)
    theta = np.arange(0.1, 20, 0.01)

    y = np.sin(theta) * theta
    x = np.cos(theta) * theta

    dists = np.r_[0.0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
    dists = dists[np.arange(0, len(dists), 20)]

    # Create Mesh1d
    mki = MeshKernel()

    # def mesh1d_add_branch(mki, branch)
    mesh1d_node_x = np.array(
        [
            0.09950042,
            0.28660095,
            0.43879128,
            0.53538953,
            0.55944897,
            0.49895573,
            0.34774848,
            0.1061058,
            -0.21903564,
            -0.61425018,
            -1.06017682,
            -1.53243485,
            -2.00285904,
            -2.44099478,
            -2.81577868,
            -3.09731897,
            -3.25868324,
            -3.27759841,
            -3.13797012,
            -2.83113599,
            -2.35677818,
            -1.72343644,
            -0.9485811,
            -0.05822672,
            0.91391061,
            1.92768649,
            2.93818398,
            3.89768376,
            4.75786287,
            5.47212274,
            5.99793747,
            6.29910941,
            6.34781957,
            6.12636709,
            5.62850319,
            4.86028133,
            3.84036588,
            2.59976488,
            1.18097874,
            -0.36341679,
            -1.97270765,
            -3.58042781,
            -5.11710117,
            -6.51322582,
            -7.70237336,
            -8.62426658,
            -9.22769553,
            -9.47313548,
            -9.33493933,
            -8.80299241,
            -7.88373862,
            -6.6005121,
            -4.99313774,
            -3.11679531,
            -1.04017448,
            1.1570199,
            3.38712238,
            5.55800473,
            7.57687716,
            9.35423652,
            10.80779395,
            11.8662112,
            12.47247849,
            12.58677787,
            12.18869449,
            11.27866268,
            9.878564,
            8.03142895,
            5.80023126,
            3.26580248,
            0.52393323,
            -2.31823644,
            -5.14640187,
            -7.84369048,
            -10.29548549,
            -12.39427773,
            -14.04434094,
            -15.16602867,
            -15.69950221,
            -15.60771894,
            -14.87853808,
            -13.52583451,
            -11.58955145,
            -9.13466558,
            -6.24908366,
            -3.04053512,
            0.36743138,
            3.84019936,
            7.23740149,
            10.41859212,
            13.24904611,
            15.6054449,
            17.38121053,
            18.49125831,
            18.87595858,
            18.50412697,
            17.3748994,
            15.51839191,
            12.99509423,
            9.89399732,
        ]
    )
    mesh1d_node_y = np.array(
        [
            9.98334166e-03,
            8.86560620e-02,
            2.39712769e-01,
            4.50952381e-01,
            7.04994219e-01,
            9.80328096e-01,
            1.25262564e00,
            1.49624248e00,
            1.68583018e00,
            1.79797017e00,
            1.81273967e00,
            1.71512199e00,
            1.49618036e00,
            1.15392568e00,
            6.93823055e-01,
            1.28900054e-01,
            -5.20560791e-01,
            -1.22774130e00,
            -1.96039372e00,
            -2.68228802e00,
            -3.35493616e00,
            -3.93951353e00,
            -4.39888553e00,
            -4.69963931e00,
            -4.81401780e00,
            -4.72165488e00,
            -4.41101744e00,
            -3.88047179e00,
            -3.13890759e00,
            -2.20587232e00,
            -1.11119128e00,
            1.05927573e-01,
            1.39827992e00,
            2.71249447e00,
            3.99123437e00,
            5.17568018e00,
            6.20818733e00,
            7.03499983e00,
            7.60889540e00,
            7.89163660e00,
            7.85610747e00,
            7.48802622e00,
            6.78714046e00,
            5.76783230e00,
            4.45908562e00,
            2.90379510e00,
            1.15742614e00,
            -7.13935644e-01,
            -2.63607808e00,
            -4.52960535e00,
            -6.31321355e00,
            -7.90716384e00,
            -9.23680548e00,
            -1.02359947e01,
            -1.08502552e01,
            -1.10395337e01,
            -1.07804175e01,
            -1.00677000e01,
            -8.91520793e00,
            -7.35583164e00,
            -5.44073432e00,
            -3.23775103e00,
            -8.29023717e-01,
            1.69204693e00,
            4.22442026e00,
            6.66346518e00,
            8.90527784e00,
            1.08510898e01,
            1.24115800e01,
            1.35109043e01,
            1.40902624e01,
            1.41108391e01,
            1.35559783e01,
            1.24324784e01,
            1.07709321e01,
            8.62507273e00,
            6.07013077e00,
            3.20024597e00,
            1.25021985e-01,
            -3.03465144e00,
            -6.15134982e00,
            -9.09625202e00,
            -1.17444581e01,
            -1.39802677e01,
            -1.57021958e01,
            -1.68275116e01,
            -1.72960977e01,
            -1.70734551e01,
            -1.61527094e01,
            -1.45555123e01,
            -1.23317792e01,
            -9.55824719e00,
            -6.33589144e00,
            -2.78628178e00,
            9.52988800e-01,
            4.73363337e00,
            8.40255146e00,
            1.18080275e01,
            1.48059963e01,
            1.72661176e01,
        ]
    )
    nlinks = len(mesh1d_node_y) - 1
    new_edge_nodes = np.stack([np.arange(nlinks), np.arange(nlinks) + 1], axis=1)
    m1d = Mesh1d(
        node_x=mesh1d_node_x.astype(np.float64),
        node_y=mesh1d_node_y.astype(np.float64),
        edge_nodes=new_edge_nodes.ravel().astype(np.int32),
    )
    mki.mesh1d_set(m1d)

    # Add Mesh2d
    xmin, ymin, xmax, ymax = (-22, -22, 22, 22)
    dx = 2
    dy = 2
    rows = int((ymax - ymin) / dy)
    columns = int((xmax - xmin) / dx)
    params = MakeGridParameters(
        num_columns=columns,
        num_rows=rows,
        origin_x=xmin,
        origin_y=ymin,
        block_size_x=dx,
        block_size_y=dy,
    )
    mki.curvilinear_compute_rectangular_grid(params)
    mki.curvilinear_convert_to_mesh2d()  # convert to ugrid/mesh2d

    # clip mesh inner circle
    mki.mesh2d_delete(
        geometry_list=get_circle_gl(22),
        delete_option=DeleteMeshOption.INSIDE_NOT_INTERSECTED,
        invert_deletion=False,
    )

    # Add links
    mki.contacts_compute_single(
        node_mask=np.full(nlinks + 1, True, dtype=bool),
        polygons=get_circle_gl(19),
        projection_factor=1.0,
    )

    fig, ax = plt.subplots()
    mesh2d_output = mki.mesh2d_get()
    mesh1d_output = mki.mesh1d_get()
    links_output = mki.contacts_get()
    mesh2d_output.plot_edges(ax=ax, color="r")
    mesh1d_output.plot_edges(ax=ax, color="g")
    links_output.plot_edges(
        ax=ax, mesh1d=mesh1d_output, mesh2d=mesh2d_output, color="k"
    )

    contacts = mki.contacts_get()
    network1_link1d2d = np.stack(
        [contacts.mesh1d_indices, contacts.mesh2d_indices], axis=1
    )
    assert network1_link1d2d.shape == (21, 2)
