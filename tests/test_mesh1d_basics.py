import numpy as np
import pytest
from numpy import ndarray
from numpy.testing import assert_array_equal
from pytest import approx

from meshkernel import GeometryList, Mesh1d, Mesh2dFactory, MeshKernel


def test_get_mesh1d():
    r"""Tests `set_mesh1d` and `get_mesh1d` to set and get a simple mesh.

      1   3
     / \ /
    0   2
    """
    mk = MeshKernel()

    node_x = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.double)
    node_y = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.double)
    edge_nodes = np.array([0, 1, 1, 2, 2, 3], dtype=np.int32)
    input_mesh1d = Mesh1d(node_x, node_y, edge_nodes)

    mk.set_mesh1d(input_mesh1d)

    output_mesh1d = mk.get_mesh1d()

    # Test if the input and output differs
    assert_array_equal(output_mesh1d.edge_nodes, input_mesh1d.edge_nodes)
    assert_array_equal(output_mesh1d.node_x, input_mesh1d.node_x)
    assert_array_equal(output_mesh1d.node_y, input_mesh1d.node_y)


def test_compute_single_contacts():
    """Tests `compute_single_contacts` with a 6x6 Mesh2d and a Mesh1d with 5 nodes.

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

    mesh2d = Mesh2dFactory.create_rectilinear_mesh(6, 6)

    node_x = np.array([0.75, 1.75, 2.75, 3.75, 4.75], dtype=np.double)
    node_y = np.array([0.25, 1.25, 2.25, 3.25, 4.25], dtype=np.double)
    edge_nodes = np.array([0, 1, 1, 2, 2, 3, 3, 4], dtype=np.int32)
    mesh1d = Mesh1d(node_x, node_y, edge_nodes)

    mk.set_mesh2d(mesh2d)
    mk.set_mesh1d(mesh1d)

    compute_nodes = np.array([1, 1, 1, 1, 1], dtype=np.int32)

    polygon_x = np.array([-1.0, 6.0, 6.0, -1.0, -1.0], dtype=np.double)
    polygon_y = np.array([-1.0, -1.0, 6.0, 6.0, -1.0], dtype=np.double)
    polygon = GeometryList(polygon_x, polygon_y)

    mk.compute_single_contacts(compute_nodes, polygon)

    contacts = mk.get_contacts()

    assert contacts.mesh1d_indices.size == 3
    assert contacts.mesh2d_indices.size == 3

    assert contacts.mesh1d_indices[0] == 1
    assert contacts.mesh1d_indices[1] == 2
    assert contacts.mesh1d_indices[2] == 3

    assert contacts.mesh2d_indices[0] == 6
    assert contacts.mesh2d_indices[1] == 12
    assert contacts.mesh2d_indices[2] == 18


def test_compute_multiple_contacts():
    """Tests `compute_multiple_contacts` with a 6x6 Mesh2d and a Mesh1d with 5 nodes.

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

    mesh2d = Mesh2dFactory.create_rectilinear_mesh(6, 6)

    node_x = np.array([0.75, 1.75, 2.75, 3.75, 4.75], dtype=np.double)
    node_y = np.array([0.25, 1.25, 2.25, 3.25, 4.25], dtype=np.double)
    edge_nodes = np.array([0, 1, 1, 2, 2, 3, 3, 4], dtype=np.int32)
    mesh1d = Mesh1d(node_x, node_y, edge_nodes)

    mk.set_mesh2d(mesh2d)
    mk.set_mesh1d(mesh1d)

    compute_nodes = np.array([1, 1, 1, 1, 1], dtype=np.int32)

    mk.compute_multiple_contacts(compute_nodes)

    contacts = mk.get_contacts()

    assert contacts.mesh1d_indices.size == 4
    assert contacts.mesh2d_indices.size == 4

    assert contacts.mesh1d_indices[0] == 0
    assert contacts.mesh1d_indices[1] == 1
    assert contacts.mesh1d_indices[2] == 2
    assert contacts.mesh1d_indices[3] == 3

    assert contacts.mesh2d_indices[0] == 0
    assert contacts.mesh2d_indices[1] == 6
    assert contacts.mesh2d_indices[2] == 12
    assert contacts.mesh2d_indices[3] == 18


def test_compute_with_polygons_contacts():
    """Tests `compute_single_contacts` with a 6x6 Mesh2d and a Mesh1d with 5 nodes.

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

    mesh2d = Mesh2dFactory.create_rectilinear_mesh(6, 6)

    node_x = np.array([0.5, 1.5, 2.5, 3.5, 4.5], dtype=np.double)
    node_y = np.array([0.5, 1.5, 2.5, 3.5, 4.5], dtype=np.double)
    edge_nodes = np.array([0, 1, 1, 2, 2, 3, 3, 4], dtype=np.int32)
    mesh1d = Mesh1d(node_x, node_y, edge_nodes)

    mk.set_mesh2d(mesh2d)
    mk.set_mesh1d(mesh1d)

    compute_nodes = np.array([1, 1, 1, 1, 1], dtype=np.int32)

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

    mk.compute_with_polygons_contacts(compute_nodes, polygon)

    contacts = mk.get_contacts()

    assert contacts.mesh1d_indices.size == 2
    assert contacts.mesh2d_indices.size == 2

    assert contacts.mesh1d_indices[0] == 1
    assert contacts.mesh1d_indices[1] == 3

    assert contacts.mesh2d_indices[0] == 10
    assert contacts.mesh2d_indices[1] == 14


def test_compute_with_points_contacts():
    """Tests `compute_with_points_contacts` with a 6x6 Mesh2d and a Mesh1d with 5 nodes.

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

    mesh2d = Mesh2dFactory.create_rectilinear_mesh(6, 6)

    node_x = np.array([0.5, 1.5, 2.5, 3.5, 4.5], dtype=np.double)
    node_y = np.array([0.5, 1.5, 2.5, 3.5, 4.5], dtype=np.double)
    edge_nodes = np.array([0, 1, 1, 2, 2, 3, 3, 4], dtype=np.int32)
    mesh1d = Mesh1d(node_x, node_y, edge_nodes)

    mk.set_mesh2d(mesh2d)
    mk.set_mesh1d(mesh1d)

    compute_nodes = np.array([1, 1, 1, 1, 1], dtype=np.int32)

    # Three points in Mesh2d faces 10, 8, 14
    points_x = np.array([0.5, 3.5, 4.5], dtype=np.double)
    points_y = np.array([2.5, 1.5, 2.5], dtype=np.double)
    points = GeometryList(points_x, points_y)

    mk.compute_with_points_contacts(compute_nodes, points)

    contacts = mk.get_contacts()

    assert contacts.mesh1d_indices.size == 3
    assert contacts.mesh2d_indices.size == 3

    assert contacts.mesh1d_indices[0] == 1
    assert contacts.mesh1d_indices[1] == 2
    assert contacts.mesh1d_indices[2] == 3

    assert contacts.mesh2d_indices[0] == 10
    assert contacts.mesh2d_indices[1] == 8
    assert contacts.mesh2d_indices[2] == 14


def test_compute_boundary_contacts():
    """Tests `compute_boundary_contacts` with a 3x3 Mesh2d and a Mesh1d with 5 nodes.


    2-----3---4
    |
    |   6---7---8
    1   |   |   |
    |   3---4---5
    0   |   |   |
        0---1---2
    """

    mk = MeshKernel()

    mesh2d = Mesh2dFactory.create_rectilinear_mesh(3, 3)

    node_x = np.array([-1.0, -1.0, -1.0, 0.5, 1.5], dtype=np.double)
    node_y = np.array([0.5, 1.5, 3.0, 3.0, 3.0], dtype=np.double)
    edge_nodes = np.array([0, 1, 1, 2, 2, 3, 3, 4], dtype=np.int32)
    mesh1d = Mesh1d(node_x, node_y, edge_nodes)

    mk.set_mesh2d(mesh2d)
    mk.set_mesh1d(mesh1d)

    compute_nodes = np.array([1, 1, 1, 1, 1], dtype=np.int32)

    polygon_x = np.array([-1.1, 3.1, 3.1, -1.1, -1.1], dtype=np.double)
    polygon_y = np.array([-0.1, -0.1, 3.1, 3.1, -0.1], dtype=np.double)
    polygon = GeometryList(polygon_x, polygon_y)

    mk.compute_boundary_contacts(compute_nodes, polygon, 2.0)

    contacts = mk.get_contacts()

    assert contacts.mesh1d_indices.size == 3
    assert contacts.mesh2d_indices.size == 3

    assert contacts.mesh1d_indices[0] == 0
    assert contacts.mesh1d_indices[1] == 1
    assert contacts.mesh1d_indices[2] == 4

    assert contacts.mesh2d_indices[0] == 0
    assert contacts.mesh2d_indices[1] == 2
    assert contacts.mesh2d_indices[2] == 3
