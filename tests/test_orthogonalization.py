import numpy as np
from numpy.testing import assert_array_equal
from pytest import approx

from meshkernel import (
    GeometryList,
    Mesh2d,
    Mesh2dFactory,
    MeshKernel,
    OrthogonalizationParameters,
)


def test_compute_orthogonalization_mesh2d():
    """Tests `compute_orthogonalization_mesh2d` with a 3x3 Mesh2d with an uncentered middle node.
    6---7---8
    |   |   |
    3---4*--5
    |   |   |
    0---1---2
    """

    mk = MeshKernel()

    node_x = np.array(
        [0.0, 1.0, 2.0, 0.0, 1.3, 2.0, 0.0, 1.0, 2.0],
        dtype=np.double,
    )
    node_y = np.array(
        [0.0, 0.0, 0.0, 1.0, 1.3, 1.0, 2.0, 2.0, 2.0],
        dtype=np.double,
    )
    edge_nodes = np.array(
        [0, 1, 1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 0, 3, 1, 4, 2, 5, 3, 6, 4, 7, 5, 8],
        dtype=np.int32,
    )

    mk.set_mesh2d(Mesh2d(node_x, node_y, edge_nodes))

    polygon_x = np.array([-0.1, 2.1, 2.1, -0.1, -0.1], dtype=np.double)
    polygon_y = np.array([-0.1, -0.1, 2.1, 2.1, -0.1], dtype=np.double)
    polygon = GeometryList(polygon_x, polygon_y)

    land_boundary_x = np.array([0.0, 1.0, 2.0], dtype=np.double)
    land_boundary_y = np.array([0.0, 0.0, 0.0], dtype=np.double)
    land_boundary = GeometryList(land_boundary_x, land_boundary_y)

    mk.compute_orthogonalization_mesh2d(
        0, OrthogonalizationParameters(outer_iterations=10), polygon, land_boundary
    )

    mesh2d = mk.get_mesh2d()

    assert 1.0 <= mesh2d.node_x[4] < 1.3
    assert 1.0 <= mesh2d.node_y[4] < 1.3


def test_get_orthogonality_mesh2d_orthogonal_mesh2d():
    """Tests `get_orthogonality_mesh2d` with an orthogonal 3x3 Mesh2d.
    6---7---8
    |   |   |
    3---4---5
    |   |   |
    0---1---2
    """

    mk = MeshKernel()
    mk.set_mesh2d(Mesh2dFactory.create_rectilinear_mesh(3, 3))

    orthogonality = mk.get_orthogonality_mesh2d()

    assert orthogonality.values.size == 12

    exp_orthogonality = np.array(
        [
            -999.0,
            0.0,
            -999.0,
            -999.0,
            0.0,
            -999.0,
            -999.0,
            -999.0,
            0.0,
            0.0,
            -999.0,
            -999.0,
        ],
        dtype=np.double,
    )

    assert_array_equal(orthogonality.values, exp_orthogonality)


def test_get_orthogonality_mesh2d_not_orthogonal_mesh2d():
    """Tests `get_orthogonality_mesh2d` with a non-orthogonal 3x3 Mesh2d.
    6---7---8
    |   |   |
    3---4*--5
    |   |   |
    0---1---2
    """

    mk = MeshKernel()

    node_x = np.array(
        [0.0, 1.0, 2.0, 0.0, 1.8, 2.0, 0.0, 1.0, 2.0],
        dtype=np.double,
    )
    node_y = np.array(
        [0.0, 0.0, 0.0, 1.0, 1.8, 1.0, 2.0, 2.0, 2.0],
        dtype=np.double,
    )
    edge_nodes = np.array(
        [0, 1, 1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 0, 3, 1, 4, 2, 5, 3, 6, 4, 7, 5, 8],
        dtype=np.int32,
    )

    mk.set_mesh2d(Mesh2d(node_x, node_y, edge_nodes))

    orthogonality = mk.get_orthogonality_mesh2d()

    assert orthogonality.values.size == 12

    assert orthogonality.values[0] == -999.0
    assert orthogonality.values[1] == -999.0
    assert orthogonality.values[2] > 0.0
    assert orthogonality.values[3] > 0.0
    assert orthogonality.values[4] == -999.0
    assert orthogonality.values[5] == -999.0
    assert orthogonality.values[6] == -999.0
    assert orthogonality.values[7] > 0.0
    assert orthogonality.values[8] == -999.0
    assert orthogonality.values[9] == -999.0
    assert orthogonality.values[10] > 0.0
    assert orthogonality.values[11] == -999.0
