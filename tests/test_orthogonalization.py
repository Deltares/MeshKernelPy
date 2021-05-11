import numpy as np
import pytest
from numpy import ndarray
from numpy.testing import assert_array_equal
from pytest import approx

from meshkernel import (
    GeometryList,
    Mesh2d,
    Mesh2dFactory,
    MeshKernel,
    OrthogonalizationParameters,
    ProjectToLandBoundaryOption,
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
