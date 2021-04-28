import numpy as np
import pytest
from numpy.testing import assert_array_equal

from meshkernel import InputError, Mesh2d, Mesh2dFactory


def test_create_rectilinear_mesh_simple():
    """Test create_rectilinear_mesh``by creating a simple 3x3 mesh

    6---7---8
    |   |   |
    3---4---5
    |   |   |
    0---1---2

    """
    mesh2d = Mesh2dFactory.create_rectilinear_mesh(3, 3)

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
    """Test create_rectilinear_mesh``by creating a 3x4 mesh.
    Also set
        - origin_x to -1.0 and spacing_x to 2.0 and
        - origin_y to 1.0 and spacing_y to 3.0.


    8---9---10---11
    |   |   |   |
    4---5---6---7
    |   |   |   |
    0---1---2---3

    """
    mesh2d = Mesh2dFactory.create_rectilinear_mesh(
        3, 4, origin_x=-1.0, origin_y=1.0, spacing_x=2.0, spacing_y=3.0
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
        Mesh2dFactory.create_rectilinear_mesh(3, 3, spacing_x=-1.0)

    with pytest.raises(InputError):
        Mesh2dFactory.create_rectilinear_mesh(3, 3, spacing_y=-1.0)


def test_create_rectilinear_mesh_reject_negative_rows_columns():
    """Tests if `create_rectilinear_mesh` rejects negative spacing."""
    with pytest.raises(InputError):
        Mesh2dFactory.create_rectilinear_mesh(-1, 3)

    with pytest.raises(InputError):
        Mesh2dFactory.create_rectilinear_mesh(3, -1)
