import math

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from meshkernel import Mesh2d, MeshKernel, MeshKernelError


def test_constructor():
    MeshKernel(False)


def test_deallocate():
    meshlib = MeshKernel(False)
    meshlib.deallocate_state()


def test_set_mesh():
    """Test to set a simple mesh and then get it again with new parameters

    3---2
    |   |
    0---1
    """
    meshlib = MeshKernel(False)

    edge_nodes = np.array(
        [0, 1, 1, 2, 2, 3, 3, 0],
        dtype=int,
    )
    node_x = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.double)
    node_y = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.double)

    input_mesh2d = Mesh2d(edge_nodes, node_x, node_y)
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
