import math

import numpy as np
import pytest
import math

from meshkernel import MeshKernel, MeshKernelError, Mesh2d


def test_constructor():
    MeshKernel(False)


def test_deallocate():
    meshlib = MeshKernel(False)
    meshlib.deallocate_state()


def test_set_mesh():
    meshlib = MeshKernel(False)

    edge_nodes = np.array(
        [0, 1, 1, 2, 2, 3, 3, 0],
        dtype=int,
    )
    face_nodes = np.array([0, 1, 2, 3], dtype=int)
    nodes_per_face = np.array([4], dtype=int)
    node_x = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.double)
    node_y = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.double)
    edge_x = np.array([0.5, 1.0, 0.5, 0.0], dtype=np.double)
    edge_y = np.array([0.0, 0.5, 1.0, 0.5], dtype=np.double)
    face_x = np.array([0.5], dtype=np.double)
    face_y = np.array([0.5], dtype=np.double)

    original_mesh2d = Mesh2d(
        edge_nodes,
        face_nodes,
        nodes_per_face,
        node_x,
        node_y,
        edge_x,
        edge_y,
        face_x,
        face_y,
    )

    meshlib.set_mesh2d(original_mesh2d)
    new_mesh2d = meshlib.get_mesh2d()

    np.testing.assert_array_equal(original_mesh2d.node_x, new_mesh2d.node_x)
