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
    """Tests `compute_single_contacts` with a 4x4 Mesh2d and a Mesh1d with 3 nodes."""

    mk = MeshKernel()

    mesh2d = Mesh2dFactory.create_rectilinear_mesh(4, 4)

    node_x = np.array([0.0, 1.0, 2.0], dtype=np.double)
    node_y = np.array([-1.0, -1.0, -1.0], dtype=np.double)
    edge_nodes = np.array([0, 1, 1, 2], dtype=np.int32)
    mesh1d = Mesh1d(node_x, node_y, edge_nodes)

    mk.set_mesh2d(mesh2d)
    mk.set_mesh1d(mesh1d)

    compute_nodes = np.array([1, 1, 1], dtype=np.int32)

    polygon_x = np.array([-1.0, 3.0, 3.0, -1.0, -1.0], dtype=np.double)
    polygon_y = np.array([-2.0, -2.0, 3.0, 3.0, -2.0], dtype=np.double)
    polygon = GeometryList(polygon_x, polygon_y)

    mk.compute_single_contacts(compute_nodes, polygon)

    contacts = mk.get_contacts()

    assert contacts.mesh1d_indices.size == 3
    assert contacts.mesh2d_indices.size == 3
