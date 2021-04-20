import math

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from meshkernel import (
    DeleteMeshOption,
    GeometryList,
    Mesh2d,
    Mesh2dFactory,
    MeshKernel,
    MeshKernelError,
)


def test_constructor():
    """Test if the constructor works"""
    MeshKernel(False)


def test_different_id():
    """Test if the meshkernelid of two instances differs"""
    meshlib_1 = MeshKernel(False)
    meshlib_2 = MeshKernel(False)

    assert meshlib_1._meshkernelid != meshlib_2._meshkernelid


def test_set_mesh():
    """Test to set a simple mesh and then get it again with new parameters

    3---2
    |   |
    0---1
    """
    meshlib = MeshKernel(False)

    edge_nodes = np.array(
        [0, 1, 1, 2, 2, 3, 3, 0],
        dtype=np.int32,
    )
    node_x = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.double)
    node_y = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.double)

    input_mesh2d = Mesh2d(node_x, node_y, edge_nodes)
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


def test_delete_mesh2d():
    """Test `delete_mesh2d` by deleting a polygon from a 5x5 mesh2d.

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
    mesh2d = Mesh2dFactory.create_rectilinear_mesh(5, 5)
    meshkernel = MeshKernel(False)

    meshkernel.set_mesh2d(mesh2d)

    # Polygon for indices 6, 8, 18, 16
    x_coordinates = np.array([1.0, 3.0, 3.0, 1.0], dtype=np.double)
    y_coordinates = np.array([1.0, 1.0, 3.0, 3.0], dtype=np.double)

    geometry_list = GeometryList(x_coordinates, y_coordinates)
    delete_option = DeleteMeshOption.ALLNODES
    invert_deletion = False

    meshkernel.delete_mesh2d(geometry_list, delete_option, invert_deletion)
    mesh2d = meshkernel.get_mesh2d()

    # assert mesh2d.node_x.size == 16
    # assert mesh2d.edge_x.size == 16
    # assert mesh2d.face_x.size == 1
