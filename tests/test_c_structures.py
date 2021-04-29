import numpy as np
import pytest
from numpy.ctypeslib import as_array
from numpy.testing import assert_array_equal

from meshkernel import GeometryList, Mesh2d
from meshkernel.c_structures import CGeometryList, CMesh2d


def test_cmesh2d_from_mesh2d():
    """Tests `from_mesh2d` of the `CMesh2D` class with a simple mesh."""

    # 2---3
    # |   |
    # 0---1
    node_x = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.double)
    node_y = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.double)
    edge_nodes = np.array([0, 1, 1, 3, 3, 2, 2, 0], dtype=np.int32)
    face_nodes = np.array([0, 1, 2, 3], dtype=np.int32)
    nodes_per_face = np.array([4], dtype=np.int32)
    edge_x = np.array([0.5, 1.0, 0.5, 0.0], dtype=np.double)
    edge_y = np.array([0.0, 0.5, 1.0, 0.5], dtype=np.double)
    face_x = np.array([0.5], dtype=np.double)
    face_y = np.array([0.5], dtype=np.double)

    mesh2d = Mesh2d(node_x, node_y, edge_nodes)
    mesh2d.face_nodes = face_nodes
    mesh2d.nodes_per_face = nodes_per_face
    mesh2d.edge_x = edge_x
    mesh2d.edge_y = edge_y
    mesh2d.face_x = face_x
    mesh2d.face_y = face_y

    cmesh2d = CMesh2d.from_mesh2d(mesh2d)

    # Get the numpy arrays from the ctypes object
    cmesh2d_node_x = as_array(cmesh2d.node_x, (4,))
    cmesh2d_node_y = as_array(cmesh2d.node_y, (4,))
    cmesh2d_edge_nodes = as_array(cmesh2d.edge_nodes, (8,))
    cmesh2d_face_nodes = as_array(cmesh2d.face_nodes, (4,))
    cmesh2d_nodes_per_face = as_array(cmesh2d.nodes_per_face, (1,))
    cmesh2d_edge_x = as_array(cmesh2d.edge_x, (4,))
    cmesh2d_edge_y = as_array(cmesh2d.edge_y, (4,))
    cmesh2d_face_x = as_array(cmesh2d.face_x, (1,))
    cmesh2d_face_y = as_array(cmesh2d.face_y, (1,))

    # Assert data is correct
    assert_array_equal(cmesh2d_node_x, node_x)
    assert_array_equal(cmesh2d_node_y, node_y)
    assert_array_equal(cmesh2d_edge_nodes, edge_nodes)
    assert_array_equal(cmesh2d_face_nodes, face_nodes)
    assert_array_equal(cmesh2d_nodes_per_face, nodes_per_face)
    assert_array_equal(cmesh2d_edge_x, edge_x)
    assert_array_equal(cmesh2d_edge_y, edge_y)
    assert_array_equal(cmesh2d_face_x, face_x)
    assert_array_equal(cmesh2d_face_y, face_y)

    assert cmesh2d.num_nodes == 4
    assert cmesh2d.num_edges == 4
    assert cmesh2d.num_faces == 1
    assert cmesh2d.num_face_nodes == 4


def test_cmesh2d_allocate_memory():
    """Tests `allocate_memory` of the `CMesh2D` class."""

    cmesh2d = CMesh2d()
    cmesh2d.num_nodes = 4
    cmesh2d.num_edges = 4
    cmesh2d.num_faces = 1
    cmesh2d.num_face_nodes = 4

    mesh2d = cmesh2d.allocate_memory()

    assert mesh2d.node_x.size == 4
    assert mesh2d.node_y.size == 4
    assert mesh2d.edge_nodes.size == 8
    assert mesh2d.face_nodes.size == 4
    assert mesh2d.nodes_per_face.size == 1
    assert mesh2d.edge_x.size == 4
    assert mesh2d.edge_y.size == 4
    assert mesh2d.face_x.size == 1
    assert mesh2d.face_y.size == 1


def test_cgeometrylist_from_geometrylist():
    """Tests `from_geometrylist` of the `CGeometryList` class."""

    x_coordinates = np.array([0.0, 2.0, 4.0, 6.0, 8.0], dtype=np.double)
    y_coordinates = np.array([1.0, 3.0, 5.0, 7.0, 9.0], dtype=np.double)
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.double)
    geometry_separator = -12.3
    inner_outer_separator = 45.6

    geometrylist = GeometryList(
        x_coordinates, y_coordinates, values, geometry_separator, inner_outer_separator
    )

    cgeometrylist = CGeometryList.from_geometrylist(geometrylist)

    # Get the numpy arrays from the ctypes object
    cgeometrylist_x_coordinates = as_array(cgeometrylist.x_coordinates, (5,))
    cgeometrylist_y_coordinates = as_array(cgeometrylist.y_coordinates, (5,))
    cgeometrylist_values = as_array(cgeometrylist.values, (5,))

    assert_array_equal(cgeometrylist_x_coordinates, x_coordinates)
    assert_array_equal(cgeometrylist_y_coordinates, y_coordinates)
    assert_array_equal(cgeometrylist_values, values)

    assert cgeometrylist.geometry_separator == -12.3
    assert cgeometrylist.inner_outer_separator == 45.6
    assert cgeometrylist.n_coordinates == 5


def test_cgeometrylist_allocate_memory():
    """Tests `allocate_memory` of the `CGeometryList` class."""

    cgeometrylist = CGeometryList()
    cgeometrylist.n_coordinates = 5
    cgeometrylist.geometry_separator = -12.3
    cgeometrylist.inner_outer_separator = 45.6

    geometrylist = cgeometrylist.allocate_memory()

    assert geometrylist.x_coordinates.size == 5
    assert geometrylist.y_coordinates.size == 5
    assert geometrylist.values.size == 5
    assert geometrylist.geometry_separator == -12.3
    assert geometrylist.inner_outer_separator == 45.6


def test_corthogonalizationparameters_from_orthogonalizationparameters():
    pass


def test_cinterpolationparameters_from_interpolationparameters():
    pass


def test_csamplerefineparameters_from_samplerefinementparameters():
    pass
