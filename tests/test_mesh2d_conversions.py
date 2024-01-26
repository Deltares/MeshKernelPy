import numpy as np
from numpy.testing import assert_array_equal

from meshkernel import (
    Mesh2d,
    MeshKernel,
    GeometryList,
)


def test_mesh2d_implicit_int_to_double_conversions():
    """Test implicit conversion from int to double for Mesh2d works
    """
    mk = MeshKernel()

    edge_nodes = np.array([0, 1, 1, 2, 2, 3, 3, 0], dtype=np.int32)
    node_x = np.array([0, 1, 1, 0], dtype=np.int32)
    node_y = np.array([0, 0, 1, 1], dtype=np.int32)

    input_mesh2d = Mesh2d(node_x, node_y, edge_nodes)
    mk.mesh2d_set(input_mesh2d)

    output_mesh2d = mk.mesh2d_get()

    # Test if the input and output differs
    assert_array_equal(output_mesh2d.edge_nodes, input_mesh2d.edge_nodes)
    assert_array_equal(output_mesh2d.node_x, input_mesh2d.node_x)
    assert_array_equal(output_mesh2d.node_y, input_mesh2d.node_y)

    # Test if faces are correctly calculated
    assert_array_equal(output_mesh2d.face_nodes, np.array([0, 1, 2, 3]))
    assert_array_equal(output_mesh2d.nodes_per_face, np.array([4]))
    assert_array_equal(output_mesh2d.face_x, np.array([0.5]))
    assert_array_equal(output_mesh2d.face_y, np.array([0.5]))


def test_mesh2d_implicit_unicode_to_double_conversions():
    """Test implicit conversion from unicode to double for Mesh2d works
    """
    mk = MeshKernel()

    node_x = np.array(['0', '1.0', '1', '0'], dtype=np.unicode)
    node_y = np.array(['0', '0', '1', '1.0'], dtype=np.unicode)
    edge_nodes = np.array(['0', '1', '1', '2', '2', '3', '3', '0'], dtype=np.unicode)

    input_mesh2d = Mesh2d(node_x, node_y, edge_nodes)
    mk.mesh2d_set(input_mesh2d)

    output_mesh2d = mk.mesh2d_get()

    # Test if the input and output differs
    assert_array_equal(output_mesh2d.edge_nodes, input_mesh2d.edge_nodes)
    assert_array_equal(output_mesh2d.node_x, input_mesh2d.node_x)
    assert_array_equal(output_mesh2d.node_y, input_mesh2d.node_y)

    # Test if faces are correctly calculated
    assert_array_equal(output_mesh2d.face_nodes, np.array([0, 1, 2, 3]))
    assert_array_equal(output_mesh2d.nodes_per_face, np.array([4]))
    assert_array_equal(output_mesh2d.face_x, np.array([0.5]))
    assert_array_equal(output_mesh2d.face_y, np.array([0.5]))


def test_geometrylist_implicit_int_to_double_conversions():
    """Test implicit conversion from int to double for GeometryList works
    """

    x_coordinates = np.array([2, 5, 3, 0, 2], dtype=np.int32)
    y_coordinates = np.array([5, 3, 5, 2, 0], dtype=np.int32)
    values = np.array([0, 0, 1, 1, 1], dtype=np.int32)

    geometry_list = GeometryList(x_coordinates=x_coordinates,
                                 y_coordinates=y_coordinates,
                                 values=values)

    x_coordinates_valid = np.array([2, 5, 3, 0, 2], dtype=np.double)
    y_coordinates_valid = np.array([5, 3, 5, 2, 0], dtype=np.double)
    values_valid = np.array([0, 0, 1, 1, 1], dtype=np.double)

    assert_array_equal(geometry_list.x_coordinates, x_coordinates_valid)
    assert_array_equal(geometry_list.y_coordinates, y_coordinates_valid)
    assert_array_equal(geometry_list.values, values_valid)

def test_geometrylist_implicit_unicode_to_double_conversions():
    """Test implicit conversion from unicode to double for GeometryList works
    """

    x_coordinates = np.array(['2', '5.0', '3', '0', '2'], dtype=np.unicode)
    y_coordinates = np.array(['5', '3', '5.0', '2', '0'], dtype=np.unicode)
    values = np.array(['0.0', '0', '1.0', '1', '1'], dtype=np.unicode)

    geometry_list = GeometryList(x_coordinates=x_coordinates,
                                 y_coordinates=y_coordinates,
                                 values=values)

    x_coordinates_valid = np.array([2, 5, 3, 0, 2], dtype=np.double)
    y_coordinates_valid = np.array([5, 3, 5, 2, 0], dtype=np.double)
    values_valid = np.array([0, 0, 1, 1, 1], dtype=np.double)

    assert_array_equal(geometry_list.x_coordinates, x_coordinates_valid)
    assert_array_equal(geometry_list.y_coordinates, y_coordinates_valid)
    assert_array_equal(geometry_list.values, values_valid)
