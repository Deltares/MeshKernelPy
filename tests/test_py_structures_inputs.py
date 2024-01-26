import numpy as np
import pytest
from numpy.testing import assert_array_equal

from meshkernel import (
    Mesh2d,
    MeshKernel,
    GeometryList,
    OrthogonalizationParameters,
    InputError
)


def test_mesh2d_implicit_int_conversions():
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


def test_mesh2d_implicit_unicode_conversions():
    """Test implicit conversions and exceptions for Mesh2d
    """
    node_x = ['1.0', '2.0', '3.0']
    node_y = ['4.0', '5.0', '6.0']
    edge_nodes = ['0', '1', '1', '2']
    face_nodes = ['0', '1', '2']
    nodes_per_face = ['3', '4', '5']
    edge_x = ['1.5', '2.5']
    edge_y = ['4.5', '5.5']
    face_x = ['2.0', '3.0']
    face_y = ['5.0', '6.0']
    edge_faces = ['0', '1']
    face_edges = ['2', '3']

    mesh_2d = Mesh2d(
        node_x=node_x,
        node_y=node_y,
        edge_nodes=edge_nodes,
        face_nodes=face_nodes,
        nodes_per_face=nodes_per_face,
        edge_x=edge_x,
        edge_y=edge_y,
        face_x=face_x,
        face_y=face_y,
        edge_faces=edge_faces,
        face_edges=face_edges
    )

    assert_array_equal(mesh_2d.node_x, np.array([1.0, 2.0, 3.0], dtype=np.double))
    assert_array_equal(mesh_2d.node_y, np.array([4.0, 5.0, 6.0], dtype=np.double))
    assert_array_equal(mesh_2d.edge_nodes, np.array([0, 1, 1, 2], dtype=np.int32))
    assert_array_equal(mesh_2d.face_nodes, np.array([0, 1, 2], dtype=np.int32))
    assert_array_equal(mesh_2d.nodes_per_face, np.array([3, 4, 5], dtype=np.int32))
    assert_array_equal(mesh_2d.edge_x, np.array([1.5, 2.5], dtype=np.double))
    assert_array_equal(mesh_2d.edge_y, np.array([4.5, 5.5], dtype=np.double))
    assert_array_equal(mesh_2d.face_x, np.array([2.0, 3.0], dtype=np.double))
    assert_array_equal(mesh_2d.face_y, np.array([5.0, 6.0], dtype=np.double))
    assert_array_equal(mesh_2d.edge_faces, np.array([0, 1], dtype=np.int32))
    assert_array_equal(mesh_2d.face_edges, np.array([2, 3], dtype=np.int32))

def test_mesh2d_invalid_input():
    with pytest.raises(ValueError):
        Mesh2d(node_x=['1.0', 'invalid', '3.0'])

    with pytest.raises(ValueError):
        Mesh2d(edge_nodes=['0', '1', 'invalid', '2', '3'])

    with pytest.raises(ValueError):
        Mesh2d(face_nodes=['0', '1', '2', 'abc'])

def test_geometrylist_implicit_int_conversions():
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


def test_geometrylist_implicit_unicode_conversions():
    """Test implicit conversions and exceptions for GeometryList
    """

    # Test implicit conversion and valid input
    x_coordinates = ['1.0', '2.0', '3.0']
    y_coordinates = ['4.0', '5.0', '6.0']
    values = ['0.1', '0.2', '0.3']

    geometry_list = GeometryList(
        x_coordinates=x_coordinates,
        y_coordinates=y_coordinates,
        values=values
    )

    assert_array_equal(geometry_list.x_coordinates, np.array([1.0, 2.0, 3.0], dtype=np.double))
    assert_array_equal(geometry_list.y_coordinates, np.array([4.0, 5.0, 6.0], dtype=np.double))
    assert_array_equal(geometry_list.values, np.array([0.1, 0.2, 0.3], dtype=np.double))
    assert geometry_list.geometry_separator == -999.0
    assert geometry_list.inner_outer_separator == -998.0

def test_geometrylist_invalid_input():
    """Test implicit conversions and exceptions for GeometryList
    """

    with pytest.raises(InputError):
        GeometryList(x_coordinates=['1.0', '2.0'], y_coordinates=['4.0'])

    with pytest.raises(InputError):
        GeometryList(x_coordinates=['1.0', '2.0', '3.0'], values=['0.1', '0.2'])

    with pytest.raises(ValueError):
        GeometryList(x_coordinates=['1.0', 'invalid', '3.0'], y_coordinates=['4.0', '5.0', '6.0'])

def test_orthogonalization_parameters_implicit_unicode_conversions():
    """Test implicit conversion from unicode to double for OrthogonalizationParameters works
    """

    # Test implicit conversion from unicode to double
    orthogonalization_parameters = OrthogonalizationParameters(
        outer_iterations='2',
        boundary_iterations='25',
        inner_iterations='25',
        orthogonalization_to_smoothing_factor='0.975',
        orthogonalization_to_smoothing_factor_at_boundary='1.0',
        areal_to_angle_smoothing_factor='1.0'
    )

    assert orthogonalization_parameters.outer_iterations == 2
    assert orthogonalization_parameters.boundary_iterations == 25
    assert orthogonalization_parameters.inner_iterations == 25
    assert orthogonalization_parameters.orthogonalization_to_smoothing_factor == 0.975
    assert orthogonalization_parameters.orthogonalization_to_smoothing_factor_at_boundary == 1.0
    assert orthogonalization_parameters.areal_to_angle_smoothing_factor == 1.0

def test_orthogonalization_parameters_invalid_input():
    """Test orthogonalization parameters invalid inputs works
    """

    # Test exceptions for invalid inputs
    with pytest.raises(ValueError):
        OrthogonalizationParameters(outer_iterations='invalid')

    with pytest.raises(ValueError):
        OrthogonalizationParameters(boundary_iterations='invalid')

    with pytest.raises(ValueError):
        OrthogonalizationParameters(inner_iterations='invalid')

    with pytest.raises(ValueError):
        OrthogonalizationParameters(orthogonalization_to_smoothing_factor='invalid')

    with pytest.raises(ValueError):
        OrthogonalizationParameters(orthogonalization_to_smoothing_factor_at_boundary='invalid')

    with pytest.raises(ValueError):
        OrthogonalizationParameters(areal_to_angle_smoothing_factor='invalid')


