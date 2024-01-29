import numpy as np
import pytest
from numpy.testing import assert_array_equal

from meshkernel import (
    Contacts,
    CurvilinearGrid,
    CurvilinearParameters,
    GeometryList,
    GriddedSamples,
    InputError,
    MakeGridParameters,
    Mesh1d,
    Mesh2d,
    MeshKernel,
    MeshRefinementParameters,
    OrthogonalizationParameters,
    RefinementType,
    SplinesToCurvilinearParameters,
)


def test_mesh2d_implicit_int_conversions():
    """Test implicit conversion from int to double for Mesh2d works"""
    mk = MeshKernel()

    edge_nodes = np.array([0, 1, 1, 2, 2, 3, 3, 0], dtype=int)
    node_x = np.array([0, 1, 1, 0], dtype=int)
    node_y = np.array([0, 0, 1, 1], dtype=int)

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


def test_mesh2d_implicit_string_conversions():
    """Test implicit conversions from string to doubles works for Mesh2d"""
    node_x = ["1.0", "2.0", "3.0"]
    node_y = ["4.0", "5.0", "6.0"]
    edge_nodes = ["0", "1", "1", "2"]
    face_nodes = ["0", "1", "2"]
    nodes_per_face = ["3", "4", "5"]
    edge_x = ["1.5", "2.5"]
    edge_y = ["4.5", "5.5"]
    face_x = ["2.0", "3.0"]
    face_y = ["5.0", "6.0"]
    edge_faces = ["0", "1"]
    face_edges = ["2", "3"]

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
        face_edges=face_edges,
    )

    assert_array_equal(mesh_2d.node_x, np.array([1.0, 2.0, 3.0], dtype=np.double))
    assert_array_equal(mesh_2d.node_y, np.array([4.0, 5.0, 6.0], dtype=np.double))
    assert_array_equal(mesh_2d.edge_nodes, np.array([0, 1, 1, 2], dtype=int))
    assert_array_equal(mesh_2d.face_nodes, np.array([0, 1, 2], dtype=int))
    assert_array_equal(mesh_2d.nodes_per_face, np.array([3, 4, 5], dtype=int))
    assert_array_equal(mesh_2d.edge_x, np.array([1.5, 2.5], dtype=np.double))
    assert_array_equal(mesh_2d.edge_y, np.array([4.5, 5.5], dtype=np.double))
    assert_array_equal(mesh_2d.face_x, np.array([2.0, 3.0], dtype=np.double))
    assert_array_equal(mesh_2d.face_y, np.array([5.0, 6.0], dtype=np.double))
    assert_array_equal(mesh_2d.edge_faces, np.array([0, 1], dtype=int))
    assert_array_equal(mesh_2d.face_edges, np.array([2, 3], dtype=int))


def test_mesh2d_invalid_input():
    """Test exceptions due to malformed input for Mesh2d"""
    with pytest.raises(ValueError):
        Mesh2d(node_x=["1.0", "invalid", "3.0"])

    with pytest.raises(ValueError):
        Mesh2d(edge_nodes=["0", "1", "invalid", "2", "3"])

    with pytest.raises(ValueError):
        Mesh2d(face_nodes=["0", "1", "2", "abc"])


def test_geometrylist_implicit_int_conversions():
    """Test implicit conversion from int to double for GeometryList works"""

    x_coordinates = np.array([2, 5, 3, 0, 2], dtype=int)
    y_coordinates = np.array([5, 3, 5, 2, 0], dtype=int)
    values = np.array([0, 0, 1, 1, 1], dtype=int)

    geometry_list = GeometryList(
        x_coordinates=x_coordinates, y_coordinates=y_coordinates, values=values
    )

    x_coordinates_valid = np.array([2, 5, 3, 0, 2], dtype=np.double)
    y_coordinates_valid = np.array([5, 3, 5, 2, 0], dtype=np.double)
    values_valid = np.array([0, 0, 1, 1, 1], dtype=np.double)

    assert_array_equal(geometry_list.x_coordinates, x_coordinates_valid)
    assert_array_equal(geometry_list.y_coordinates, y_coordinates_valid)
    assert_array_equal(geometry_list.values, values_valid)


def test_geometrylist_implicit_string_conversions():
    """Test implicit conversions from string to doubles works for GeometryList"""

    x_coordinates = ["1.0", "2.0", "3.0"]
    y_coordinates = ["4.0", "5.0", "6.0"]
    values = ["0.1", "0.2", "0.3"]

    geometry_list = GeometryList(
        x_coordinates=x_coordinates, y_coordinates=y_coordinates, values=values
    )

    assert_array_equal(
        geometry_list.x_coordinates, np.array([1.0, 2.0, 3.0], dtype=np.double)
    )
    assert_array_equal(
        geometry_list.y_coordinates, np.array([4.0, 5.0, 6.0], dtype=np.double)
    )
    assert_array_equal(geometry_list.values, np.array([0.1, 0.2, 0.3], dtype=np.double))
    assert geometry_list.geometry_separator == -999.0
    assert geometry_list.inner_outer_separator == -998.0


def test_geometrylist_invalid_input():
    """Test exceptions due to malformed input for GeometryList"""

    with pytest.raises(InputError):
        GeometryList(x_coordinates=["1.0", "2.0"], y_coordinates=["4.0"])

    with pytest.raises(InputError):
        GeometryList(x_coordinates=["1.0", "2.0", "3.0"], values=["0.1", "0.2"])

    with pytest.raises(ValueError):
        GeometryList(
            x_coordinates=["1.0", "invalid", "3.0"], y_coordinates=["4.0", "5.0", "6.0"]
        )


def test_orthogonalization_parameters_implicit_string_conversions():
    """Test implicit conversion from string to double for OrthogonalizationParameters works"""

    orthogonalization_parameters = OrthogonalizationParameters(
        outer_iterations="2",
        boundary_iterations="25",
        inner_iterations="25",
        orthogonalization_to_smoothing_factor="0.975",
        orthogonalization_to_smoothing_factor_at_boundary="1.0",
        areal_to_angle_smoothing_factor="1.0",
    )

    assert orthogonalization_parameters.outer_iterations == 2
    assert orthogonalization_parameters.boundary_iterations == 25
    assert orthogonalization_parameters.inner_iterations == 25
    assert orthogonalization_parameters.orthogonalization_to_smoothing_factor == 0.975
    assert (
        orthogonalization_parameters.orthogonalization_to_smoothing_factor_at_boundary
        == 1.0
    )
    assert orthogonalization_parameters.areal_to_angle_smoothing_factor == 1.0


def test_orthogonalization_parameters_invalid_input():
    """Test exceptions due to malformed input for OrthogonalizationParameters"""

    # Test exceptions for invalid inputs
    with pytest.raises(ValueError):
        OrthogonalizationParameters(outer_iterations="invalid")

    with pytest.raises(ValueError):
        OrthogonalizationParameters(boundary_iterations="invalid")

    with pytest.raises(ValueError):
        OrthogonalizationParameters(inner_iterations="invalid")

    with pytest.raises(ValueError):
        OrthogonalizationParameters(orthogonalization_to_smoothing_factor="invalid")

    with pytest.raises(ValueError):
        OrthogonalizationParameters(
            orthogonalization_to_smoothing_factor_at_boundary="invalid"
        )

    with pytest.raises(ValueError):
        OrthogonalizationParameters(areal_to_angle_smoothing_factor="invalid")


def test_curvilinear_grid_implicit_string_conversions():
    """Test implicit conversion from string to double for CurvilinearGrid works"""

    # Test valid input
    node_x = ["1.0", "2.0", "3.0"]
    node_y = ["4.0", "5.0", "6.0"]
    num_m = "3"
    num_n = "3"

    curvilinear_grid = CurvilinearGrid(
        node_x=node_x, node_y=node_y, num_m=num_m, num_n=num_n
    )

    assert_array_equal(
        curvilinear_grid.node_x, np.array([1.0, 2.0, 3.0], dtype=np.double)
    )
    assert_array_equal(
        curvilinear_grid.node_y, np.array([4.0, 5.0, 6.0], dtype=np.double)
    )
    assert curvilinear_grid.num_m == 3
    assert curvilinear_grid.num_n == 3


def test_curvilinear_grid_invalid_input():
    """Test exceptions due to malformed input for CurvilinearGrid"""
    with pytest.raises(ValueError):
        CurvilinearGrid(
            node_x=["1.0", "invalid", "3.0"],
            node_y=["4.0", "5.0", "6.0"],
            num_m="3",
            num_n="3",
        )

    with pytest.raises(ValueError):
        CurvilinearGrid(
            node_x=["1.0", "2.0", "3.0"],
            node_y=["4.0", "5.0", "6.0"],
            num_m="3",
            num_n="abc",
        )


def test_curvilinear_parameters_implicit_string_conversions():
    """Test implicit conversion from string to double for CurvilinearParameters works"""
    # Test valid input
    curvilinear_parameters = CurvilinearParameters(
        m_refinement="2000",
        n_refinement="40",
        smoothing_iterations="10",
        smoothing_parameter="0.5",
        attraction_parameter="0.0",
    )

    assert curvilinear_parameters.m_refinement == 2000
    assert curvilinear_parameters.n_refinement == 40
    assert curvilinear_parameters.smoothing_iterations == 10
    assert curvilinear_parameters.smoothing_parameter == 0.5
    assert curvilinear_parameters.attraction_parameter == 0.0


def test_curvilinear_parameters_invalid_input():
    """Test exceptions due to malformed input for CurvilinearParameters"""
    with pytest.raises(ValueError):
        CurvilinearParameters(m_refinement="abc")

    with pytest.raises(ValueError):
        CurvilinearParameters(n_refinement="2.5")

    with pytest.raises(ValueError):
        CurvilinearParameters(smoothing_iterations="xyz")

    with pytest.raises(ValueError):
        CurvilinearParameters(smoothing_parameter="invalid")


def test_splines_to_curvilinear_parameters_implicit_string_conversions():
    """Test implicit conversion from string to double for SplinesToCurvilinearParameters works"""

    splines_to_curvilinear_parameters = SplinesToCurvilinearParameters(
        aspect_ratio="0.1",
        aspect_ratio_grow_factor="1.1",
        average_width="500.0",
        curvature_adapted_grid_spacing="1",
        grow_grid_outside="0",
        maximum_num_faces_in_uniform_part="5",
        nodes_on_top_of_each_other_tolerance="0.0001",
        min_cosine_crossing_angles="0.95",
        check_front_collisions="0",
        remove_skinny_triangles="1",
    )

    assert splines_to_curvilinear_parameters.aspect_ratio == 0.1
    assert splines_to_curvilinear_parameters.aspect_ratio_grow_factor == 1.1
    assert splines_to_curvilinear_parameters.average_width == 500.0
    assert splines_to_curvilinear_parameters.curvature_adapted_grid_spacing == 1
    assert splines_to_curvilinear_parameters.grow_grid_outside == 0
    assert splines_to_curvilinear_parameters.maximum_num_faces_in_uniform_part == 5
    assert (
        splines_to_curvilinear_parameters.nodes_on_top_of_each_other_tolerance == 0.0001
    )
    assert splines_to_curvilinear_parameters.min_cosine_crossing_angles == 0.95
    assert splines_to_curvilinear_parameters.check_front_collisions == 0
    assert splines_to_curvilinear_parameters.remove_skinny_triangles == 1


def test_splines_to_curvilinear_parameters_invalid_input():
    """Test exceptions due to malformed input for SplinesToCurvilinearParameters"""
    with pytest.raises(ValueError):
        SplinesToCurvilinearParameters(aspect_ratio="abc")

    with pytest.raises(ValueError):
        SplinesToCurvilinearParameters(average_width="invalid")

    with pytest.raises(ValueError):
        SplinesToCurvilinearParameters(curvature_adapted_grid_spacing="2.5")

    with pytest.raises(ValueError):
        SplinesToCurvilinearParameters(nodes_on_top_of_each_other_tolerance="xyz")


def test_mesh_refinement_implicit_string_conversions():
    """Test implicit conversion from string to double for MeshRefinementParameters works"""

    # Test valid input
    refinement_parameters = MeshRefinementParameters(
        refine_intersected="1",
        use_mass_center_when_refining=False,
        min_edge_size="0.5",
        refinement_type=2,
        connect_hanging_nodes=False,
        account_for_samples_outside_face="1",
        max_refinement_iterations="10",
        smoothing_iterations="5",
        max_courant_time="120.0",
        directional_refinement=True,
    )

    assert refinement_parameters.refine_intersected
    assert not refinement_parameters.use_mass_center_when_refining
    assert refinement_parameters.min_edge_size == 0.5
    assert refinement_parameters.refinement_type == RefinementType.REFINEMENT_LEVELS
    assert not refinement_parameters.connect_hanging_nodes
    assert refinement_parameters.account_for_samples_outside_face
    assert refinement_parameters.max_refinement_iterations == 10
    assert refinement_parameters.smoothing_iterations == 5
    assert refinement_parameters.max_courant_time == 120.0
    assert refinement_parameters.directional_refinement


def test_mesh_refinement_parameters_invalid_input():
    """Test exceptions due to malformed input for MeshRefinementParameters"""

    # Test exceptions for invalid inputs
    with pytest.raises(ValueError):
        MeshRefinementParameters(min_edge_size="abc")

    with pytest.raises(ValueError):
        MeshRefinementParameters(max_refinement_iterations="2.5")

    with pytest.raises(ValueError):
        MeshRefinementParameters(max_courant_time="xyz")


def test_make_grid_parameters_implicit_string_conversions():
    """Test implicit conversion from string to double for MakeGridParameters works"""

    # Test valid input
    make_grid_parameters = MakeGridParameters(
        num_columns="3",
        num_rows="3",
        angle="0.0",
        origin_x="0.0",
        origin_y="0.0",
        block_size_x="10.0",
        block_size_y="10.0",
        upper_right_x="0.0",
        upper_right_y="0.0",
    )

    assert make_grid_parameters.num_columns == 3
    assert make_grid_parameters.num_rows == 3
    assert make_grid_parameters.angle == 0.0
    assert make_grid_parameters.origin_x == 0.0
    assert make_grid_parameters.origin_y == 0.0
    assert make_grid_parameters.block_size_x == 10.0
    assert make_grid_parameters.block_size_y == 10.0
    assert make_grid_parameters.upper_right_x == 0.0
    assert make_grid_parameters.upper_right_y == 0.0


def test_cmake_grid_parameters_invalid_input():
    """Test exceptions due to malformed input for MakeGridParameters"""

    with pytest.raises(ValueError):
        MakeGridParameters(num_columns="abc")

    with pytest.raises(ValueError):
        MakeGridParameters(origin_x="invalid")

    with pytest.raises(ValueError):
        MakeGridParameters(num_rows="2.5")


def test_mesh1d_implicit_string_conversions():
    """Test implicit conversion from string to double for Mesh1d works"""

    node_x = ["1.0", "2.0", "3.0"]
    node_y = ["4.0", "5.0", "6.0"]
    edge_nodes = ["0", "1", "1", "2"]

    mesh_1d = Mesh1d(node_x=node_x, node_y=node_y, edge_nodes=edge_nodes)

    assert np.array_equal(mesh_1d.node_x, np.array([1.0, 2.0, 3.0], dtype=np.double))
    assert np.array_equal(mesh_1d.node_y, np.array([4.0, 5.0, 6.0], dtype=np.double))
    assert np.array_equal(mesh_1d.edge_nodes, np.array([0, 1, 1, 2], dtype=int))


def test_mesh1d_invalid_input():
    """Test exceptions due to malformed input for Mesh1d"""

    # Test exceptions for invalid inputs
    with pytest.raises(ValueError):
        Mesh1d(
            node_x=["1.0", "abc", "3.0"],
            node_y=["4.0", "5.0", "6.0"],
            edge_nodes=["0", "1"],
        )

    with pytest.raises(TypeError):
        Mesh1d(edge_nodes=["0", "1", "1", "2.5"])


def test_contacts_implicit_string_conversions():
    """Test implicit conversion from string to double for Contacts works"""

    # Test valid input
    mesh1d_indices = ["0", "1", "2"]
    mesh2d_indices = ["0", "1", "2"]

    contacts = Contacts(mesh1d_indices=mesh1d_indices, mesh2d_indices=mesh2d_indices)

    assert np.array_equal(contacts.mesh1d_indices, np.array([0, 1, 2], dtype=np.int))
    assert np.array_equal(contacts.mesh2d_indices, np.array([0, 1, 2], dtype=np.int))


def test_contacts_invalid_input():
    """Test exceptions due to malformed input for Contacts"""
    with pytest.raises(ValueError):
        Contacts(mesh1d_indices=["0", "1", "abc"], mesh2d_indices=["0", "1", "2"])

    with pytest.raises(ValueError):
        Contacts(mesh2d_indices=["0", "1", "2.5"], mesh1d_indices=["0", "1", "2"])


def test_gridded_parameters_implicit_string_conversions():
    """Test implicit conversion from string to double for GriddedSamples works"""

    num_x = "3"
    num_y = "4"
    x_origin = "1.0"
    y_origin = "2.0"
    cell_size = "0.5"
    x_coordinates = ["1.0", "2.0", "3.0"]
    y_coordinates = ["2.0", "3.0", "4.0"]
    values = np.array(
        [
            "10.0",
            "20.0",
            "30.0",
            "40.0",
            "50.0",
            "60.0",
            "70.0",
            "80.0",
            "90.0",
            "100.0",
        ]
    )

    gridded_samples = GriddedSamples(
        num_x=num_x,
        num_y=num_y,
        x_origin=x_origin,
        y_origin=y_origin,
        cell_size=cell_size,
        x_coordinates=x_coordinates,
        y_coordinates=y_coordinates,
        values=values,
    )

    assert gridded_samples.num_x == 3
    assert gridded_samples.num_y == 4
    assert gridded_samples.x_origin == 1.0
    assert gridded_samples.y_origin == 2.0
    assert gridded_samples.cell_size == 0.5
    assert np.array_equal(
        gridded_samples.x_coordinates, np.array([1.0, 2.0, 3.0], dtype=float)
    )
    assert np.array_equal(
        gridded_samples.y_coordinates, np.array([2.0, 3.0, 4.0], dtype=float)
    )
    assert np.array_equal(
        gridded_samples.values,
        np.array(
            [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
            dtype=float,
        ),
    )


def test_gridded_parameters_invalid_input():
    """Test exceptions due to malformed input for GriddedSamples"""

    with pytest.raises(ValueError):
        GriddedSamples(num_x="abc")

    with pytest.raises(ValueError):
        GriddedSamples(cell_size="invalid")

    with pytest.raises(ValueError):
        GriddedSamples(x_origin="invalid")
