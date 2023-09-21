import numpy as np
from pytest import approx

from meshkernel import (
    CurvilinearParameters,
    GeometryList,
    MakeGridParameters,
    MeshKernel,
    OrthogonalizationParameters,
    SplinesToCurvilinearParameters,
)


def create_meshkernel_instance_with_curvilinear_grid():
    r"""A function for creating an instance of meshkernel with a uniform curvilinear grid."""
    mk = MeshKernel()

    separator = -999.0
    splines_x = np.array(
        [
            2.0,
            4.0,
            7.0,
            separator,
            -1.0,
            1.0,
            5.0,
            separator,
            3.0,
            -2.0,
            separator,
            7.0,
            4.0,
        ],
        dtype=np.double,
    )
    splines_y = np.array(
        [
            1.0,
            3.0,
            4.0,
            separator,
            4.0,
            6.0,
            7.0,
            separator,
            1.0,
            6.0,
            separator,
            3.0,
            8.0,
        ],
        dtype=np.double,
    )
    splines_values = np.zeros_like(splines_x)
    splines = GeometryList(splines_x, splines_y, splines_values)

    curvilinear_parameters = CurvilinearParameters()
    curvilinear_parameters.n_refinement = 10
    curvilinear_parameters.m_refinement = 10

    mk.curvilinear_compute_transfinite_from_splines(splines, curvilinear_parameters)

    return mk


def create_meshkernel_instance_with_skewed_curvilinear_grid(num_columns=3, num_rows=3):
    """A function for creating an instance of meshkernel with a non uniform curvilinear grid.

    Args:
        num_columns (int): The number of columns to generate.
        num_rows (int): The number of rows to generate.
    """
    mk = MeshKernel()

    make_grid_parameters = MakeGridParameters()
    make_grid_parameters.num_columns = num_columns
    make_grid_parameters.num_rows = num_rows
    make_grid_parameters.angle = 0.0
    make_grid_parameters.block_size = 0.0
    make_grid_parameters.origin_x = 0.0
    make_grid_parameters.origin_y = 0.0
    make_grid_parameters.block_size_x = 10.0
    make_grid_parameters.block_size_y = 10.0

    node_x = np.empty(0, dtype=np.double)
    node_y = np.empty(0, dtype=np.double)
    geometry_list = GeometryList(node_x, node_y)

    mk.curvilinear_make_uniform(make_grid_parameters, geometry_list)

    # Skew the grid by moving a node
    mk.curvilinear_move_node(10.0, 20.0, 18.0, 12.0)

    return mk


def test_curvilinear_compute_transfinite_from_splines():
    r"""Tests `curvilinear_compute_transfinite_from_splines` generates a curvilinear grid."""
    mk = create_meshkernel_instance_with_curvilinear_grid()

    output_curvilinear = mk.curvilineargrid_get()

    # Test the number of m and n are as expected
    assert output_curvilinear.num_m == 11
    assert output_curvilinear.num_n == 11


def test_curvilinear_compute_orthogonal_from_splines():
    r"""Tests `curvilinear_compute_orthogonal_from_splines` generates a curvilinear grid using
    the advancing front algorithm.
    """
    mk = MeshKernel()

    separator = -999.0
    splines_x = np.array(
        [
            152.001571655273,
            374.752960205078,
            850.255920410156,
            separator,
            72.5010681152344,
            462.503479003906,
            separator,
        ],
        dtype=np.double,
    )
    splines_y = np.array(
        [
            86.6264953613281,
            336.378997802734,
            499.130676269531,
            separator,
            391.129577636719,
            90.3765411376953,
            separator,
        ],
        dtype=np.double,
    )

    splines_values = np.zeros_like(splines_x)
    splines = GeometryList(splines_x, splines_y, splines_values)

    curvilinear_parameters = CurvilinearParameters()
    curvilinear_parameters.n_refinement = 40
    curvilinear_parameters.m_refinement = 20

    splines_to_curvilinear_parameters = SplinesToCurvilinearParameters()
    splines_to_curvilinear_parameters.aspect_ratio = 0.1
    splines_to_curvilinear_parameters.aspect_ratio_grow_factor = 1.1
    splines_to_curvilinear_parameters.average_width = 500.0
    splines_to_curvilinear_parameters.nodes_on_top_of_each_other_tolerance = 1e-4
    splines_to_curvilinear_parameters.min_cosine_crossing_angles = 0.95
    splines_to_curvilinear_parameters.check_front_collisions = 0
    splines_to_curvilinear_parameters.curvature_adapted_grid_spacing = 1
    splines_to_curvilinear_parameters.remove_skinny_triangles = 0

    mk.curvilinear_compute_orthogonal_from_splines(
        splines, curvilinear_parameters, splines_to_curvilinear_parameters
    )

    output_curvilinear = mk.curvilineargrid_get()

    # Test the number of m and n are as expected
    assert output_curvilinear.num_m == 3
    assert output_curvilinear.num_n == 9


def test_curvilinear_convert_to_mesh2d():
    r"""Tests `curvilinear_compute_transfinite_from_splines` converts a curvilinear mesh into an unstructured mesh."""
    mk = create_meshkernel_instance_with_curvilinear_grid()

    mk.curvilinear_convert_to_mesh2d()

    mesh2d = mk.mesh2d_get()

    curvilinear_grid = mk.curvilineargrid_get()

    # Test curvilinear grid is empty and mesh2d is filled
    assert curvilinear_grid.num_m == 0
    assert curvilinear_grid.num_n == 0
    assert len(mesh2d.node_x) == 121
    assert len(mesh2d.edge_nodes) == 440


def test_curvilinear_make_uniform():
    r"""Tests `curvilinear_make_uniform` makes a curvilinear grid."""
    mk = MeshKernel()

    make_grid_parameters = MakeGridParameters()
    make_grid_parameters.num_columns = 3
    make_grid_parameters.num_rows = 3
    make_grid_parameters.angle = 0.0
    make_grid_parameters.block_size = 0.0
    make_grid_parameters.origin_x = 0.0
    make_grid_parameters.origin_y = 0.0
    make_grid_parameters.block_size_x = 10.0
    make_grid_parameters.block_size_y = 10.0

    node_x = np.empty(0, dtype=np.double)
    node_y = np.empty(0, dtype=np.double)
    geometry_list = GeometryList(node_x, node_y)

    mk.curvilinear_make_uniform(make_grid_parameters, geometry_list)

    curvilinear_grid = mk.curvilineargrid_get()

    # Test the number of m and n
    assert curvilinear_grid.num_m == 4
    assert curvilinear_grid.num_n == 4


def test_curvilinear_make_uniform_defined_extension_spherical_coordinates():
    r"""Tests `curvilinear_make_uniform` makes a curvilinear grid within
    a defined extension in spherical coordinates."""
    mk = MeshKernel(is_geographic=True)

    make_grid_parameters = MakeGridParameters()
    make_grid_parameters.origin_x = -1.0
    make_grid_parameters.origin_y = 49.1
    make_grid_parameters.upper_right_x = -0.2
    make_grid_parameters.upper_right_y = 49.6
    make_grid_parameters.block_size_x = 0.01
    make_grid_parameters.block_size_y = 0.01

    mk.curvilinear_make_uniform_on_extension(make_grid_parameters)

    curvilinear_grid = mk.curvilineargrid_get()

    # Test the number of m and n
    assert curvilinear_grid.num_m == 103
    assert curvilinear_grid.num_n == 81


def test_curvilinear_make_uniform_with_polygon():
    r"""Tests `curvilinear_make_uniform` makes a curvilinear grid using a polygon."""
    mk = MeshKernel()

    make_grid_parameters = MakeGridParameters()
    make_grid_parameters.num_columns = 3
    make_grid_parameters.num_rows = 3
    make_grid_parameters.angle = 0.0
    make_grid_parameters.block_size = 0.0
    make_grid_parameters.origin_x = 0.0
    make_grid_parameters.origin_y = 0.0
    make_grid_parameters.block_size_x = 1.0
    make_grid_parameters.block_size_y = 1.0

    node_x = np.array([2.5, 5.5, 3.5, 0.5, 2.5], dtype=np.double)
    node_y = np.array([0.5, 3.0, 5.0, 2.5, 0.5], dtype=np.double)
    geometry_list = GeometryList(node_x, node_y)

    mk.curvilinear_make_uniform(make_grid_parameters, geometry_list)

    curvilinear_grid = mk.curvilineargrid_get()

    # Test the number of m and n
    assert curvilinear_grid.num_m == 11
    assert curvilinear_grid.num_n == 11


def test_curvilinear_refine_derefine():
    r"""Tests `curvilinear_refine` refines a curvilinear grid and
    `curvilinear_derefine` de-refines a curvilinear grid .
    """
    mk = create_meshkernel_instance_with_curvilinear_grid()

    mk.curvilinear_refine(2.299, 4.612, 3.074, 3.684, 2)

    curvilinear_grid = mk.curvilineargrid_get()

    # After refinement
    assert curvilinear_grid.num_m == 11
    assert curvilinear_grid.num_n == 14

    mk.curvilinear_derefine(2.299, 4.612, 3.074, 3.684)

    curvilinear_grid = mk.curvilineargrid_get()

    # After de-refinement
    assert curvilinear_grid.num_m == 11
    assert curvilinear_grid.num_n == 9


def test_curvilinear_compute_transfinite_from_polygon():
    r"""Tests `curvilinear_compute_transfinite_from_polygon` generates curvilinear grid from a polygon.

    Input polygon:
    6---5---4
    |       |
    7       3
    |       |
    0---1---2

    Generated curvilinear grid:

    6---7---8
    |   |   |
    3---4---5
    |   |   |
    0---1---2
    """

    # The input polygon
    node_x = np.array([0, 5, 10, 10, 10, 5, 0, 0, 0], dtype=np.double)
    node_y = np.array([0, 0, 0, 5, 10, 10, 10, 5, 0], dtype=np.double)
    geometry_list = GeometryList(node_x, node_y)

    mk = MeshKernel()

    mk.curvilinear_compute_transfinite_from_polygon(geometry_list, 0, 2, 4, False)

    curvilinear_grid = mk.curvilineargrid_get()

    assert curvilinear_grid.num_m == 3
    assert curvilinear_grid.num_n == 3


def test_curvilinear_compute_transfinite_from_triangle():
    r"""Tests `curvilinear_compute_transfinite_from_triangle` computes a curvilinear grid from a polygon
    having a triangular shape.
    """

    node_x = np.array(
        [
            444.504791,
            427.731781,
            405.640503,
            381.094666,
            451.050354,
            528.778931,
            593.416260,
            558.643005,
            526.733398,
            444.095703,
            444.504791,
        ],
        dtype=np.double,
    )

    node_y = np.array(
        [
            437.155945,
            382.745758,
            317.699005,
            262.470612,
            262.879700,
            263.288788,
            266.561584,
            324.653687,
            377.836578,
            436.746857,
            437.155945,
        ],
        dtype=np.double,
    )

    geometry_list = GeometryList(node_x, node_y)

    mk = MeshKernel()

    mk.curvilinear_compute_transfinite_from_triangle(geometry_list, 0, 3, 6)

    curvilinear_grid = mk.curvilineargrid_get()

    # Test a curvilinear grid was generated
    assert curvilinear_grid.num_m == 4
    assert curvilinear_grid.num_n == 4


def test_curvilinear_grid_orthogonalization():
    r"""Tests `curvilinear_orthogonalize` orthogonalizes a curvilinear grid."""

    mk = create_meshkernel_instance_with_skewed_curvilinear_grid()

    orthogonalization_parameters = OrthogonalizationParameters()
    orthogonalization_parameters.outer_iterations = 1
    orthogonalization_parameters.boundary_iterations = 25
    orthogonalization_parameters.inner_iterations = 25
    orthogonalization_parameters.orthogonalization_to_smoothing_factor = 0.975

    # Initialize the curvilinear grid orthogonalization algorithm
    mk.curvilinear_initialize_orthogonalize(orthogonalization_parameters)

    # Initialize the curvilinear grid orthogonalization algorithm
    mk.curvilinear_set_block_orthogonalize(0.0, 0.0, 30.0, 30.0)

    # Performs orthogonalization
    mk.curvilinear_orthogonalize()
    curvilinear_grid = mk.curvilineargrid_get()

    # Assert nodal position after orthogonalization is closer to 10.0, 20.0
    assert curvilinear_grid.node_x[9] == approx(11.841396536135521, 0.0001)
    assert curvilinear_grid.node_y[9] == approx(18.158586078094562, 0.0001)


def test_curvilinear_grid_orthogonalization_with_frozen_line():
    r"""Tests `curvilinear_orthogonalize` with a frozen line orthogonalizes a curvilinear grid,
    except on the frozen line, whe nodal positions are fixed.
    """
    mk = create_meshkernel_instance_with_skewed_curvilinear_grid()

    orthogonalization_parameters = OrthogonalizationParameters()
    orthogonalization_parameters.outer_iterations = 1
    orthogonalization_parameters.boundary_iterations = 25
    orthogonalization_parameters.inner_iterations = 25
    orthogonalization_parameters.orthogonalization_to_smoothing_factor = 0.975

    # Initialize the curvilinear grid orthogonalization algorithm
    mk.curvilinear_initialize_orthogonalize(orthogonalization_parameters)

    # Initialize the curvilinear grid orthogonalization algorithm
    mk.curvilinear_set_block_orthogonalize(0.0, 0.0, 30.0, 30.0)

    # Freeze the vertical grid line where the moved node is
    mk.curvilinear_set_frozen_lines_orthogonalize(10.0, 0.0, 10.0, 30.0)

    # Performs orthogonalization
    mk.curvilinear_orthogonalize()

    # Get the result
    curvilinear_grid = mk.curvilineargrid_get()

    # Assert the position of the skewed node after orthogonalization has not changed
    assert curvilinear_grid.node_x[9] == approx(18.0, 0.0001)
    assert curvilinear_grid.node_y[9] == approx(12.0, 0.0001)


def test_curvilinear_smoothing():
    r"""Tests `curvilinear_smoothing` smooths a curvilinear grid."""

    mk = create_meshkernel_instance_with_skewed_curvilinear_grid()

    # Perform smoothing
    mk.curvilinear_smoothing(10, 0.0, 0.0, 30.0, 30.0)

    # Get the result
    curvilinear_grid = mk.curvilineargrid_get()

    # Assert the nodal position after smoothing
    assert curvilinear_grid.node_x[9] == approx(10.315557749152806, 0.0001)
    assert curvilinear_grid.node_y[9] == approx(19.68444225084719, 0.0001)


def test_curvilinear_smoothing_directional():
    r"""Tests `curvilinear_smoothing_directional` smooths a curvilinear grid along one direction."""

    mk = create_meshkernel_instance_with_skewed_curvilinear_grid()

    # Perform directional smoothing
    mk.curvilinear_smoothing_directional(
        10, 10.0, 0.0, 10.0, 30.0, 0.0, 0.0, 30.0, 30.0
    )

    # Get the result
    curvilinear_grid = mk.curvilineargrid_get()

    # Assert the nodal position after directional smoothing
    assert curvilinear_grid.node_x[9] == approx(14.278566438378412, 0.0001)
    assert curvilinear_grid.node_y[9] == approx(20.37322551364857, 0.0001)


def test_curvilinear_line_shift():
    r"""Tests curvilinear line shift shift workflow, where some nodes on a grid line are shifted,
    and the shifting is distributed on a curvilinear grid block.
    """
    mk = create_meshkernel_instance_with_skewed_curvilinear_grid(5, 5)

    # Initialize line shift algorithm
    mk.curvilinear_initialize_line_shift()

    # Sets the line to shift
    mk.curvilinear_set_line_line_shift(0.0, 0.0, 0.0, 50.0)

    # Sets the block where the shift will be distributed
    mk.curvilinear_set_block_line_shift(0.0, 0.0, 20.0, 50.0)

    # Move left side nodes on a new position
    mk.curvilinear_move_node_line_shift(0.0, 0.0, -10.0, 0.0)
    mk.curvilinear_move_node_line_shift(0.0, 10.0, -10.0, 10.0)
    mk.curvilinear_move_node_line_shift(0.0, 20.0, -10.0, 20.0)
    mk.curvilinear_move_node_line_shift(0.0, 30.0, -10.0, 30.0)
    mk.curvilinear_move_node_line_shift(0.0, 40.0, -10.0, 40.0)
    mk.curvilinear_move_node_line_shift(0.0, 50.0, -10.0, 50.0)

    # Performs curvilinear grid line shift
    mk.curvilinear_line_shift()

    # Get the result
    curvilinear_grid = mk.curvilineargrid_get()

    # Assert the nodal position after line shift
    assert curvilinear_grid.node_x[0] == -10.0
    assert curvilinear_grid.node_x[6] == -10.0
    assert curvilinear_grid.node_x[12] == -10.0
    assert curvilinear_grid.node_x[18] == -10.0
    assert curvilinear_grid.node_x[24] == -10.0
    assert curvilinear_grid.node_x[30] == -10.0

    assert curvilinear_grid.node_y[0] == 0
    assert curvilinear_grid.node_y[6] == 10.0
    assert curvilinear_grid.node_y[12] == 20.0
    assert curvilinear_grid.node_y[18] == 30.0
    assert curvilinear_grid.node_y[24] == 40.0
    assert curvilinear_grid.node_y[30] == 50.0


def test_curvilinear_insert_face():
    r"""Tests 'curvilinear_insert_face' inserts two new faces."""
    mk = create_meshkernel_instance_with_skewed_curvilinear_grid(5, 5)

    # Inserts two faces
    mk.curvilinear_insert_face(-10.0, 5.0)
    mk.curvilinear_insert_face(-5.0, 10.0)

    # Get the result
    curvilinear_grid = mk.curvilineargrid_get()

    # Assert two new faces have been inserted
    assert curvilinear_grid.node_x[0] == -20.0
    assert curvilinear_grid.node_x[1] == -10.0
    assert curvilinear_grid.node_x[8] == -20.0
    assert curvilinear_grid.node_x[9] == -10.0

    assert curvilinear_grid.node_y[0] == 0.0
    assert curvilinear_grid.node_y[1] == 0.0
    assert curvilinear_grid.node_y[8] == 10.0
    assert curvilinear_grid.node_y[9] == 10.0


def test_curvilinear_delete_node():
    r"""Tests 'curvilinear_delete_node' deletes a node."""
    mk = create_meshkernel_instance_with_skewed_curvilinear_grid(5, 5)

    # Inserts two faces
    mk.curvilinear_delete_node(0.0, 0.0)

    # Get the result
    curvilinear_grid = mk.curvilineargrid_get()

    # Test the bottom right corner is marked as deleted (invalid coordinates)
    assert curvilinear_grid.node_x[0] == -999.0
    assert curvilinear_grid.node_y[0] == -999.0


def test_curvilinear_line_attraction_repulsion():
    r"""Tests 'curvilinear_line_attraction_repulsion' repels lines."""
    mk = create_meshkernel_instance_with_skewed_curvilinear_grid(5, 5)

    # Repels lines
    mk.curvilinear_line_attraction_repulsion(
        1.0, 30.0, 0.0, 30.0, 50.0, 10.0, 0.0, 50.0, 50.0
    )

    # Get the result
    curvilinear_grid = mk.curvilineargrid_get()

    # Test a node on the third grid line was shifted from 20 to 15
    assert curvilinear_grid.node_x[2] == 15.0


def test_curvilinear_line_mirror():
    r"""Tests 'curvilinear_line_mirror' replicates (mirrors) a boundary grid line."""
    mk = create_meshkernel_instance_with_skewed_curvilinear_grid(5, 5)

    # Mirrors the left grid line to left
    mk.curvilinear_line_mirror(2.0, 0.0, 0.0, 0.0, 50.0)

    # Get the result
    curvilinear_grid = mk.curvilineargrid_get()

    # Test some few nodes have been generated in the left side
    assert curvilinear_grid.node_x[0] == -20.0
    assert curvilinear_grid.node_x[7] == -20.0
    assert curvilinear_grid.node_y[0] == 0.0
    assert curvilinear_grid.node_y[7] == 10.0
