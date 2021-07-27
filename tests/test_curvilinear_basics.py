import numpy as np

from meshkernel import (
    CurvilinearParameters,
    GeometryList,
    MeshKernel,
    SplinesToCurvilinearParameters
)


def test_curvilinear_compute_transfinite_from_splines():
    r"""Tests `curvilinear_compute_transfinite_from_splines` generates a curvilinear grid.
    """
    mk = MeshKernel()

    separator = -999.0
    splines_x = np.array([2.172341E+02, 4.314185E+02, 8.064374E+02, separator,
                          2.894012E+01, 2.344944E+02, 6.424647E+02, separator,
                          2.265137E+00, 2.799988E+02, separator,
                          5.067361E+02, 7.475956E+02], dtype=np.double)
    splines_y = np.array([-2.415445E+01, 1.947381E+02, 3.987241E+02, separator,
                          2.010146E+02, 3.720490E+02, 5.917262E+02, separator,
                          2.802553E+02, -2.807726E+01, separator,
                          6.034946E+02, 3.336055E+02], dtype=np.double)
    splines_values = np.zeros_like(splines_x)
    splines = GeometryList(splines_x, splines_y, splines_values)

    curvilinearParameters = CurvilinearParameters()
    curvilinearParameters.n_refinement = 40
    curvilinearParameters.m_refinement = 20

    mk.curvilinear_compute_transfinite_from_splines(splines, curvilinearParameters)

    output_curvilinear = mk.curvilineargrid_get()

    # Test the number of m and n are as expected
    assert output_curvilinear.num_m == 21
    assert output_curvilinear.num_n == 41


def test_curvilinear_compute_orthogonal_from_splines():
    r"""Tests `curvilinear_compute_orthogonal_from_splines` generates a curvilinear grid.
    """
    mk = MeshKernel()

    separator = -999.0
    splines_x = np.array([152.001571655273, 374.752960205078, 850.255920410156, separator,
                          72.5010681152344, 462.503479003906, separator], dtype=np.double)
    splines_y = np.array([86.6264953613281, 336.378997802734, 499.130676269531, separator,
                          391.129577636719, 90.3765411376953, separator], dtype=np.double)

    splines_values = np.zeros_like(splines_x)
    splines = GeometryList(splines_x, splines_y, splines_values)

    curvilinearParameters = CurvilinearParameters()
    curvilinearParameters.n_refinement = 40
    curvilinearParameters.m_refinement = 20

    splinesToCurvilinearParameters = SplinesToCurvilinearParameters()
    splinesToCurvilinearParameters.aspect_ratio = 0.1
    splinesToCurvilinearParameters.aspect_ratio_grow_factor = 1.1
    splinesToCurvilinearParameters.average_width = 500.0
    splinesToCurvilinearParameters.nodes_on_top_of_each_other_tolerance = 1e-4
    splinesToCurvilinearParameters.min_cosine_crossing_angles = 0.95
    splinesToCurvilinearParameters.check_front_collisions = 0
    splinesToCurvilinearParameters.curvature_adapted_grid_spacing = 1
    splinesToCurvilinearParameters.remove_skinny_triangles = 0

    mk.curvilinear_compute_orthogonal_from_splines(splines, curvilinearParameters, splinesToCurvilinearParameters)

    output_curvilinear = mk.curvilineargrid_get()

    # Test the number of m and n are as expected
    assert output_curvilinear.num_m == 3
    assert output_curvilinear.num_n == 9


def test_mkernel_curvilinear_convert_to_mesh2d():
    r"""Tests `mkernel_curvilinear_convert_to_mesh2d` converts a curvilinear mesh into an unstructured mesh.
    """
    mk = MeshKernel()

    separator = -999.0
    splines_x = np.array([2.172341E+02, 4.314185E+02, 8.064374E+02, separator,
                          2.894012E+01, 2.344944E+02, 6.424647E+02, separator,
                          2.265137E+00, 2.799988E+02, separator,
                          5.067361E+02, 7.475956E+02], dtype=np.double)
    splines_y = np.array([-2.415445E+01, 1.947381E+02, 3.987241E+02, separator,
                          2.010146E+02, 3.720490E+02, 5.917262E+02, separator,
                          2.802553E+02, -2.807726E+01, separator,
                          6.034946E+02, 3.336055E+02], dtype=np.double)
    splines_values = np.zeros_like(splines_x)
    splines = GeometryList(splines_x, splines_y, splines_values)

    curvilinearParameters = CurvilinearParameters()
    curvilinearParameters.n_refinement = 40
    curvilinearParameters.m_refinement = 20

    mk.curvilinear_compute_transfinite_from_splines(splines, curvilinearParameters)

    mk.curvilinear_convert_to_mesh2d()

    mesh2d = mk.mesh2d_get()

    curvilineargrid = mk.curvilineargrid_get()

    # Test the number of m and n are as expected
    assert curvilineargrid.num_m == 0
    assert curvilineargrid.num_n == 0
    assert len(mesh2d.node_x) == 861
    assert len(mesh2d.edge_nodes) == 3320
