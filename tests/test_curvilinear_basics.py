import numpy as np

from meshkernel import (
    GeometryList,
    CurvilinearParameters,
    MeshKernel,
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
