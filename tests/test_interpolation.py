import numpy as np
import pytest
from numpy.testing import assert_array_equal
from pytest import approx

from meshkernel import GeometryList, Mesh2d, Mesh2dFactory, Mesh2dLocation, MeshKernel
from meshkernel.py_structures import AveragingMethod


@pytest.fixture(scope="function")
def meshkernel_with_mesh2d():
    """Creates a new instance of 'meshkernel' and sets a Mesh2d with the specified dimensions.

    Args:
        rows (int): Number of node rows
        columns (int): Number of node columns

    Returns:
        MeshKernel: The created instance of `meshkernel`
    """

    def _create(rows: int, columns: int):
        mesh2d = Mesh2dFactory.create_rectilinear_mesh(rows, columns)
        mk = MeshKernel()

        mk.set_mesh2d(mesh2d)

        return mk

    return _create


def test_triangulation_interpolation_mesh2d_on_faces(
    meshkernel_with_mesh2d: MeshKernel,
):
    """Tests `triangulation_interpolation_mesh2d` on the faces of a 4x4 Mesh2d."""

    mk = meshkernel_with_mesh2d(4, 4)

    samples_x = np.array([0.4, 1.3, 2.6, 0.6, 1.6, 2.4, 0.4, 1.6, 2.5], dtype=np.double)
    samples_y = np.array([0.5, 0.5, 0.5, 1.5, 1.5, 1.5, 2.5, 2.5, 2.5], dtype=np.double)
    samples_values = np.array(
        [0.9, 1.8, 3.1, 4.1, 5.1, 5.9, 6.9, 8.1, 9], dtype=np.double
    )
    samples = GeometryList(samples_x, samples_y, samples_values)

    interpolation = mk.triangulation_interpolation_mesh2d(samples, Mesh2dLocation.FACES)

    assert interpolation.x_coordinates[0] == 0.5
    assert interpolation.x_coordinates[1] == 1.5
    assert interpolation.x_coordinates[2] == 2.5
    assert interpolation.x_coordinates[3] == 0.5
    assert interpolation.x_coordinates[4] == 1.5
    assert interpolation.x_coordinates[5] == 2.5
    assert interpolation.x_coordinates[6] == 0.5
    assert interpolation.x_coordinates[7] == 1.5
    assert interpolation.x_coordinates[8] == 2.5

    assert interpolation.y_coordinates[0] == 0.5
    assert interpolation.y_coordinates[1] == 0.5
    assert interpolation.y_coordinates[2] == 0.5
    assert interpolation.y_coordinates[3] == 1.5
    assert interpolation.y_coordinates[4] == 1.5
    assert interpolation.y_coordinates[5] == 1.5
    assert interpolation.y_coordinates[6] == 2.5
    assert interpolation.y_coordinates[7] == 2.5
    assert interpolation.y_coordinates[8] == 2.5

    assert interpolation.values[0] == approx(1, abs=0.00000001)
    assert interpolation.values[1] == approx(2, abs=0.00000001)
    assert interpolation.values[2] == approx(3, abs=0.00000001)
    assert interpolation.values[3] == approx(4, abs=0.00000001)
    assert interpolation.values[4] == approx(5, abs=0.00000001)
    assert interpolation.values[5] == approx(6, abs=0.00000001)
    assert interpolation.values[6] == approx(7, abs=0.00000001)
    assert interpolation.values[7] == approx(8, abs=0.00000001)
    assert interpolation.values[8] == approx(9, abs=0.00000001)


def test_triangulation_interpolation_mesh2d_on_nodes(
    meshkernel_with_mesh2d: MeshKernel,
):
    """Tests `triangulation_interpolation_mesh2d` on the nodes of a 4x4 Mesh2d."""

    mk = meshkernel_with_mesh2d(3, 3)

    samples_x = np.array([0.0, 0.9, 2.1, 0.1, 1.1, 2.2, 0.0, 1.2, 2.1], dtype=np.double)
    samples_y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0], dtype=np.double)
    samples_values = np.array(
        [1, 1.9, 3.1, 4.1, 5.1, 6.2, 7.0, 8.2, 9.1], dtype=np.double
    )
    samples = GeometryList(samples_x, samples_y, samples_values)

    interpolation = mk.triangulation_interpolation_mesh2d(samples, Mesh2dLocation.NODES)

    assert interpolation.x_coordinates[0] == 0.0
    assert interpolation.x_coordinates[1] == 1.0
    assert interpolation.x_coordinates[2] == 2.0
    assert interpolation.x_coordinates[3] == 0.0
    assert interpolation.x_coordinates[4] == 1.0
    assert interpolation.x_coordinates[5] == 2.0
    assert interpolation.x_coordinates[6] == 0.0
    assert interpolation.x_coordinates[7] == 1.0
    assert interpolation.x_coordinates[8] == 2.0

    assert interpolation.y_coordinates[0] == 0.0
    assert interpolation.y_coordinates[1] == 0.0
    assert interpolation.y_coordinates[2] == 0.0
    assert interpolation.y_coordinates[3] == 1.0
    assert interpolation.y_coordinates[4] == 1.0
    assert interpolation.y_coordinates[5] == 1.0
    assert interpolation.y_coordinates[6] == 2.0
    assert interpolation.y_coordinates[7] == 2.0
    assert interpolation.y_coordinates[8] == 2.0

    assert interpolation.values[0] == approx(1, abs=0.00000001)
    assert interpolation.values[1] == approx(2, abs=0.00000001)
    assert interpolation.values[2] == approx(3, abs=0.00000001)
    assert interpolation.values[3] == approx(4, abs=0.00000001)
    assert interpolation.values[4] == approx(5, abs=0.00000001)
    assert interpolation.values[5] == approx(6, abs=0.00000001)
    assert interpolation.values[6] == approx(7, abs=0.00000001)
    assert interpolation.values[7] == approx(8, abs=0.00000001)
    assert interpolation.values[8] == approx(9, abs=0.00000001)


def test_triangulation_interpolation_mesh2d_on_edges(
    meshkernel_with_mesh2d: MeshKernel,
):
    """Tests `triangulation_interpolation_mesh2d` on the edges of a 4x4 Mesh2d."""

    mk = meshkernel_with_mesh2d(3, 3)

    samples_x = np.array(
        [0.0, 1.1, 2.2, 0.0, 0.9, 2.0, 0.4, 1.6, 0.6, 1.3, 0.2, 1.5], dtype=np.double
    )
    samples_y = np.array(
        [0.5, 0.5, 0.5, 1.5, 1.5, 1.5, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0], dtype=np.double
    )
    samples_values = np.array(
        [1, 2.1, 3.2, 1.0, 1.9, 3.0, 1.4, 2.6, 1.6, 2.3, 1.2, 2.5], dtype=np.double
    )
    samples = GeometryList(samples_x, samples_y, samples_values)

    interpolation = mk.triangulation_interpolation_mesh2d(samples, Mesh2dLocation.EDGES)

    assert interpolation.x_coordinates[0] == 0.0
    assert interpolation.x_coordinates[1] == 1.0
    assert interpolation.x_coordinates[2] == 2.0
    assert interpolation.x_coordinates[3] == 0.0
    assert interpolation.x_coordinates[4] == 1.0
    assert interpolation.x_coordinates[5] == 2.0
    assert interpolation.x_coordinates[6] == 0.5
    assert interpolation.x_coordinates[7] == 1.5
    assert interpolation.x_coordinates[8] == 0.5
    assert interpolation.x_coordinates[9] == 1.5
    assert interpolation.x_coordinates[10] == 0.5
    assert interpolation.x_coordinates[11] == 1.5

    assert interpolation.y_coordinates[0] == 0.5
    assert interpolation.y_coordinates[1] == 0.5
    assert interpolation.y_coordinates[2] == 0.5
    assert interpolation.y_coordinates[3] == 1.5
    assert interpolation.y_coordinates[4] == 1.5
    assert interpolation.y_coordinates[5] == 1.5
    assert interpolation.y_coordinates[6] == 0.0
    assert interpolation.y_coordinates[7] == 0.0
    assert interpolation.y_coordinates[8] == 1.0
    assert interpolation.y_coordinates[9] == 1.0
    assert interpolation.y_coordinates[10] == 2.0
    assert interpolation.y_coordinates[11] == 2.0

    assert interpolation.values[0] == approx(1, abs=0.00000001)
    assert interpolation.values[1] == approx(2, abs=0.00000001)
    assert interpolation.values[2] == approx(3, abs=0.00000001)
    assert interpolation.values[3] == approx(1, abs=0.00000001)
    assert interpolation.values[4] == approx(2, abs=0.00000001)
    assert interpolation.values[5] == approx(3, abs=0.00000001)
    assert interpolation.values[6] == approx(1.5, abs=0.00000001)
    assert interpolation.values[7] == approx(2.5, abs=0.00000001)
    assert interpolation.values[8] == approx(1.5, abs=0.00000001)
    assert interpolation.values[9] == approx(2.5, abs=0.00000001)
    assert interpolation.values[10] == approx(1.5, abs=0.00000001)
    assert interpolation.values[11] == approx(2.5, abs=0.00000001)


cases_averaging_interpolation_mesh2d = [
    (
        AveragingMethod.SIMPLE_AVERAGING,
        np.array([3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0]),
    ),
    (
        AveragingMethod.CLOSEST_POINT,
        np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]),
    ),
    (
        AveragingMethod.MAX,
        np.array([5.0, 6.0, 6.0, 8.0, 9.0, 9.0, 8.0, 9.0, 9.0]),
    ),
    (
        AveragingMethod.MIN,
        np.array([1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 4.0, 4.0, 5.0]),
    ),
    (
        AveragingMethod.MIN_ABS,
        np.array([1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 4.0, 4.0, 5.0]),
    ),
]


@pytest.mark.parametrize(
    "averaging_method, exp_values",
    cases_averaging_interpolation_mesh2d,
)
def test_averaging_interpolation_mesh2d(
    meshkernel_with_mesh2d: MeshKernel,
    averaging_method: AveragingMethod,
    exp_values: np.ndarray,
):
    """Tests `averaging_interpolation_mesh2d` on the faces of a 4x4 Mesh2d."""

    mk = meshkernel_with_mesh2d(4, 4)

    samples_x = np.array([0.5, 1.5, 2.5, 0.5, 1.5, 2.5, 0.5, 1.5, 2.5], dtype=np.double)
    samples_y = np.array([0.5, 0.5, 0.5, 1.5, 1.5, 1.5, 2.5, 2.5, 2.5], dtype=np.double)
    samples_values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.double)
    samples = GeometryList(samples_x, samples_y, samples_values)

    interpolation = mk.averaging_interpolation_mesh2d(
        samples, Mesh2dLocation.FACES, averaging_method, 1.5, 1
    )

    assert interpolation.x_coordinates[0] == 0.5
    assert interpolation.x_coordinates[1] == 1.5
    assert interpolation.x_coordinates[2] == 2.5
    assert interpolation.x_coordinates[3] == 0.5
    assert interpolation.x_coordinates[4] == 1.5
    assert interpolation.x_coordinates[5] == 2.5
    assert interpolation.x_coordinates[6] == 0.5
    assert interpolation.x_coordinates[7] == 1.5
    assert interpolation.x_coordinates[8] == 2.5

    assert interpolation.y_coordinates[0] == 0.5
    assert interpolation.y_coordinates[1] == 0.5
    assert interpolation.y_coordinates[2] == 0.5
    assert interpolation.y_coordinates[3] == 1.5
    assert interpolation.y_coordinates[4] == 1.5
    assert interpolation.y_coordinates[5] == 1.5
    assert interpolation.y_coordinates[6] == 2.5
    assert interpolation.y_coordinates[7] == 2.5
    assert interpolation.y_coordinates[8] == 2.5

    assert_array_equal(interpolation.values, exp_values)
