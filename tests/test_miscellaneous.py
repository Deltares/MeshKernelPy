import numpy as np
import pytest

from meshkernel import GeometryList, MeshKernel

cases_get_splines = [
    (
        20,  # number_of_points_between_nodes
        np.array([10.0, 20.0, 30.0], dtype=np.double),  # x_coordinates
        np.array([-5.0, 5.0, -5.0], dtype=np.double),  # y_coordinates
    ),
    (
        1,  # number_of_points_between_nodes
        np.array([10.0, 20.0, 30.0], dtype=np.double),  # x_coordinates
        np.array([-5.0, 5.0, -5.0], dtype=np.double),  # y_coordinates
    ),
    (
        10000,  # number_of_points_between_nodes
        np.array([-5.0, 0.0, 5.0], dtype=np.double),  # x_coordinates
        np.array([-5.0, 100.0, 5.0], dtype=np.double),  # y_coordinates
    ),
]


@pytest.mark.parametrize(
    "number_of_points_between_nodes, x_coordinates, y_coordinates", cases_get_splines
)
def test_get_splines(
    number_of_points_between_nodes: int,
    x_coordinates: np.ndarray,
    y_coordinates: np.ndarray,
):
    """Test `get_splines` by checking if the dimensions of the generated spline are correct"""
    mk = MeshKernel()
    geometry_list_in = GeometryList(x_coordinates, y_coordinates)

    geometry_list_out = mk.get_splines(geometry_list_in, number_of_points_between_nodes)

    original_number_of_coordinates = geometry_list_in.x_coordinates.size
    expected_new_number_of_coordinates = (
        original_number_of_coordinates * number_of_points_between_nodes
        - number_of_points_between_nodes
        + original_number_of_coordinates
        + 1
    )

    assert expected_new_number_of_coordinates == geometry_list_out.x_coordinates.size
