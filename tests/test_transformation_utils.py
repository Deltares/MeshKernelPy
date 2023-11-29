import numpy as np
import pytest
import transformation_utils
from numpy.testing import assert_almost_equal, assert_array_equal


def test_rotate_point():
    """Tests `translate` applied to a single point"""
    point = [1.0, 2.0]
    origin = [0.0, 0.0]
    angle = -30.0
    point_rotated = transformation_utils.rotate(point, origin, angle)
    new_angle = np.deg2rad(angle) + np.arctan(point[1] / point[0])
    norm = np.sqrt(point[0] * point[0] + point[1] * point[1])
    assert_almost_equal(point_rotated[0], norm * np.cos(new_angle))
    assert_almost_equal(point_rotated[1], norm * np.sin(new_angle))


def test_translate_point():
    """Tests `translate` applied to a single point"""
    point = [1.0, 2.0]
    translation = [3.0, 4.0]
    point_translated = transformation_utils.translate(point, translation)
    assert point_translated[0] == point[0] + translation[0]
    assert point_translated[1] == point[1] + translation[1]


def test_translate_points():
    """Tests `translate` applied to several points"""
    point_x = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.double)
    point_y = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.double)
    translation = [3.0, 4.0]
    translated_point_x, translated_point_y = transformation_utils.translate(
        [point_x, point_y], translation
    )
    assert_array_equal(point_x + translation[0], translated_point_x)
    assert_array_equal(point_y + translation[1], translated_point_y)
