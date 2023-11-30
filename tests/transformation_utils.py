import numpy as np


def rotate(point, origin, angle):
    """
    Rotate a point by a given angle about a given origin.

    Args:
    point: The point to rotate
    origin: The point about which the point is to be rotated
    angle: The angle in degrees by which the point is to be rotated
    """
    point_x, point_y = point
    origin_x, origin_y = origin
    angle_rad = np.deg2rad(angle)
    angle_cos = np.cos(angle_rad)
    angle_sin = np.sin(angle_rad)
    offset_x = point_x - origin_x
    offset_y = point_y - origin_y
    point_rot_x = origin_x + angle_cos * offset_x - angle_sin * offset_y
    point_rot_y = origin_y + angle_sin * offset_x + angle_cos * offset_y
    return point_rot_x, point_rot_y


def translate(point, translation):
    """
    Translate a point

    Args:
    point: The point to translate
    translation: The translation vector
    """
    point_x, point_y = point
    translation_x, translation_y = translation
    return point_x + translation_x, point_y + translation_y
