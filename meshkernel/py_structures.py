from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum, unique

import numpy as np
from numpy import ndarray


@unique
class DeleteMeshOption(IntEnum):
    """Option to delete the mesh inside a polygon"""

    """Delete all nodes inside the polygon. """
    ALL_NODES = 0

    """ Delete all faces of which the circum center is inside the polygon """
    ALL_FACE_CIRCUMCENTERS = 1

    """ Delete all faces of which the complete face is inside the polygon. """
    ALL_COMPLETE_FACES = 2


@unique
class ProjectToLandBoundaryOption(IntEnum):
    """Option how to project to the land boundary."""

    DO_NOT_PROJECT_TO_LANDBOUNDARY = 0
    TO_ORIGINAL_NETBOUNDARY = 1
    OUTER_MESH_BOUNDARY_TO_LANDBOUNDARY = 2
    INNER_AND_OUTER_MESH_BOUNDARY_TO_LANDBOUNDARY = 3
    WHOLE_MESH = 4


@dataclass
class Mesh2d:
    """This class is used for getting and setting two-dimensional mesh data

    Attributes:
        node_x (ndarray): A 1D double array describing the x-coordinates of the nodes.
        node_y (ndarray): A 1D double array describing the y-coordinates of the nodes.
        edge_nodes (ndarray, optional): A 1D integer array describing the nodes composing each mesh 2d edge.
        face_nodes (ndarray, optional): A 1D integer array describing the nodes composing each mesh 2d face.
        nodes_per_face (ndarray, optional): A 1D integer array describing the nodes composing each mesh 2d face.
        edge_x (ndarray, optional): A 1D double array describing x-coordinates of the mesh edges' middle points.
        edge_y (ndarray, optional): A 1D double array describing x-coordinates of the mesh edges' middle points.
        face_x (ndarray, optional): A 1D double array describing x-coordinates of the mesh faces' mass centers.
        face_y (ndarray, optional): A 1D double array describing y-coordinates of the mesh faces' mass centers.

    """

    node_x: ndarray
    node_y: ndarray
    edge_nodes: ndarray
    face_nodes: ndarray = np.empty(0, dtype=np.int32)
    nodes_per_face: ndarray = np.empty(0, dtype=np.int32)
    edge_x: ndarray = np.empty(0, dtype=np.double)
    edge_y: ndarray = np.empty(0, dtype=np.double)
    face_x: ndarray = np.empty(0, dtype=np.double)
    face_y: ndarray = np.empty(0, dtype=np.double)


@dataclass
class GeometryList:
    """A class to describe a list of geometries.

    Attributes:
        x_coordinates (ndarray): A 1D double array describing the x-coordinates of the nodes.
        y_coordinates (ndarray): A 1D double array describing the y-coordinates of the nodes.
        values (ndarray, optional): A 1D double array describing the values of the nodes.
        geometry_separator (float, optional): The value used as a separator in the coordinates. Default is `-999.0`
        inner_outer_separator (float, optional): The value used to separate the inner part of a polygon from its outer
                                                 part. Default is `-998.0`
    """

    x_coordinates: ndarray
    y_coordinates: ndarray
    values: ndarray = np.empty(0, dtype=np.double)
    geometry_separator: float = -999.0
    inner_outer_separator: float = -998.0


@dataclass
class OrthogonalizationParameters:
    """A class holding the parameters for orthogonalization.

    Attributes:
        outer_iterations (int): Number of outer iterations in orthogonalization. Increase this parameter for complex
                                grids..
        boundary_iterations (int): Number of boundary iterations in grid/net orthogonalization within itatp.
        inner_iterations (int): Number of inner iterations in grid/net orthogonalization within itbnd.
        orthogonalization_to_smoothing_factor (float): Factor from 0 to 1. between grid smoothing and grid
                                                       orthogonality.
        orthogonalization_to_smoothing_factor_at_boundary (float): Minimum ATPF on the boundary.
        areal_to_angle_smoothing_factor (float): Factor between smoother 1d0 and area-homogenizer 0d0.
    """

    outer_iterations: int
    boundary_iterations: int
    inner_iterations: int
    orthogonalization_to_smoothing_factor: float
    orthogonalization_to_smoothing_factor_at_boundary: float
    areal_to_angle_smoothing_factor: float
