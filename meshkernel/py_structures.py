from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum, unique

import numpy as np


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


@unique
class RefinementType(IntEnum):
    """Refinement type"""

    RIDGE_REFINEMENT = 1
    WAVE_COURANT = 2
    REFINEMENT_LEVELS = 3


@dataclass
class Mesh2d:
    """This class is used for getting and setting two-dimensional mesh data

    Attributes:
        node_x (np.ndarray): A 1D double array describing the x-coordinates of the nodes.
        node_y (np.ndarray): A 1D double array describing the y-coordinates of the nodes.
        edge_nodes (np.ndarray, optional): A 1D integer array describing the nodes composing each mesh 2d edge.
        face_nodes (np.ndarray, optional): A 1D integer array describing the nodes composing each mesh 2d face.
        nodes_per_face (np.ndarray, optional): A 1D integer array describing the nodes composing each mesh 2d face.
        edge_x (np.ndarray, optional): A 1D double array describing x-coordinates of the mesh edges' middle points.
        edge_y (np.ndarray, optional): A 1D double array describing x-coordinates of the mesh edges' middle points.
        face_x (np.ndarray, optional): A 1D double array describing x-coordinates of the mesh faces' mass centers.
        face_y (np.ndarray, optional): A 1D double array describing y-coordinates of the mesh faces' mass centers.

    """

    node_x: np.ndarray
    node_y: np.ndarray
    edge_nodes: np.ndarray
    face_nodes: np.ndarray = np.empty(0, dtype=np.int32)
    nodes_per_face: np.ndarray = np.empty(0, dtype=np.int32)
    edge_x: np.ndarray = np.empty(0, dtype=np.double)
    edge_y: np.ndarray = np.empty(0, dtype=np.double)
    face_x: np.ndarray = np.empty(0, dtype=np.double)
    face_y: np.ndarray = np.empty(0, dtype=np.double)


@dataclass
class GeometryList:
    """A class to describe a list of geometries.

    Attributes:
        x_coordinates (np.ndarray): A 1D double array describing the x-coordinates of the nodes.
        y_coordinates (np.ndarray): A 1D double array describing the y-coordinates of the nodes.
        values (np.ndarray, optional): A 1D double array describing the values of the nodes.
        geometry_separator (float, optional): The value used as a separator in the coordinates. Default is `-999.0`
        inner_outer_separator (float, optional): The value used to separate the inner part of a polygon from its outer
                                                 part. Default is `-998.0`
    """

    x_coordinates: np.ndarray
    y_coordinates: np.ndarray
    values: np.ndarray = np.empty(0, dtype=np.double)
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


@dataclass
class InterpolationParameters:
    """A class holding the parameters for interpolation.

    Attributes:
        max_refinement_iterations (int): Maximum number of refinement iterations, set to 1 if only one refinement is
                                         wanted.
        averaging_method (int): Averaging method : 1 = simple averaging, 2 = closest point, 3 = max, 4 = min,
                                5 = inverse weighted distance, 6 = minabs, 7 = kdtree.
        minimum_points (int): Minimum number of points needed inside cell to handle the cell
        relative_search_radius (float): Relative search cell size, 1 = actual cell size, 2 = twice as large,
                                        search radius can be larger than cell so more sample are included.
        interpolate_to (int): Interpolation settings, 1 = bathy, 2 = zk, 3 = s1, 4 = Zc.
        refine_intersected (bool): Whether to compute faces intersected by polygon.
        use_mass_center_when_refining (bool): Whether to use the mass center when splitting a face in the refinement
                                              process.
    """

    max_refinement_iterations: int
    averaging_method: int
    minimum_points: int
    relative_search_radius: float
    interpolate_to: int
    refine_intersected: bool
    use_mass_center_when_refining: bool


@dataclass
class SampleRefineParameters:
    """A class holding the parameters for sample refinement.

    Attributes:
        max_refinement_iterations (int): Maximum number of refinement iterations.
        min_face_size (float): Minimum cell size.
        refinement_type (RefinementType): Refinement criterion type.
        connect_hanging_nodes (bool): Whether to connect hanging nodes at the end of the iteration.
        max_time_step (float): Maximum time-step in Courant grid.
        account_for_samples_outside (bool): Whether to take samples outside face into account.
    """

    max_refinement_iterations: int
    min_face_size: float
    refinement_type: RefinementType
    connect_hanging_nodes: bool
    max_time_step: float
    account_for_samples_outside_face: bool
