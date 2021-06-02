from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, unique

import numpy as np
from numpy import ndarray

from meshkernel.utils import plot_edges


@unique
class DeleteMeshOption(IntEnum):
    """Option to delete the mesh inside a polygon."""

    """Delete all nodes inside the polygon. """
    ALL_NODES = 0

    """ Delete all faces of which the circum center is inside the polygon. """
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

    WAVE_COURANT = 1
    REFINEMENT_LEVELS = 2


@unique
class Mesh2dLocation(IntEnum):
    """The Mesh2d location types."""

    FACES = 0
    NODES = 1
    EDGES = 2


@unique
class AveragingMethod(IntEnum):
    """The averaging methods."""

    """Computes a simple mean. """
    SIMPLE_AVERAGING = 1

    """Takes the value of the closest sample to the interpolation location. """
    CLOSEST_POINT = 2

    """Takes the maximum sample value. """
    MAX = 3

    """Takes the minimum sample value. """
    MIN = 4

    """Computes the inverse weighted sample mean. """
    INVERSE_WEIGHT_DISTANCE = 5

    """Computes the minimum absolute value. """
    MIN_ABS = 6


@dataclass
class Mesh2d:
    """This class is used for getting and setting two-dimensional mesh data.

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

    def plot_edges(self, ax, *args, **kwargs):
        """Plots the edges at a given axes.
        `args` and `kwargs` will be used as parameters of the `plot` method of matplotlib.

        Args:
            ax (matplotlib.axes.Axes): The axes where to plot the edges
        """
        plot_edges(self.node_x, self.node_y, self.edge_nodes, ax, *args, **kwargs)

    def plot_faces(self, ax, *args, **kwargs):
        """Plots the faces at a given axes.
        `args` and `kwargs` will be used as parameters of the `plot` method of matplotlib.

        Args:
            ax (matplotlib.axes.Axes): The axes where to plot the faces
        """
        node_position = 0
        for num_nodes in self.nodes_per_face:
            # Calculate values to draw
            face_nodes = self.face_nodes[node_position : (node_position + num_nodes)]
            face_nodes_x = self.node_x[face_nodes]
            face_nodes_y = self.node_y[face_nodes]
            node_position += num_nodes

            # Draw polygon
            ax.fill(face_nodes_x, face_nodes_y, *args, **kwargs)


@dataclass
class GeometryList:
    """A class to describe a list of geometries.

    Attributes:
        x_coordinates (ndarray): A 1D double array describing the x-coordinates of the nodes.
        y_coordinates (ndarray): A 1D double array describing the y-coordinates of the nodes.
        values (ndarray, optional): A 1D double array describing the values of the nodes.
        geometry_separator (float, optional): The value used as a separator in the coordinates. Default is `-999.0`.
        inner_outer_separator (float, optional): The value used to separate the inner part of a polygon from its outer
                                                 part. Default is `-998.0`.
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
        outer_iterations (int, optional): Number of outer iterations in orthogonalization. Increase this parameter
                                          for complex grids. Default is `2`.
        boundary_iterations (int, optional): Number of boundary iterations in grid/net orthogonalization within itatp.
                                             Default is `25`.
        inner_iterations (int, optional): Number of inner iterations in grid/net orthogonalization within itbnd.
                                          Default is `25`.
        orthogonalization_to_smoothing_factor (float, optional): Factor from between grid smoothing (0) and
                                                                 grid orthogonality (1). Default is `0.975`.
        orthogonalization_to_smoothing_factor_at_boundary (float, optional): Minimum ATPF on the boundary.
                                                                             Default is `1.0`.
        areal_to_angle_smoothing_factor (float, optional): Factor between smoother 1d0 and area-homogenizer 0d0.
                                                           Default is `1.0`.
    """

    outer_iterations: int = 2
    boundary_iterations: int = 25
    inner_iterations: int = 25
    orthogonalization_to_smoothing_factor: float = 0.975
    orthogonalization_to_smoothing_factor_at_boundary: float = 1.0
    areal_to_angle_smoothing_factor: float = 1.0


@dataclass
class MeshRefinementParameters:
    """A class holding the parameters for Mesh2d refinement.

    Attributes:
        refine_intersected (bool): Whether to compute faces intersected by polygon.
        use_mass_center_when_refining (bool): Whether to use the mass center when splitting a face in the refinement
                                              process.
        min_face_size (float): Minimum cell size.
        refinement_type (RefinementType): Refinement criterion type.
        connect_hanging_nodes (bool): Whether to connect hanging nodes at the end of the iteration.
        account_for_samples_outside (bool): Whether to take samples outside face into account.
        max_refinement_iterations (int, optional): Maximum number of refinement iterations. Default is `10`.
    """

    refine_intersected: bool
    use_mass_center_when_refining: bool
    min_face_size: float
    refinement_type: RefinementType
    connect_hanging_nodes: bool
    account_for_samples_outside_face: bool
    max_refinement_iterations: int = 10


@dataclass
class Mesh1d:
    """This class is used for getting and setting one-dimensional mesh data.

    Attributes:
        node_x (ndarray): A 1D double array describing the x-coordinates of the nodes.
        node_y (ndarray): A 1D double array describing the y-coordinates of the nodes.
        edge_nodes (ndarray, optional): A 1D integer array describing the nodes composing each mesh edge.
    """

    node_x: ndarray
    node_y: ndarray
    edge_nodes: ndarray

    def plot_edges(self, ax, *args, **kwargs):
        """Plots the edges at a given axes.
        `args` and `kwargs` will be used as parameters of the `plot` method of matplotlib.

        Args:
            ax (matplotlib.axes.Axes): The axes where to plot the edges
        """
        plot_edges(self.node_x, self.node_y, self.edge_nodes, ax, *args, **kwargs)


@dataclass
class Contacts:
    """This class describes the contacts between a mesh1d and mesh2d.

    Attributes:
        mesh1d_indices (ndarray): A 1D integer array describing the mesh1d node indices.
        mesh2d_indices (ndarray): A 1D integer array describing the mesh2d face indices.
    """

    mesh1d_indices: ndarray
    mesh2d_indices: ndarray

    def plot_edges(self, ax, mesh1d, mesh2d, *args, **kwargs):
        """Plots the edges at a given axes.
        `args` and `kwargs` will be used as parameters of the `plot` method of matplotlib.

        Args:
            ax (matplotlib.axes.Axes): The axes where to plot the edges
            mesh1d (Mesh1d): The mesh1d instance used to plot the contacts
            mesh2d (Mesh2d): The mesh2d instance used to plot the contacts
        """

        for mesh1d_index, mesh2d_index in zip(self.mesh1d_indices, self.mesh2d_indices):
            node_x = [mesh1d.node_x[mesh1d_index], mesh2d.face_x[mesh2d_index]]
            node_y = [mesh1d.node_y[mesh1d_index], mesh2d.face_y[mesh2d_index]]

            ax.plot(node_x, node_y, *args, **kwargs)
