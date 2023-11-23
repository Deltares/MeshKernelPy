from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, unique

import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.patches import Polygon
from numpy import ndarray

import meshkernel.errors as mk_errors
from meshkernel.utils import plot_edges


@unique
class DeleteMeshOption(IntEnum):
    """Option to delete the mesh inside a polygon."""

    """Deletes mesh inside and not intersected """
    INSIDE_NOT_INTERSECTED = 0

    """Deletes mesh inside and intersected """
    INSIDE_AND_INTERSECTED = 1


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

    """Refinement that ensures that the courant criteria is not exceeded considering the sample depths."""
    WAVE_COURANT = 1

    """Refinement that refines recursively a fixed number of times."""
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


@unique
class ProjectionType(IntEnum):
    """The projection types."""

    CARTESIAN = 0
    SPHERICAL = 1
    SPHERICALACCURATE = 2


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
        edge_faces (ndarray, optional): A 1D integer array describing for each edge the indices of the faces.
        face_edges (ndarray, optional): A 1D integer array describing for each face the indices of the edges.

    """

    def __init__(
        self,
        node_x=np.empty(0, dtype=np.double),
        node_y=np.empty(0, dtype=np.double),
        edge_nodes=np.empty(0, dtype=np.int32),
        face_nodes=np.empty(0, dtype=np.int32),
        nodes_per_face=np.empty(0, dtype=np.int32),
        edge_x=np.empty(0, dtype=np.double),
        edge_y=np.empty(0, dtype=np.double),
        face_x=np.empty(0, dtype=np.double),
        face_y=np.empty(0, dtype=np.double),
        edge_faces=np.empty(0, dtype=np.int32),
        face_edges=np.empty(0, dtype=np.int32),
    ):
        self.node_x: ndarray = node_x
        self.node_y: ndarray = node_y
        self.edge_nodes: ndarray = edge_nodes
        self.face_nodes: ndarray = face_nodes
        self.nodes_per_face: ndarray = nodes_per_face
        self.edge_x: ndarray = edge_x
        self.edge_y: ndarray = edge_y
        self.face_x: ndarray = face_x
        self.face_y: ndarray = face_y
        self.edge_faces: ndarray = edge_faces
        self.face_edges: ndarray = face_edges

    def __eq__(self, other: Mesh2d):
        return (
            (self.node_x == other.node_x).all()
            and np.array_equal(self.node_y, other.node_y)
            and np.array_equal(self.face_x, other.face_x)
            and np.array_equal(self.face_y, other.face_y)
            and np.array_equal(self.edge_x, other.edge_x)
            and np.array_equal(self.edge_y, other.edge_y)
            and np.array_equal(self.face_edges, other.face_edges)
            and np.array_equal(self.face_nodes, other.face_nodes)
            and np.array_equal(self.edge_faces, other.edge_faces)
            and np.array_equal(self.edge_nodes, other.edge_nodes)
            and np.array_equal(self.nodes_per_face, other.nodes_per_face)
        )

    def almost_equal(
        self, other: Mesh2d, rel_tol: float = 1.0e-5, abs_tol: float = 1.0e-8
    ):
        return (
            np.allclose(self.node_x, other.node_x, rel_tol, abs_tol)
            and np.allclose(self.node_y, other.node_y, rel_tol, abs_tol)
            and np.allclose(self.face_x, other.face_x, rel_tol, abs_tol)
            and np.allclose(self.face_y, other.face_y, rel_tol, abs_tol)
            and np.allclose(self.edge_x, other.edge_x, rel_tol, abs_tol)
            and np.allclose(self.edge_y, other.edge_y, rel_tol, abs_tol)
            and np.array_equal(self.face_edges, other.face_edges)
            and np.array_equal(self.face_nodes, other.face_nodes)
            and np.array_equal(self.edge_faces, other.edge_faces)
            and np.array_equal(self.edge_nodes, other.edge_nodes)
            and np.array_equal(self.nodes_per_face, other.nodes_per_face)
        )

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

        n = len(self.nodes_per_face)
        m = self.nodes_per_face.max()
        face_node_connectivity = np.full((n, m), -1)
        is_node = (np.tile(np.arange(m), n).reshape((n, m)).T < self.nodes_per_face).T
        face_node_connectivity[is_node] = self.face_nodes

        node_xy = np.column_stack((self.node_x, self.node_y))
        vertices = node_xy[face_node_connectivity]
        vertices[~is_node] = np.nan

        collection = PolyCollection(vertices)

        ax.add_collection(collection)

        # Ensure that you can see the full mesh
        x_min = self.node_x.min()
        x_max = self.node_x.max()
        y_min = self.node_y.min()
        y_max = self.node_y.max()
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)


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

    def __init__(
        self,
        x_coordinates=np.empty(0, dtype=np.double),
        y_coordinates=np.empty(0, dtype=np.double),
        values=np.empty(0, dtype=np.double),
        geometry_separator=-999.0,
        inner_outer_separator=-998.0,
    ):
        self.x_coordinates: ndarray = x_coordinates
        self.y_coordinates: ndarray = y_coordinates
        self.values: ndarray = values
        self.geometry_separator: float = geometry_separator
        self.inner_outer_separator: float = inner_outer_separator

        if len(self.x_coordinates) != len(self.y_coordinates):
            raise mk_errors.InputError(
                "The length of x_coordinates is not equal to the length of y_coordinates"
            )

        if len(self.values) > 0 and len(self.values) != len(self.x_coordinates):
            raise mk_errors.InputError(
                "The length of values is not equal to the length of x_coordinates"
            )


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

    def __init__(
        self,
        outer_iterations=2,
        boundary_iterations=25,
        inner_iterations=25,
        orthogonalization_to_smoothing_factor=0.975,
        orthogonalization_to_smoothing_factor_at_boundary=1.0,
        areal_to_angle_smoothing_factor=1.0,
    ):
        self.outer_iterations: int = outer_iterations
        self.boundary_iterations: int = boundary_iterations
        self.inner_iterations: int = inner_iterations
        self.orthogonalization_to_smoothing_factor: float = (
            orthogonalization_to_smoothing_factor
        )
        self.orthogonalization_to_smoothing_factor_at_boundary: float = (
            orthogonalization_to_smoothing_factor_at_boundary
        )
        self.areal_to_angle_smoothing_factor: float = areal_to_angle_smoothing_factor


class CurvilinearGrid:
    """This class is used for getting and setting curvilinear grid data.

    Attributes:
        node_x (ndarray): A 1D double array describing the x-coordinates of the nodes.
        node_y (ndarray): A 1D double array describing the y-coordinates of the nodes.
        num_m (int): The number of curvilinear grid nodes along m.
        num_n (int): The number of curvilinear grid nodes along n.
    """

    def __init__(self, node_x, node_y, num_m, num_n):
        self.node_x: ndarray = node_x
        self.node_y: ndarray = node_y
        self.num_m: int = num_m
        self.num_n: int = num_n

    def plot_edges(self, ax, *args, **kwargs):
        """Plots the edges at a given axes.
        `args` and `kwargs` will be used as parameters of the `plot` method of matplotlib.

        Args:
            ax (matplotlib.axes.Axes): The axes where to plot the edges
        """

        # construct the edges
        node_indices = np.fromiter(
            (int(x) for x in range(self.num_m * self.num_n)), int
        )
        node_indices = node_indices.reshape((self.num_m, self.num_n))

        invalid_value = -999.0
        edge_nodes = np.zeros(
            (self.num_m * (self.num_n - 1) + (self.num_m - 1) * self.num_n) * 2,
            dtype=np.int_,
        )
        index = 0
        for m in range(self.num_m - 1):
            for n in range(self.num_n):
                if (
                    self.node_x[node_indices[m][n]] != invalid_value
                    and self.node_x[node_indices[m + 1][n]] != invalid_value
                ):
                    edge_nodes[index] = node_indices[m][n]
                    index += 1
                    edge_nodes[index] = node_indices[m + 1][n]
                    index += 1

        for m in range(self.num_m):
            for n in range(self.num_n - 1):
                if (
                    self.node_x[node_indices[m][n]] != invalid_value
                    and self.node_x[node_indices[m][n + 1]] != invalid_value
                ):
                    edge_nodes[index] = node_indices[m][n]
                    index += 1
                    edge_nodes[index] = node_indices[m][n + 1]
                    index += 1

        edge_nodes = np.resize(edge_nodes, index)
        plot_edges(self.node_x, self.node_y, edge_nodes, ax, *args, **kwargs)


class CurvilinearParameters:
    """A class holding the parameters for generating a curvilinear grid from splines.

    Attributes:
        m_refinement (int, optional): M-refinement factor for regular grid generation. Default is `2000`.
        n_refinement (int, optional): N-refinement factor for regular grid generation. Default is `40`.
        smoothing_iterations (int, optional): Nr. of inner iterations in regular grid smoothing. Default is `25`.
        smoothing_parameter (float, optional): Smoothing parameter. Default is `0.5`.
        attraction_parameter (float, optional): Attraction/repulsion parameter. Default is `0.0`.
    """

    def __init__(
        self,
        m_refinement=2000,
        n_refinement=40,
        smoothing_iterations=10,
        smoothing_parameter=0.5,
        attraction_parameter=0.0,
    ):
        self.m_refinement: int = m_refinement
        self.n_refinement: int = n_refinement
        self.smoothing_iterations: int = smoothing_iterations
        self.smoothing_parameter: float = smoothing_parameter
        self.attraction_parameter: float = attraction_parameter


class SplinesToCurvilinearParameters:
    """A class holding the additional parameters required for generating a curvilinear grid from splines
    using the advancing front method.

    Attributes:
        aspect_ratio (float, optional): Aspect ratio. Default is `0.1`.
        aspect_ratio_grow_factor (float, optional): Grow factor of aspect ratio. Default is `1.1`.
        average_width (float, optional): Average mesh width on center spline. Default is `0.005`.
        curvature_adapted_grid_spacing (int, optional): Curvature adapted grid spacing. Default is `1`.
        grow_grid_outside (int, optional): Grow the grid outside the prescribed grid height. Default is `0`.
        maximum_num_faces_in_uniform_part (int, optional): Maximum number of layers in the uniform part Default is `5`.
        nodes_on_top_of_each_other_tolerance (float, optional): On-top-of-each-other tolerance.). Default is `0.0001`.
        min_cosine_crossing_angles (float, optional): Minimum allowed absolute value of crossing-angle cosine.
        Default is `0.95`.
        check_front_collisions (int, optional): Check for collisions with other parts of the front. Default is `0`.
        remove_skinny_triangles (int, optional): Check for collisions with other parts of the front. Default is `1`.
    """

    def __init__(
        self,
        aspect_ratio=0.1,
        aspect_ratio_grow_factor=1.1,
        average_width=500.0,
        curvature_adapted_grid_spacing=1,
        grow_grid_outside=0,
        maximum_num_faces_in_uniform_part=5,
        nodes_on_top_of_each_other_tolerance=0.0001,
        min_cosine_crossing_angles=0.95,
        check_front_collisions=0,
        remove_skinny_triangles=1,
    ):
        self.aspect_ratio: float = aspect_ratio
        self.aspect_ratio_grow_factor: float = aspect_ratio_grow_factor
        self.average_width: float = average_width
        self.curvature_adapted_grid_spacing: int = curvature_adapted_grid_spacing
        self.grow_grid_outside: int = grow_grid_outside
        self.maximum_num_faces_in_uniform_part: int = maximum_num_faces_in_uniform_part
        self.nodes_on_top_of_each_other_tolerance: float = (
            nodes_on_top_of_each_other_tolerance
        )
        self.min_cosine_crossing_angles: float = min_cosine_crossing_angles
        self.check_front_collisions: int = check_front_collisions
        self.remove_skinny_triangles: int = remove_skinny_triangles


class MeshRefinementParameters:
    """A class holding the parameters for Mesh2d refinement.

    Attributes:
        refine_intersected (bool): Whether to compute faces intersected by polygon. Default is `False`.
        use_mass_center_when_refining (bool): Whether to use the mass center when splitting a face in the refinement
                                              process. Default is `True`.
        min_edge_size (float): Minimum edge size. Default is `0.5`.
        refinement_type (RefinementType): Refinement criterion type. Default is `RefinementType.REFINEMENT_LEVELS`.
        connect_hanging_nodes (bool): Whether to connect hanging nodes at the end of the iteration. Default is `True`.
        account_for_samples_outside (bool): Whether to take samples outside face into account.  Default is `False`.
        max_refinement_iterations (int, optional): Maximum number of refinement iterations. Default is `10`.
        smoothing_iterations (int, optional): The number of smoothing iterations. Default is `5`.
        max_courant_time (double, optional): Maximum courant time in seconds. Default is `120`.
        directional_refinement (bool, optional): Directional refinement, cannot be used when the number of smoothing
        iterations is larger than 0. Default is `False`.
    """

    def __init__(
        self,
        refine_intersected=False,
        use_mass_center_when_refining=True,
        min_edge_size=0.5,
        refinement_type=RefinementType.REFINEMENT_LEVELS,
        connect_hanging_nodes=True,
        account_for_samples_outside_face=False,
        max_refinement_iterations=10,
        smoothing_iterations=5,
        max_courant_time=120.0,
        directional_refinement=False,
    ):
        self.refine_intersected: bool = refine_intersected
        self.use_mass_center_when_refining: bool = use_mass_center_when_refining
        self.min_edge_size: float = min_edge_size
        self.refinement_type: RefinementType = refinement_type
        self.connect_hanging_nodes: bool = connect_hanging_nodes
        self.account_for_samples_outside_face: bool = account_for_samples_outside_face
        self.max_refinement_iterations: int = max_refinement_iterations
        self.smoothing_iterations: int = smoothing_iterations
        self.max_courant_time: float = max_courant_time
        self.directional_refinement: bool = directional_refinement


class MakeGridParameters:
    """A class holding the necessary parameters to create a new curvilinear grid in a C-compatible manner.

    Attributes:
        num_columns (int, optional): The number of columns in x direction. Default is `3`.
        num_rows (int, optional): The number of columns in y direction. Default is `3`.
        angle (float, optional): The grid angle. Default is `0.0`.
        origin_x (float, optional): The x coordinate of the origin, located at the bottom left corner.
        Default is `0.0`.
        origin_y (float, optional): The y coordinate of the origin, located at the bottom left corner.
        Default is `0.0`.
        block_size_x (float, optional): The grid block size in x dimension, used only for squared grids.
        Default is `10.0`.
        block_size_y (float, optional): The grid block size in y dimension, used only for squared grids.
        Default is `10.0`.
        upper_right_x (float, optional): The x coordinate of the upper right corner.
        Default is `0.0`.
        upper_right_y (float, optional): The y coordinate of the upper right corner.
        Default is `0.0`.
    """

    def __init__(
        self,
        num_columns=3,
        num_rows=3,
        angle=0.0,
        origin_x=0.0,
        origin_y=0.0,
        block_size_x=10.0,
        block_size_y=10.0,
        upper_right_x=0.0,
        upper_right_y=0.0,
    ):
        self.num_columns: int = num_columns
        self.num_rows: int = num_rows
        self.angle: float = angle
        self.origin_x: float = origin_x
        self.origin_y: float = origin_y
        self.block_size_x: float = block_size_x
        self.block_size_y: float = block_size_y
        self.upper_right_x: float = upper_right_x
        self.upper_right_y: float = upper_right_y


class Mesh1d:
    """This class is used for getting and setting one-dimensional mesh data.

    Attributes:
        node_x (ndarray): A 1D double array describing the x-coordinates of the nodes.
        node_y (ndarray): A 1D double array describing the y-coordinates of the nodes.
        edge_nodes (ndarray, optional): A 1D integer array describing the nodes composing each mesh edge.
    """

    def __init__(self, node_x, node_y, edge_nodes):
        self.node_x: ndarray = node_x
        self.node_y: ndarray = node_y
        self.edge_nodes: ndarray = edge_nodes

    def plot_edges(self, ax, *args, **kwargs):
        """Plots the edges at a given axes.
        `args` and `kwargs` will be used as parameters of the `plot` method of matplotlib.

        Args:
            ax (matplotlib.axes.Axes): The axes where to plot the edges
        """
        plot_edges(self.node_x, self.node_y, self.edge_nodes, ax, *args, **kwargs)


class Contacts:
    """This class describes the contacts between a mesh1d and mesh2d.

    Attributes:
        mesh1d_indices (ndarray): A 1D integer array describing the mesh1d node indices.
        mesh2d_indices (ndarray): A 1D integer array describing the mesh2d face indices.
    """

    def __init__(self, mesh1d_indices, mesh2d_indices):
        self.mesh1d_indices: ndarray = mesh1d_indices
        self.mesh2d_indices: ndarray = mesh2d_indices

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


class GriddedSamples:
    """A class holding gridded samples, both for uniform gridding and non-uniform gridding.

    Attributes:
        num_x (int, optional): Number of x gridded samples coordinates. The number of grid points is num_x * num_y.
        Default is `0`.
        num_y (int, optional): Number of y gridded samples coordinates. The number of grid points is num_x * num_y.
        Default is `0`.
        x_origin (float, optional): X coordinate of the grid origin. Default is `0.0`.
        y_origin (float, optional): Y coordinate of the grid origin. Default is `0.0`.
        cell_size (float, optional): Constant grid edge size. Default is `0.0`.
        x_coordinates (ndarray, optional): Coordinates for non-uniform grid spacing in x direction.
        y_coordinates (ndarray, optional): Coordinates for non-uniform grid spacing in y direction.
        values (ndarray): Sample values.
    """

    def __init__(
        self,
        num_x=0,
        num_y=0,
        x_origin=0.0,
        y_origin=0.0,
        cell_size=0.0,
        x_coordinates=np.empty(0, dtype=np.double),
        y_coordinates=np.empty(0, dtype=np.double),
        values=np.empty(0, dtype=np.double),
    ):
        self.num_x: int = num_x
        self.num_y: int = num_y
        self.x_origin: float = x_origin
        self.y_origin: float = y_origin
        self.cell_size: float = cell_size
        self.x_coordinates: ndarray = x_coordinates
        self.y_coordinates: ndarray = y_coordinates
        self.values: ndarray = values
