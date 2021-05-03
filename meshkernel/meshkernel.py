import logging
import os
import platform
import sys
from ctypes import CDLL, POINTER, byref, c_bool, c_char_p, c_double, c_int
from enum import Enum, IntEnum, unique
from pathlib import Path
from typing import Callable, Iterable, Tuple

import numpy as np
from numpy import ndarray

from meshkernel.c_structures import (
    CGeometryList,
    CInterpolationParameters,
    CMesh2d,
    CSampleRefineParameters,
)
from meshkernel.errors import InputError, MeshKernelError
from meshkernel.py_structures import (
    DeleteMeshOption,
    GeometryList,
    InterpolationParameters,
    Mesh2d,
    SampleRefineParameters,
)

logger = logging.getLogger(__name__)


@unique
class Status(IntEnum):
    SUCCESS = 0
    EXCEPTION = 1
    INVALID_GEOMETRY = 2


class MeshKernel:
    """This class is the low-level entry point
    for interacting with the MeshKernel library
    """

    def __init__(self, is_geographic: bool = False):
        """Constructor of MeshKernel

        Args:
            is_geographic (bool, optional): [description]. Defaults to False.

        Raises:
            OSError: This gets raised in case MeshKernel is used within an unsupported OS.
        """

        # Determine OS
        if platform.system() == "Windows":
            lib_path = Path(__file__).parent.parent / "lib" / "MeshKernelApi.dll"
        elif platform.system() == "Linux":
            lib_path = Path(__file__).parent.parent / "lib" / "libMeshKernelApi.so"
        else:
            raise OSError("Unsupported operating system")

        # LoadLibraryEx flag: LOAD_WITH_ALTERED_SEARCH_PATH 0x08
        # -> uses the altered search path for resolving dll dependencies
        # `winmode` has no effect while running on Linux or macOS
        self.lib = CDLL(str(lib_path), winmode=0x08)

        self.libname = os.path.basename(lib_path)
        self._allocate_state(is_geographic)

    def __del__(self):
        self._deallocate_state()

    def _allocate_state(self, is_geographic: bool) -> None:
        """Creates a new empty mesh.

        Args:
            isGeographic (bool): Cartesian (False) or spherical (True) mesh
        """

        self._meshkernelid = c_int()
        self._execute_function(
            self.lib.mkernel_allocate_state,
            c_int(is_geographic),
            byref(self._meshkernelid),
        )

    def _deallocate_state(self) -> None:
        """
        Deallocate mesh state.

        This method is called by the destructor and
        should never be called manually
        """

        self._execute_function(
            self.lib.mkernel_deallocate_state,
            self._meshkernelid,
        )

    def _get_separator(self) -> float:
        """Gets the value used in the MeshKernel library as separator and missing value."""

        self.lib.mkernel_get_separator.restype = c_double
        return self.lib.mkernel_get_separator()

    def _get_inner_outer_separator(self) -> float:
        """Gets the value used in the MeshKernel as separator for the inner and outer part of a polygon."""

        self.lib.mkernel_get_inner_outer_separator.restype = c_double
        return self.lib.mkernel_get_inner_outer_separator()

    def set_mesh2d(self, mesh2d: Mesh2d) -> None:
        """Sets the two-dimensional mesh state of the MeshKernel.

        Please note that this involves a copy of the data.

        Args:
            mesh2d (Mesh2d): The input data used for setting the state
        """
        cmesh2d = CMesh2d.from_mesh2d(mesh2d)

        self._execute_function(
            self.lib.mkernel_set_mesh2d, self._meshkernelid, byref(cmesh2d)
        )

    def get_mesh2d(self) -> Mesh2d:
        """Gets the two-dimensional mesh state from the MeshKernel.

        Please note that this involves a copy of the data.

        Returns:
            Mesh2d: A copy of the two-dimensional mesh state
        """
        cmesh2d = CMesh2d()
        self._execute_function(
            self.lib.mkernel_get_dimensions_mesh2d, self._meshkernelid, byref(cmesh2d)
        )

        mesh2d = cmesh2d.allocate_memory()
        self._execute_function(
            self.lib.mkernel_get_data_mesh2d, self._meshkernelid, byref(cmesh2d)
        )

        return mesh2d

    def delete_mesh2d(
        self,
        geometry_list: GeometryList,
        delete_option: DeleteMeshOption,
        invert_deletion: bool,
    ) -> None:
        """Deletes a mesh in a polygon using several options.

        Args:
            geometry_list (GeometryList): The GeometryList describing the polygon where to perform the operation.
            delete_option (DeleteMeshOption): The option describing the strategy to delete the mesh.
            invert_deletion (bool): Whether or not to invert the deletion.
        """

        c_geometry_list = CGeometryList.from_geometrylist(geometry_list)

        self._execute_function(
            self.lib.mkernel_delete_mesh2d,
            self._meshkernelid,
            byref(c_geometry_list),
            c_int(delete_option),
            c_bool(invert_deletion),
        )

    def insert_edge_mesh2d(self, start_node: int, end_node: int) -> int:
        """Insert a new mesh2d edge connecting two given nodes.

        Args:
            start_node (int): The index of the first node.
            end_node (int): The index of the second node.

        Returns:
            int: The index of the new edge.
        """

        edge_index = c_int()
        self._execute_function(
            self.lib.mkernel_insert_edge_mesh2d,
            self._meshkernelid,
            c_int(start_node),
            c_int(end_node),
            byref(edge_index),
        )

        return edge_index.value

    def insert_node_mesh2d(self, x: float, y: float) -> int:
        """Insert a new node at the specified coordinates

        Args:
            x (float): The x-coordinate of the new node
            y (float): The y-coordinate of the new node

        Returns:
            int: The index of the new node
        """

        x_array = np.array([x], dtype=np.double)
        y_array = np.array([y], dtype=np.double)
        geometry_list = GeometryList(x_array, y_array)
        c_geometry_list = CGeometryList.from_geometrylist(geometry_list)
        index = c_int()

        self._execute_function(
            self.lib.mkernel_insert_node_mesh2d,
            self._meshkernelid,
            byref(c_geometry_list),
            byref(index),
        )
        return index.value

    def delete_node_mesh2d(self, node_index: int) -> None:
        """Deletes a Mesh2d node with the given `index`.

        Args:
            node_index (int): The index of the node to be deleted.

        Raises:
            InputError: Raised when `node_index` is smaller than 0.
        """

        if node_index < 0:
            raise InputError("node_index needs to be a positive integer")

        self._execute_function(
            self.lib.mkernel_delete_node_mesh2d, self._meshkernelid, c_int(node_index)
        )

    def move_node_mesh2d(self, x: float, y: float, node_index: int) -> None:
        """Moves a Mesh2d node with the given `index` to the .

        Args:
            x (float): The x-coordinate of the new position of the node
            y (float): The y-coordinate of the new position of the node
            node_index (int): The index of the node to be moved.

        Raises:
            InputError: Raised when `node_index` is smaller than 0.
        """

        if node_index < 0:
            raise InputError("node_index needs to be a positive integer")

        x_array = np.array([x], dtype=np.double)
        y_array = np.array([y], dtype=np.double)
        geometry_list = GeometryList(x_array, y_array)
        c_geometry_list = CGeometryList.from_geometrylist(geometry_list)

        self._execute_function(
            self.lib.mkernel_move_node_mesh2d,
            self._meshkernelid,
            byref(c_geometry_list),
            c_int(node_index),
        )

    def delete_edge_mesh2d(self, geometry_list: GeometryList) -> None:
        """Deletes the closest mesh2d edge to a point.
        The coordinates of the edge middle points are used for calculating the distances to the point.

        Args:
            geometry_list (GeometryList): A geometry list containing the coordinate of the point.
        """

        c_geometry_list = CGeometryList.from_geometrylist(geometry_list)

        self._execute_function(
            self.lib.mkernel_delete_edge_mesh2d,
            self._meshkernelid,
            byref(c_geometry_list),
        )

    def get_edge_mesh2d(self, geometry_list: GeometryList) -> int:
        """Gets the closest mesh2d edge to a point.

        Args:
            geometry_list (GeometryList): A geometry list containing the coordinate of the point.

        Returns:
            int: The index of the edge
        """

        c_geometry_list = CGeometryList.from_geometrylist(geometry_list)
        index = c_int()

        self._execute_function(
            self.lib.mkernel_get_edge_mesh2d,
            self._meshkernelid,
            byref(c_geometry_list),
            byref(index),
        )

        return index.value

    def get_node_index_mesh2d(
        self, geometry_list: GeometryList, search_radius: float
    ) -> int:
        """Finds the node closest to a point within a given search radius.

        Args:
            geometry_list (GeometryList): A geometry list containing the coordinate of the point.
            search_radius (float): The search radius.

        Returns:
            int: The index of node
        """

        c_geometry_list = CGeometryList.from_geometrylist(geometry_list)
        index = c_int()

        self._execute_function(
            self.lib.mkernel_get_node_index_mesh2d,
            self._meshkernelid,
            byref(c_geometry_list),
            c_double(search_radius),
            byref(index),
        )

        return index.value

    def count_hanging_edges_mesh2d(self) -> int:
        """Count the number of hanging edges in a Mesh2d.
        A hanging edge is an edge where one of the two nodes is not connected.

        Returns:
            int: The number of hanging edges
        """

        count = c_int()
        self._execute_function(
            self.lib.mkernel_count_hanging_edges_mesh2d,
            self._meshkernelid,
            byref(count),
        )
        return count.value

    def delete_hanging_edges_mesh2d(self) -> None:
        """Delete the hanging edges in the Mesh2d.
        A hanging edge is an edge where one of the two nodes is not connected.
        """

        self._execute_function(
            self.lib.mkernel_delete_hanging_edges_mesh2d, self._meshkernelid
        )

    def make_mesh_from_polygon_mesh2d(self, polygon: GeometryList):
        """Generates a triangular mesh2d within a polygon. The size of the triangles is determined from the length of
        the polygon edges.

        Args:
            polygon (GeometryList): The polygon.
        """

        c_geometry_list = CGeometryList.from_geometrylist(polygon)

        self._execute_function(
            self.lib.mkernel_make_mesh_from_polygon_mesh2d,
            self._meshkernelid,
            byref(c_geometry_list),
        )

    def make_mesh_from_samples_mesh2d(self, sample_points: GeometryList):
        """Makes a triangular mesh from a set of samples, triangulating the sample points.

        Args:
            sample_points (GeometryList): The sample points.
        """

        c_geometry_list = CGeometryList.from_geometrylist(sample_points)

        self._execute_function(
            self.lib.mkernel_make_mesh_from_samples_mesh2d,
            self._meshkernelid,
            byref(c_geometry_list),
        )

    def refine_polygon(
        self,
        polygon: GeometryList,
        first_node: int,
        second_node: int,
        target_edge_length: float,
    ) -> GeometryList:
        """Refines the polygon perimeter between two nodes. This interval is refined to achieve a target edge length.

        Args:
            polygon (GeometryList): The input polygon to refine.
            first_node (int): The first index of the refinement interval.
            second_node (int): The second index of the refinement interval.
            target_edge_length (float): The target interval edge length.

        Returns:
            int: The refined polygon.
        """
        c_polygon = CGeometryList.from_geometrylist(polygon)
        c_n_polygon_nodes = c_int()

        self._execute_function(
            self.lib.mkernel_count_refine_polygon,
            self._meshkernelid,
            byref(c_polygon),
            c_int(first_node),
            c_int(second_node),
            c_double(target_edge_length),
            byref(c_n_polygon_nodes),
        )

        c_refined_polygon = CGeometryList()
        c_refined_polygon.n_coordinates = c_n_polygon_nodes.value
        c_refined_polygon.inner_outer_separator = c_double(
            polygon.inner_outer_separator
        )
        c_refined_polygon.geometry_separator = c_double(polygon.geometry_separator)

        refined_polygon = c_refined_polygon.allocate_memory()

        self._execute_function(
            self.lib.mkernel_refine_polygon,
            self._meshkernelid,
            byref(c_polygon),
            c_int(first_node),
            c_int(second_node),
            c_double(target_edge_length),
            byref(c_refined_polygon),
        )

        return refined_polygon

    def refine_based_on_samples_mesh2d(
        self,
        samples: GeometryList,
        interpolation_params: InterpolationParameters,
        sample_refine_params: SampleRefineParameters,
    ):
        """Refines a mesh2d based on samples. Refinement is achieved by successive splits of the face edges.
        The number of successive splits is indicated by the sample value.
        For example:
        - a value of 0 means no split and hence no refinement;
        - a value of 1 means a single split (a quadrilateral face generates 4 faces);
        - a value of 2 two splits (a quadrilateral face generates 16 faces);

        Args:
            samples (GeometryList): The samples.
            interpolation_params (InterpolationParameters): The interpolation parameters.
            sample_refine_params (SampleRefineParameters): The sample refinement parameters.
        """

        c_samples = CGeometryList.from_geometrylist(samples)
        c_interpolation_params = CInterpolationParameters.from_interpolationparameters(
            interpolation_params
        )
        c_sample_refine_params = (
            CSampleRefineParameters.from_samplerefinementparameters(
                sample_refine_params
            )
        )

        self._execute_function(
            self.lib.mkernel_refine_based_on_samples_mesh2d,
            self._meshkernelid,
            byref(c_samples),
            byref(c_interpolation_params),
            byref(c_sample_refine_params),
        )

    def refine_based_on_polygon_mesh2d(
        self,
        polygon: GeometryList,
        interpolation_params: InterpolationParameters,
    ):
        """Refines a mesh2d within a polygon. Refinement is achieved by splitting the edges contained in the polygon in two.

        Args:
            samples (GeometryList): The closed polygon.
            interpolation_params (InterpolationParameters): The interpolation parameters.
        """

        c_polygon = CGeometryList.from_geometrylist(polygon)
        c_interpolation_params = CInterpolationParameters.from_interpolationparameters(
            interpolation_params
        )

        self._execute_function(
            self.lib.mkernel_refine_based_on_polygon_mesh2d,
            self._meshkernelid,
            byref(c_polygon),
            byref(c_interpolation_params),
        )

    def get_points_in_polygon(
        self, selecting_polygon: GeometryList, selected_polygon: GeometryList
    ) -> GeometryList:
        """Selects the polygon points within another polygon.

        Args:
            selecting_polygon (GeometryList): The selection polygon.
            selected_polygon (GeometryList): The polygon of which to get the selected points.

        Returns:
            GeometryList: The selection result. The selected points are contained in the values array of the returned
                          GeometryList (0.0 not selected, 1.0 selected).
        """

        c_selecting_polygon = CGeometryList.from_geometrylist(selecting_polygon)
        c_selected_polygon = CGeometryList.from_geometrylist(selected_polygon)
        c_selection = CGeometryList.from_geometrylist(selected_polygon)

        selection = c_selection.allocate_memory()

        self._execute_function(
            self.lib.mkernel_get_points_in_polygon,
            self._meshkernelid,
            byref(c_selecting_polygon),
            byref(c_selected_polygon),
            byref(c_selection),
        )

        return selection

    def flip_edges_mesh2d(
        self,
        triangulation_required: bool,
        project_to_land_boundary_required: bool,
    ):
        """Flips mesh2d edges to optimize the mesh smoothness.
        Nodes that are connected to more than six other nodes are typically enclosed by faces of highly non-uniform
        shape and wildly varying areas.

        Args:
            triangulation_required (bool): Whether to triangulate non-triangular cells.
            project_to_land_boundary_required: Whether projection to land boundaries is required.

        """
        self._execute_function(
            self.lib.mkernel_flip_edges_mesh2d,
            self._meshkernelid,
            c_bool(triangulation_required),
            c_int(project_to_land_boundary_required),
        )

    def count_obtuse_triangles_mesh2d(self) -> int:
        """Gets the number of obtuse mesh2d triangles.
        Obtuse triangles are those having one angle larger than 90°.

        Returns:
            int: The number of obtuse triangles.
        """

        n_obtuse_triangles = c_int(0)

        self._execute_function(
            self.lib.mkernel_count_obtuse_triangles_mesh2d,
            self._meshkernelid,
            byref(n_obtuse_triangles),
        )

        return n_obtuse_triangles.value

    def get_obtuse_triangles_mass_centers_mesh2d(self) -> GeometryList:
        """Gets the mass centers of obtuse mesh2d triangles.
        Obtuse triangles are those having one angle larger than 90°.

        Returns:
            GeometryList: The geometry list with the mass center coordinates.
        """
        n_obtuse_triangles = self.count_obtuse_triangles_mesh2d()

        c_geometry_list = CGeometryList()
        c_geometry_list.n_coordinates = n_obtuse_triangles
        c_geometry_list.inner_outer_separator = c_double(
            self._get_inner_outer_separator()
        )
        c_geometry_list.geometry_separator = c_double(self._get_separator())

        geometry_list = c_geometry_list.allocate_memory()

        self._execute_function(
            self.lib.mkernel_get_obtuse_triangles_mass_centers_mesh2d,
            self._meshkernelid,
            byref(c_geometry_list),
        )

        return geometry_list

    def get_splines(
        self, geometry_list: GeometryList, number_of_points_between_nodes: int
    ) -> GeometryList:
        """Get the computed spline points between two corner nodes.

        Args:
            geometry_list (GeometryList): The input corner nodes of the splines
            number_of_points_between_nodes (int): The number of spline points to generate between two corner nodes.

        Returns:
            GeometryList: The output spline.
        """

        # Allocate space for output
        original_number_of_coordinates = geometry_list.x_coordinates.size
        number_of_coordinates = (
            original_number_of_coordinates * number_of_points_between_nodes
            - number_of_points_between_nodes
            + original_number_of_coordinates
            + 1
        )
        x_coordinates = np.empty(number_of_coordinates, dtype=np.double)
        y_coordinates = np.empty(number_of_coordinates, dtype=np.double)
        values = np.empty(number_of_coordinates, dtype=np.double)
        geometry_list_out = GeometryList(x_coordinates, y_coordinates, values)

        # Convert to CGeometryList
        c_geometry_list_in = CGeometryList.from_geometrylist(geometry_list)
        c_geometry_list_out = CGeometryList.from_geometrylist(geometry_list_out)

        self._execute_function(
            self.lib.mkernel_get_splines,
            byref(c_geometry_list_in),
            byref(c_geometry_list_out),
            c_int(number_of_points_between_nodes),
        )

        return geometry_list_out

    def get_mesh_boundaries_as_polygons_mesh2d(self) -> GeometryList:
        """Retrieves the boundaries of a mesh as a series of separated polygons.

        For example, if a mesh has an single inner hole, two polygons will be generated,
        one for the inner boundary and one for the outer boundary.

        Returns:
            GeometryList: The output network boundary polygon.
        """

        # Get number of polygon nodes
        number_of_polygon_nodes = c_int()
        self._execute_function(
            self.lib.mkernel_count_mesh_boundaries_as_polygons_mesh2d,
            self._meshkernelid,
            byref(number_of_polygon_nodes),
        )

        # Create GeometryList instance
        x_coordinates = np.empty(number_of_polygon_nodes.value, dtype=np.double)
        y_coordinates = np.empty(number_of_polygon_nodes.value, dtype=np.double)
        geometry_list_out = GeometryList(x_coordinates, y_coordinates)

        # Get mesh boundary
        c_geometry_list_out = CGeometryList.from_geometrylist(geometry_list_out)
        self._execute_function(
            self.lib.mkernel_get_mesh_boundaries_as_polygons_mesh2d,
            self._meshkernelid,
            byref(c_geometry_list_out),
        )

        return geometry_list_out

    def merge_nodes_mesh2d(
        self, geometry_list: GeometryList, merging_distance: float
    ) -> None:
        """Merges the mesh2d nodes, effectively removing all small edges

        Args:
            geometry_list (GeometryList): The polygon defining the area where the operation will be performed.
            geometry_list (float): The distance below which two nodes will be merged
        """
        c_geometry_list = CGeometryList.from_geometrylist(geometry_list)
        self._execute_function(
            self.lib.mkernel_merge_nodes_mesh2d,
            self._meshkernelid,
            byref(c_geometry_list),
            c_double(merging_distance),
        )

    def merge_two_nodes_mesh2d(self, first_node: int, second_node: int) -> None:
        """Merges two mesh2d nodes into one.

        Args:
            first_node (int): The index of the first node to merge.
            second_node (int): The index of the second node to merge.
        """

        self._execute_function(
            self.lib.mkernel_merge_two_nodes_mesh2d,
            self._meshkernelid,
            c_int(first_node),
            c_int(second_node),
        )

    def get_nodes_in_polygons_mesh2d(
        self, geometry_list: GeometryList, inside: bool
    ) -> ndarray:
        """Gets the indices of the mesh2d nodes selected with a polygon.

        Args:
            geometry_list (GeometryList): The input polygon.
            inside (bool): Selection of the nodes inside the polygon (True) or outside (False)

        Returns:
            ndarray: The integer array describing the selected nodes indices
        """

        c_inside = c_int(inside)
        c_number_of_mesh_nodes = c_int()
        c_geometry_list = CGeometryList.from_geometrylist(geometry_list)

        # Get number of mesh nodes
        self._execute_function(
            self.lib.mkernel_count_nodes_in_polygons_mesh2d,
            self._meshkernelid,
            byref(c_geometry_list),
            c_inside,
            byref(c_number_of_mesh_nodes),
        )

        selected_nodes = np.empty(c_number_of_mesh_nodes.value, dtype=np.int32)
        c_selected_nodes = np.ctypeslib.as_ctypes(selected_nodes)

        # Get selected nodes
        self._execute_function(
            self.lib.mkernel_get_nodes_in_polygons_mesh2d,
            self._meshkernelid,
            byref(c_geometry_list),
            c_inside,
            c_selected_nodes,
        )

        return selected_nodes

    def _get_error(self) -> str:
        c_error_message = c_char_p()
        self.lib.mkernel_get_error(byref(c_error_message))
        return c_error_message.value.decode("ASCII")

    def _execute_function(self, function: Callable, *args):
        """Utility function to execute a C function of MeshKernel and checks its status

        Args:
            function (Callable): The function which we want to call
            args: Arguments which will be passed to `function`

        Raises:
            MeshKernelError: This exception gets raised,
                             if the MeshKernel library reports an error.
        """
        if function(*args) != Status.SUCCESS:
            error_message = self._get_error()
            raise MeshKernelError(error_message)
