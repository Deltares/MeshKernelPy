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

from meshkernel.c_structures import CGeometryList, CMesh2d
from meshkernel.errors import InputError, MeshKernelError
from meshkernel.py_structures import DeleteMeshOption, GeometryList, Mesh2d

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
            is_geographic (bool, optional): Whether the mesh is cartesian (False) or spherical (True). Defaults to False.

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

    def set_mesh2d(self, mesh2d: Mesh2d) -> None:
        """Sets the two-dimensional mesh state of the MeshKernel.

        Please note that this involves a copy of the data.

        Args:
            mesh2d (Mesh2d): The input data used for setting the state.
        """
        c_mesh2d = CMesh2d.from_mesh2d(mesh2d)

        self._execute_function(
            self.lib.mkernel_set_mesh2d, self._meshkernelid, byref(c_mesh2d)
        )

    def get_mesh2d(self) -> Mesh2d:
        """Gets the two-dimensional mesh state from the MeshKernel.

        Please note that this involves a copy of the data.

        Returns:
            Mesh2d: A copy of the two-dimensional mesh state.
        """

        c_mesh2d = self._get_dimensions_mesh2d()
        mesh2d = c_mesh2d.allocate_memory()
        self._execute_function(
            self.lib.mkernel_get_data_mesh2d, self._meshkernelid, byref(c_mesh2d)
        )

        return mesh2d

    def _get_dimensions_mesh2d(self) -> CMesh2d:
        """Gets the Mesh2d dimensions.
        The integer parameters of the Mesh2D struct are set to the corresponding dimensions.
        The pointers are set to null, and must be set to correctly sized memory before passing the struct to mkernel_get_data_mesh2d.

        Returns:
            Mesh2d: The Mesh2d dimensions.
        """
        c_mesh2d = CMesh2d()
        self._execute_function(
            self.lib.mkernel_get_dimensions_mesh2d, self._meshkernelid, byref(c_mesh2d)
        )
        return c_mesh2d

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

    def get_hanging_edges_mesh2d(self) -> ndarray:
        """Gets the indices of hanging edges. A hanging edge is an edge where one of the two nodes is not connected.

        Returns:
            ndarray:  The integer array describing the indices of the hanging edges.
        """

        # Get number of hanging edges
        number_hanging_edges = self._count_hanging_edges_mesh2d()

        # Get hanging edges
        hanging_edges = np.empty(number_hanging_edges, dtype=np.int32)
        c_hanging_edges = np.ctypeslib.as_ctypes(hanging_edges)
        self._execute_function(
            self.lib.mkernel_get_hanging_edges_mesh2d,
            self._meshkernelid,
            byref(c_hanging_edges),
        )

        return hanging_edges

    def _count_hanging_edges_mesh2d(self) -> int:
        """Count the number of hanging edges in a mesh2d. An hanging edge is an edge where one of the two nodes is not connected.

        Returns:
            int: The number of hanging edges.
        """
        c_number_hanging_edges = c_int()
        self._execute_function(
            self.lib.mkernel_count_hanging_edges_mesh2d,
            self._meshkernelid,
            byref(c_number_hanging_edges),
        )
        return c_number_hanging_edges.value

    def delete_hanging_edges_mesh2d(self) -> None:
        """Delete the hanging edges in the Mesh2d.
        A hanging edge is an edge where one of the two nodes is not connected.
        """

        self._execute_function(
            self.lib.mkernel_delete_hanging_edges_mesh2d, self._meshkernelid
        )

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
        number_of_polygon_nodes = self._count_mesh_boundaries_as_polygons_mesh2d()

        # Create GeometryList instance
        x_coordinates = np.empty(number_of_polygon_nodes, dtype=np.double)
        y_coordinates = np.empty(number_of_polygon_nodes, dtype=np.double)
        geometry_list_out = GeometryList(x_coordinates, y_coordinates)

        # Get mesh boundary
        c_geometry_list_out = CGeometryList.from_geometrylist(geometry_list_out)
        self._execute_function(
            self.lib.mkernel_get_mesh_boundaries_as_polygons_mesh2d,
            self._meshkernelid,
            byref(c_geometry_list_out),
        )

        return geometry_list_out

    def _count_mesh_boundaries_as_polygons_mesh2d(self) -> int:
        """Counts the number of polygon nodes contained in the mesh boundary polygons
        computed in function get_mesh_boundaries_as_polygons_mesh2d.

        Returns:
            int: The number of polygon nodes.
        """
        number_of_polygon_nodes = c_int()
        self._execute_function(
            self.lib.mkernel_count_mesh_boundaries_as_polygons_mesh2d,
            self._meshkernelid,
            byref(number_of_polygon_nodes),
        )
        return number_of_polygon_nodes.value

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
        """Merges `first_node` into `second_node`.

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

        # Get number of mesh nodes
        number_of_mesh_nodes = self._count_nodes_in_polygons_mesh2d(
            geometry_list, inside
        )

        selected_nodes = np.empty(number_of_mesh_nodes, dtype=np.int32)
        c_selected_nodes = np.ctypeslib.as_ctypes(selected_nodes)
        c_geometry_list = CGeometryList.from_geometrylist(geometry_list)

        # Get selected nodes
        self._execute_function(
            self.lib.mkernel_get_nodes_in_polygons_mesh2d,
            self._meshkernelid,
            byref(c_geometry_list),
            c_int(inside),
            c_selected_nodes,
        )

        return selected_nodes

    def _count_nodes_in_polygons_mesh2d(
        self, geometry_list: GeometryList, inside: int
    ) -> int:
        """Counts the number of selected mesh node indices.
        This function should be used by clients before `get_nodes_in_polygons_mesh2d`
        for allocating an integer array storing the selection results.

        Returns:
            int: The number of selected nodes
        """
        c_number_of_mesh_nodes = c_int()
        c_geometry_list = CGeometryList.from_geometrylist(geometry_list)

        # Get number of mesh nodes
        self._execute_function(
            self.lib.mkernel_count_nodes_in_polygons_mesh2d,
            self._meshkernelid,
            byref(c_geometry_list),
            c_int(inside),
            byref(c_number_of_mesh_nodes),
        )
        return c_number_of_mesh_nodes.value

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
