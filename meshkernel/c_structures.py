from __future__ import annotations

from ctypes import POINTER, Structure, c_double, c_int

import numpy as np

from meshkernel.py_structures import (
    GeometryList,
    InterpolationParameters,
    Mesh2d,
    OrthogonalizationParameters,
)


class CMesh2d(Structure):
    """C-structure intended for internal use only.
    It represents a Mesh2D struct as described by the MeshKernel API.

    Used for communicating with the MeshKernel dll.

    Attributes:
        edge_nodes (POINTER(c_int)): The nodes composing each mesh 2d edge.
        face_nodes (POINTER(c_int)): The nodes composing each mesh 2d face.
        nodes_per_face (POINTER(c_int)): The nodes composing each mesh 2d face.
        node_x (POINTER(c_double)): The x-coordinates of the nodes.
        node_y (POINTER(c_double)): The y-coordinates of the nodes.
        edge_x (POINTER(c_double)): The x-coordinates of the mesh edges' middle points.
        edge_y (POINTER(c_double)): The x-coordinates of the mesh edges' middle points.
        face_x (POINTER(c_double)): The x-coordinates of the mesh faces' mass centers.
        face_y (POINTER(c_double)): The y-coordinates of the mesh faces' mass centers.
        num_nodes (c_int): The number of mesh nodes.
        num_edges (c_int): The number of edges.
        num_faces (c_int): The number of faces.
        num_face_nodes (c_int): The total number of nodes composing the mesh 2d faces.
    """

    _fields_ = [
        ("edge_nodes", POINTER(c_int)),
        ("face_nodes", POINTER(c_int)),
        ("nodes_per_face", POINTER(c_int)),
        ("node_x", POINTER(c_double)),
        ("node_y", POINTER(c_double)),
        ("edge_x", POINTER(c_double)),
        ("edge_y", POINTER(c_double)),
        ("face_x", POINTER(c_double)),
        ("face_y", POINTER(c_double)),
        ("num_nodes", c_int),
        ("num_edges", c_int),
        ("num_faces", c_int),
        ("num_face_nodes", c_int),
    ]

    @staticmethod
    def from_mesh2d(mesh2d: Mesh2d) -> CMesh2d:
        """Creates a new CMesh instance from a given Mesh2d instance

        Args:
            mesh2d (Mesh2d): Class of numpy instances owning the state

        Returns:
            CMesh2d: The created CMesh2d instance
        """

        cmesh2d = CMesh2d()

        # Set the pointers
        cmesh2d.edge_nodes = np.ctypeslib.as_ctypes(mesh2d.edge_nodes)
        cmesh2d.face_nodes = np.ctypeslib.as_ctypes(mesh2d.face_nodes)
        cmesh2d.nodes_per_face = np.ctypeslib.as_ctypes(mesh2d.nodes_per_face)
        cmesh2d.node_x = np.ctypeslib.as_ctypes(mesh2d.node_x)
        cmesh2d.node_y = np.ctypeslib.as_ctypes(mesh2d.node_y)
        cmesh2d.edge_x = np.ctypeslib.as_ctypes(mesh2d.edge_x)
        cmesh2d.edge_y = np.ctypeslib.as_ctypes(mesh2d.edge_y)
        cmesh2d.face_x = np.ctypeslib.as_ctypes(mesh2d.face_x)
        cmesh2d.face_y = np.ctypeslib.as_ctypes(mesh2d.face_y)

        # Set the sizes
        cmesh2d.num_nodes = mesh2d.node_x.size
        cmesh2d.num_edges = mesh2d.edge_nodes.size // 2
        cmesh2d.num_faces = mesh2d.face_x.size
        cmesh2d.num_face_nodes = mesh2d.face_nodes.size

        return cmesh2d

    def allocate_memory(self) -> Mesh2d:
        """Allocate data according to the parameters with the "num_" prefix.
        The pointers are then set to the freshly allocated memory.
        The memory is owned by the Mesh2d instance which is returned by this method.

        Returns:
            Mesh2d: The object owning the allocated memory
        """

        edge_nodes = np.empty(self.num_edges * 2, dtype=np.int32)
        face_nodes = np.empty(self.num_face_nodes, dtype=np.int32)
        nodes_per_face = np.empty(self.num_faces, dtype=np.int32)
        node_x = np.empty(self.num_nodes, dtype=np.double)
        node_y = np.empty(self.num_nodes, dtype=np.double)
        edge_x = np.empty(self.num_edges, dtype=np.double)
        edge_y = np.empty(self.num_edges, dtype=np.double)
        face_x = np.empty(self.num_faces, dtype=np.double)
        face_y = np.empty(self.num_faces, dtype=np.double)

        self.edge_nodes = np.ctypeslib.as_ctypes(edge_nodes)
        self.face_nodes = np.ctypeslib.as_ctypes(face_nodes)
        self.nodes_per_face = np.ctypeslib.as_ctypes(nodes_per_face)
        self.node_x = np.ctypeslib.as_ctypes(node_x)
        self.node_y = np.ctypeslib.as_ctypes(node_y)
        self.edge_x = np.ctypeslib.as_ctypes(edge_x)
        self.edge_y = np.ctypeslib.as_ctypes(edge_y)
        self.face_x = np.ctypeslib.as_ctypes(face_x)
        self.face_y = np.ctypeslib.as_ctypes(face_y)

        return Mesh2d(
            node_x,
            node_y,
            edge_nodes,
            face_nodes,
            nodes_per_face,
            edge_x,
            edge_y,
            face_x,
            face_y,
        )


class CGeometryList(Structure):
    """C-structure intended for internal use only.
    It represents a GeometryList struct as described by the MeshKernel API.

    Used for communicating with the MeshKernel dll.

    Attributes:
        geometry_separator (c_double): The value used as a separator in the coordinates.
        inner_outer_separator (c_double): The value used to separate the inner part of a polygon from its outer part.
        n_coordinates (c_int): The number of coordinate values.
        x_coordinates (POINTER(c_double)): The x coordinates.
        y_coordinates (POINTER(c_double)): The y coordinates.
        values (POINTER(c_double)): The values on this mesh2d.
    """

    _fields_ = [
        ("geometry_separator", c_double),
        ("inner_outer_separator", c_double),
        ("n_coordinates", c_int),
        ("x_coordinates", POINTER(c_double)),
        ("y_coordinates", POINTER(c_double)),
        ("values", POINTER(c_double)),
    ]

    @staticmethod
    def from_geometrylist(geometry_list: GeometryList) -> CGeometryList:
        """Creates a new `CGeometryList` instance from the given GeometryList instance.

        Args:
            geometry_list (GeometryList): The geometry list.

        Returns:
            CGeometryList: The created C-Structure for the given GeometryList.
        """

        c_geometry_list = CGeometryList()

        c_geometry_list.geometry_separator = geometry_list.geometry_separator
        c_geometry_list.inner_outer_separator = geometry_list.inner_outer_separator
        c_geometry_list.n_coordinates = geometry_list.x_coordinates.size
        c_geometry_list.x_coordinates = np.ctypeslib.as_ctypes(
            geometry_list.x_coordinates
        )
        c_geometry_list.y_coordinates = np.ctypeslib.as_ctypes(
            geometry_list.y_coordinates
        )
        c_geometry_list.values = np.ctypeslib.as_ctypes(geometry_list.values)

        return c_geometry_list

    def allocate_memory(self) -> GeometryList:
        """Allocate data according to the number of coordinates.
        The pointers are then set to the freshly allocated memory.
        The memory is owned by the GeometryList instance which is returned by this method.

        Returns:
            GeometryList: The object owning the allocated memory
        """

        x_coordinates = np.empty(self.n_coordinates, dtype=np.double)
        y_coordinates = np.empty(self.n_coordinates, dtype=np.double)
        values = np.empty(self.n_coordinates, dtype=np.double)

        self.x_coordinates = np.ctypeslib.as_ctypes(x_coordinates)
        self.y_coordinates = np.ctypeslib.as_ctypes(y_coordinates)
        self.values = np.ctypeslib.as_ctypes(values)

        return GeometryList(
            x_coordinates,
            y_coordinates,
            values,
            self.geometry_separator,
            self.inner_outer_separator,
        )


class COrthogonalizationParameters(Structure):
    """C-structure intended for internal use only.
    It represents an OrthogonalizationParameters struct as described by the MeshKernel API.

    Used for communicating with the MeshKernel dll.

    Attributes:
        outer_iterations (c_int): Number of outer iterations in orthogonalization.
        boundary_iterations (c_int): Number of boundary iterations in grid/net orthogonalization within itatp.
        inner_iterations (c_int): Number of inner iterations in grid/net orthogonalization within itbnd.
        orthogonalization_to_smoothing_factor (c_double): Factor from between grid smoothing (0) and
                                                          grid orthogonality (1).
        orthogonalization_to_smoothing_factor_at_boundary (c_double): Minimum ATPF on the boundary.
        areal_to_angle_smoothing_factor (c_double): Factor between smoother 1d0 and area-homogenizer 0d0.
    """

    _fields_ = [
        ("outer_iterations", c_int),
        ("boundary_iterations", c_int),
        ("inner_iterations", c_int),
        ("orthogonalization_to_smoothing_factor", c_double),
        ("orthogonalization_to_smoothing_factor_at_boundary", c_double),
        ("areal_to_angle_smoothing_factor", c_double),
    ]

    @staticmethod
    def from_orthogonalizationparameters(
        orthogonalization_parameters: OrthogonalizationParameters,
    ) -> COrthogonalizationParameters:
        """Creates a new `COrthogonalizationParameters` instance from the given OrthogonalizationParameters instance.

        Args:
            orthogonalization_parameters (OrthogonalizationParameters): The orthogonalization parameters.

        Returns:
            COrthogonalizationParameters: The created C-Structure for the given OrthogonalizationParameters.
        """

        c_orthogonalizationparameters = COrthogonalizationParameters()
        c_orthogonalizationparameters.outer_iterations = (
            orthogonalization_parameters.outer_iterations
        )
        c_orthogonalizationparameters.boundary_iterations = (
            orthogonalization_parameters.boundary_iterations
        )
        c_orthogonalizationparameters.inner_iterations = (
            orthogonalization_parameters.inner_iterations
        )
        c_orthogonalizationparameters.orthogonalization_to_smoothing_factor = (
            orthogonalization_parameters.orthogonalization_to_smoothing_factor
        )
        c_orthogonalizationparameters.orthogonalization_to_smoothing_factor_at_boundary = (
            orthogonalization_parameters.orthogonalization_to_smoothing_factor_at_boundary
        )
        c_orthogonalizationparameters.areal_to_angle_smoothing_factor = (
            orthogonalization_parameters.areal_to_angle_smoothing_factor
        )


class CInterpolationParameters:
    """C-structure intended for internal use only.
    It represents an InterpolationParameters struct as described by the MeshKernel API.

    Used for communicating with the MeshKernel dll.

    Attributes:
        max_num_refinement_iterations (c_int): Maximum number of refinement iterations, set to 1 if only one refinement is wanted.
        averaging_method (c_int): Averaging method : 1 = simple averaging, 2 = closest point, 3 = max, 4 = min,
                                  5 = inverse weighted distance, 6 = minabs, 7 = kdtree.
        minimum_num_points (c_int): Minimum number of points needed inside cell to handle the cell.
        relative_search_radius (c_double): Relative search cell size, 1 = actual cell size, 2 = twice as large,
                                           search radius can be larger than cell so more sample are included.
        interpolate_to (c_int): Interpolation settings, 1 = bathy, 2 = zk, 3 = s1, 4 = Zc.
        refine_intersected (c_int): Whether to compute faces intersected by polygon (yes=1/no=0)
        use_mass_center_when_refining (c_int): Whether to use the mass center when splitting a face in the refinement
                                               process (yes=1/no=0)
    """

    _fields_ = [
        ("max_num_refinement_iterations", c_int),
        ("averaging_method", c_int),
        ("minimum_num_points", c_int),
        ("relative_search_radius", c_double),
        ("interpolate_to", c_int),
        ("refine_intersected", c_int),
        ("use_mass_center_when_refining", c_int),
    ]

    @staticmethod
    def from_interpolationparameters(
        interpolation_parameters: InterpolationParameters,
    ) -> CInterpolationParameters:
        """Creates a new `CInterpolationParameters` instance from the given InterpolationParameters instance.

        Args:
            interpolation_parameters (InterpolationParameters): The interpolation parameters.

        Returns:
            CInterpolationParameters: The created C-Structure for the given InterpolationParameters.
        """

        c_interpolationparameters = CInterpolationParameters()
        c_interpolationparameters.max_num_refinement_iterations = (
            interpolation_parameters.max_refinement_iterations
        )
        c_interpolationparameters.averaging_method = (
            interpolation_parameters.averaging_method
        )
        c_interpolationparameters.minimum_num_points = (
            interpolation_parameters.minimum_points
        )
        c_interpolationparameters.relative_search_radius = (
            interpolation_parameters.relative_search_radius
        )
        c_interpolationparameters.interpolate_to = (
            interpolation_parameters.interpolate_to
        )
        c_interpolationparameters.refine_intersected = (
            interpolation_parameters.refine_intersected
        )
        c_interpolationparameters.use_mass_center_when_refining = (
            interpolation_parameters.use_mass_center_when_refining
        )
