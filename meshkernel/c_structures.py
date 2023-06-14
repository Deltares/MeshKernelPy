from __future__ import annotations

from ctypes import POINTER, Structure, c_double, c_int

import numpy as np
from numpy.ctypeslib import as_ctypes

from meshkernel.py_structures import (
    Contacts,
    CurvilinearGrid,
    CurvilinearParameters,
    GeometryList,
    GriddedSamples,
    MakeGridParameters,
    Mesh1d,
    Mesh2d,
    MeshRefinementParameters,
    OrthogonalizationParameters,
    SplinesToCurvilinearParameters,
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
        """Creates a new CMesh instance from a given Mesh2d instance.

        Args:
            mesh2d (Mesh2d): Class of numpy instances owning the state.

        Returns:
            CMesh2d: The created CMesh2d instance.
        """

        c_mesh2d = CMesh2d()

        # Set the pointers
        c_mesh2d.edge_nodes = as_ctypes(mesh2d.edge_nodes)
        c_mesh2d.face_nodes = as_ctypes(mesh2d.face_nodes)
        c_mesh2d.nodes_per_face = as_ctypes(mesh2d.nodes_per_face)
        c_mesh2d.node_x = as_ctypes(mesh2d.node_x)
        c_mesh2d.node_y = as_ctypes(mesh2d.node_y)
        c_mesh2d.edge_x = as_ctypes(mesh2d.edge_x)
        c_mesh2d.edge_y = as_ctypes(mesh2d.edge_y)
        c_mesh2d.face_x = as_ctypes(mesh2d.face_x)
        c_mesh2d.face_y = as_ctypes(mesh2d.face_y)

        # Set the sizes
        c_mesh2d.num_nodes = mesh2d.node_x.size
        c_mesh2d.num_edges = mesh2d.edge_nodes.size // 2
        c_mesh2d.num_faces = mesh2d.face_x.size
        c_mesh2d.num_face_nodes = mesh2d.face_nodes.size

        return c_mesh2d

    def allocate_memory(self) -> Mesh2d:
        """Allocate data according to the parameters with the "num_" prefix.
        The pointers are then set to the freshly allocated memory.
        The memory is owned by the Mesh2d instance which is returned by this method.

        Returns:
            Mesh2d: The object owning the allocated memory.
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

        self.edge_nodes = as_ctypes(edge_nodes)
        self.face_nodes = as_ctypes(face_nodes)
        self.nodes_per_face = as_ctypes(nodes_per_face)
        self.node_x = as_ctypes(node_x)
        self.node_y = as_ctypes(node_y)
        self.edge_x = as_ctypes(edge_x)
        self.edge_y = as_ctypes(edge_y)
        self.face_x = as_ctypes(face_x)
        self.face_y = as_ctypes(face_y)

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
        c_geometry_list.x_coordinates = as_ctypes(geometry_list.x_coordinates)
        c_geometry_list.y_coordinates = as_ctypes(geometry_list.y_coordinates)
        c_geometry_list.values = as_ctypes(geometry_list.values)

        return c_geometry_list


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

        return c_orthogonalizationparameters


class CMeshRefinementParameters(Structure):
    """C-structure intended for internal use only.
    It represents a MeshRefinementParameters struct as described by the MeshKernel API.

    Used for communicating with the MeshKernel dll.

    Attributes:
        max_refinement_iterations (c_int): Maximum number of refinement iterations.
        refine_intersected (c_int): Whether to compute faces intersected by polygon (yes=1/no=0)
        use_mass_center_when_refining (c_int): Whether to use the mass center when splitting a face in the refinement
                                               process (yes=1/no=0)
        min_edge_size (c_double): Minimum cell size.
        refinement_type (c_int): Refinement criterion type.
        connect_hanging_nodes (c_int): Whether to connect hanging nodes at the end of the iteration.
        account_for_samples_outside (c_int): Whether to take samples outside face into account.
    """

    _fields_ = [
        ("max_refinement_iterations", c_int),
        ("refine_intersected", c_int),
        ("use_mass_center_when_refining", c_int),
        ("min_edge_size", c_double),
        ("refinement_type", c_int),
        ("connect_hanging_nodes", c_int),
        ("account_for_samples_outside_face", c_int),
        ("smoothing_iterations", c_int),
        ("max_courant_time", c_double),
        ("directional_refinement", c_int),
    ]

    @staticmethod
    def from_meshrefinementparameters(
        mesh_refinement_parameters: MeshRefinementParameters,
    ) -> CMeshRefinementParameters:
        """Creates a new `CMeshRefinementParameters` instance from the given MeshRefinementParameters instance.

        Args:
            mesh_refinement_parameters (MeshRefinementParameters): The mesh refinement parameters.

        Returns:
            CMeshRefinementParameters: The created C-Structure for the given MeshRefinementParameters.
        """

        c_parameters = CMeshRefinementParameters()
        c_parameters.max_refinement_iterations = (
            mesh_refinement_parameters.max_refinement_iterations
        )
        c_parameters.refine_intersected = mesh_refinement_parameters.refine_intersected
        c_parameters.use_mass_center_when_refining = (
            mesh_refinement_parameters.use_mass_center_when_refining
        )
        c_parameters.min_edge_size = mesh_refinement_parameters.min_edge_size
        c_parameters.refinement_type = mesh_refinement_parameters.refinement_type
        c_parameters.connect_hanging_nodes = (
            mesh_refinement_parameters.connect_hanging_nodes
        )
        c_parameters.account_for_samples_outside_face = (
            mesh_refinement_parameters.account_for_samples_outside_face
        )

        c_parameters.smoothing_iterations = (
            mesh_refinement_parameters.smoothing_iterations
        )
        c_parameters.max_courant_time = mesh_refinement_parameters.max_courant_time
        c_parameters.directional_refinement = (
            mesh_refinement_parameters.directional_refinement
        )

        return c_parameters


class CMakeGridParameters(Structure):
    """C-structure intended for internal use only.
    It represents a MakeGridParameters struct as described by the MeshKernel API.

    Used for communicating with the MeshKernel dll.

    Attributes:
        num_columns (c_int): The number of columns in x direction.
        num_rows (c_int): The number of columns in y direction.
        angle (c_double): The grid angle.
        block_size (c_double): The grid block size, used in x and y direction.
        origin_x (c_double): The x coordinate of the origin, located at the bottom left corner.
        origin_y (c_double): The y coordinate of the origin, located at the bottom left corner.
        block_size_x (c_double): The grid block size in x dimension, used only for squared grids.
        block_size_y (c_double): The grid block size in y dimension, used only for squared grids.
    """

    _fields_ = [
        ("num_columns", c_int),
        ("num_rows", c_int),
        ("angle", c_double),
        ("block_size", c_double),
        ("origin_x", c_double),
        ("origin_y", c_double),
        ("block_size_x", c_double),
        ("block_size_y", c_double),
    ]

    @staticmethod
    def from_makegridparameters(
        make_grid_parameters: MakeGridParameters,
    ) -> CMakeGridParameters:
        """Creates a new `CMeshRefinementParameters` instance from the given MeshRefinementParameters instance.

        Args:
            make_grid_parameters (MakeGridParameters): The make grid parameters.

        Returns:
            CMakeGridParameters: The created C-Structure for the given MakeGridParameters.
        """

        c_parameters = CMakeGridParameters()
        c_parameters.num_columns = make_grid_parameters.num_columns
        c_parameters.num_rows = make_grid_parameters.num_rows
        c_parameters.angle = make_grid_parameters.angle
        c_parameters.block_size = make_grid_parameters.block_size
        c_parameters.origin_x = make_grid_parameters.origin_x
        c_parameters.origin_y = make_grid_parameters.origin_y
        c_parameters.block_size_x = make_grid_parameters.block_size_x
        c_parameters.block_size_y = make_grid_parameters.block_size_y
        return c_parameters


class CMesh1d(Structure):
    """C-structure intended for internal use only.
    It represents a Mesh1D struct as described by the MeshKernel API.

    Used for communicating with the MeshKernel dll.

    Attributes:
        edge_nodes (POINTER(c_int)): The nodes composing each mesh 1d edge.
        node_x (POINTER(c_double)): The x-coordinates of the nodes.
        node_y (POINTER(c_double)): The y-coordinates of the nodes.
        num_nodes (c_int): The number of nodes.
        num_edges (c_int): The number of edges.
    """

    _fields_ = [
        ("edge_nodes", POINTER(c_int)),
        ("node_x", POINTER(c_double)),
        ("node_y", POINTER(c_double)),
        ("num_nodes", c_int),
        ("num_edges", c_int),
    ]

    @staticmethod
    def from_mesh1d(mesh1d: Mesh1d) -> CMesh1d:
        """Creates a new CMesh instance from a given Mesh1d instance.

        Args:
            mesh1d (Mesh1d): Class of numpy instances owning the state.

        Returns:
            CMesh1d: The created CMesh1d instance.
        """

        c_mesh1d = CMesh1d()

        # Set the pointers
        c_mesh1d.edge_nodes = as_ctypes(mesh1d.edge_nodes)
        c_mesh1d.node_x = as_ctypes(mesh1d.node_x)
        c_mesh1d.node_y = as_ctypes(mesh1d.node_y)

        # Set the sizes
        c_mesh1d.num_nodes = mesh1d.node_x.size
        c_mesh1d.num_edges = mesh1d.edge_nodes.size // 2

        return c_mesh1d

    def allocate_memory(self) -> Mesh1d:
        """Allocate data according to the parameters with the "num_" prefix.
        The pointers are then set to the freshly allocated memory.
        The memory is owned by the Mesh1d instance which is returned by this method.

        Returns:
            Mesh1d: The object owning the allocated memory.
        """

        edge_nodes = np.empty(self.num_edges * 2, dtype=np.int32)
        node_x = np.empty(self.num_nodes, dtype=np.double)
        node_y = np.empty(self.num_nodes, dtype=np.double)

        self.edge_nodes = as_ctypes(edge_nodes)
        self.node_x = as_ctypes(node_x)
        self.node_y = as_ctypes(node_y)

        return Mesh1d(
            node_x,
            node_y,
            edge_nodes,
        )


class CContacts(Structure):
    """C-structure intended for internal use only.
    It represents a Contacts struct as described by the MeshKernel API.

    Used for communicating with the MeshKernel dll.

    Attributes:
        mesh1d_indices (POINTER(c_int)): The indices of the mesh1d nodes.
        mesh2d_indices (POINTER(c_int)): The indices of the mesh2d faces.
        num_contacts (c_int): The number of contacts.
    """

    _fields_ = [
        ("mesh1d_indices", POINTER(c_int)),
        ("mesh2d_indices", POINTER(c_int)),
        ("num_contacts", c_int),
    ]

    @staticmethod
    def from_contacts(contacts: Contacts) -> CContacts:
        """Creates a new `CContacts` instance from the given Contacts instance.

        Args:
            contacts (Contacts): The contacts.

        Returns:
            CContacts: The created C-Structure for the given Contacts.
        """

        c_contacts = CContacts()

        c_contacts.mesh1d_indices = as_ctypes(contacts.mesh1d_indices)
        c_contacts.mesh2d_indices = as_ctypes(contacts.mesh2d_indices)
        c_contacts.num_contacts = contacts.mesh1d_indices.size

        return c_contacts

    def allocate_memory(self) -> Contacts:
        """Allocate data according to the parameters with the "num_" prefix.
        The pointers are then set to the freshly allocated memory.
        The memory is owned by the Contacts instance which is returned by this method.

        Returns:
            Contacts: The object owning the allocated memory.
        """

        mesh1d_indices = np.empty(self.num_contacts, dtype=np.int32)
        mesh2d_indices = np.empty(self.num_contacts, dtype=np.int32)

        self.mesh1d_indices = as_ctypes(mesh1d_indices)
        self.mesh2d_indices = as_ctypes(mesh2d_indices)

        return Contacts(mesh1d_indices, mesh2d_indices)


class CCurvilinearGrid(Structure):
    """C-structure intended for internal use only.
    It represents a Curvilinear struct as described by the MeshKernel API.

    Used for communicating with the MeshKernel dll.

    Attributes:
        node_x (POINTER(c_double)): The x-coordinates of the nodes.
        node_y (POINTER(c_double)): The y-coordinates of the nodes.
        num_m (c_int): The number of curvilinear grid nodes along m.
        num_n (c_int): The number of curvilinear grid nodes along n.
    """

    _fields_ = [
        ("node_x", POINTER(c_double)),
        ("node_y", POINTER(c_double)),
        ("num_m", c_int),
        ("num_n", c_int),
    ]

    @staticmethod
    def from_curvilinearGrid(curvilinear_grid: CurvilinearGrid) -> CCurvilinearGrid:
        """Creates a new CMesh instance from a given CurvilinearGrid instance.

        Args:
            curvilinear_grid (CurvilinearGrid): Class of numpy instances owning the state.

        Returns:
            CCurvilinearGrid: The created CCurvilinearGrid instance.
        """

        c_curvilinear_grid = CCurvilinearGrid()

        # Set the pointers
        c_curvilinear_grid.node_x = as_ctypes(curvilinear_grid.node_x)
        c_curvilinear_grid.node_y = as_ctypes(curvilinear_grid.node_y)

        # Set the sizes
        c_curvilinear_grid.num_m = curvilinear_grid.num_m
        c_curvilinear_grid.num_n = curvilinear_grid.num_n

        return c_curvilinear_grid

    def allocate_memory(self) -> CurvilinearGrid:
        """Allocate data according to the parameters with the "num_" prefix.
        The pointers are then set to the freshly allocated memory.
        The memory is owned by the CurvilinearGrid instance which is returned by this method.

        Returns:
            CurvilinearGrid: The object owning the allocated memory.
        """

        node_x = np.empty(self.num_m * self.num_n, dtype=np.double)
        node_y = np.empty(self.num_m * self.num_n, dtype=np.double)

        self.node_x = as_ctypes(node_x)
        self.node_y = as_ctypes(node_y)

        return CurvilinearGrid(node_x, node_y, self.num_m, self.num_n)


class CCurvilinearParameters(Structure):
    """C-structure intended for internal use only.
    It represents an CurvilinearParameters struct as described by the MeshKernel API.

    Used for communicating with the MeshKernel dll.

    Attributes:
        m_refinement (c_int): M-refinement factor for regular grid generation.
        n_refinement (c_int): N-refinement factor for regular grid generation.
        smoothing_iterations (c_int): Nr. of inner iterations in regular grid smoothing.
        smoothing_parameter (c_double): Smoothing parameter.
        attraction_parameter (c_double): Attraction/repulsion parameter.
    """

    _fields_ = [
        ("m_refinement", c_int),
        ("n_refinement", c_int),
        ("smoothing_iterations", c_int),
        ("smoothing_parameter", c_double),
        ("attraction_parameter", c_double),
    ]

    @staticmethod
    def from_curvilinearParameters(
        curvilinear_parameters: CurvilinearParameters,
    ) -> CCurvilinearParameters:
        """Creates a new `CCurvilinearParameters` instance from the given CurvilinearParameters instance.

        Args:
            curvilinear_parameters (CurvilinearParameters): The curvilinear parameters.

        Returns:
            CCurvilinearParameters: The created C-Structure for the given CurvilinearParameters.
        """

        c_curvilinear_parameters = CCurvilinearParameters()
        c_curvilinear_parameters.m_refinement = curvilinear_parameters.m_refinement
        c_curvilinear_parameters.n_refinement = curvilinear_parameters.n_refinement
        c_curvilinear_parameters.smoothing_iterations = (
            curvilinear_parameters.smoothing_iterations
        )
        c_curvilinear_parameters.smoothing_parameter = (
            curvilinear_parameters.smoothing_parameter
        )
        c_curvilinear_parameters.attraction_parameter = (
            curvilinear_parameters.attraction_parameter
        )

        return c_curvilinear_parameters


class CSplinesToCurvilinearParameters(Structure):
    """C-structure intended for internal use only.
    It represents an SplinesToCurvilinearParameters struct as described by the MeshKernel API.

    Used for communicating with the MeshKernel dll.

    Attributes:
        aspect_ratio (c_double): Aspect ratio.
        aspect_ratio_grow_factor (c_double): Grow factor of aspect ratio.
        average_width (c_double): Average mesh width on center spline.
        curvature_adapted_grid_spacing (c_int): Curvature adapted grid spacing.
        grow_grid_outside (c_int): Grow the grid outside the prescribed grid height.
        maximum_num_faces_in_uniform_part (c_int): Maximum number of layers in the uniform part.
        nodes_on_top_of_each_other_tolerance (c_double): On-top-of-each-other tolerance.).
        min_cosine_crossing_angles (c_double): Minimum allowed absolute value of crossing-angle cosine.
        check_front_collisions (c_int): Check for collisions with other parts of the front.
        remove_skinny_triangles (c_int): Check for collisions with other parts of the front.
    """

    _fields_ = [
        ("aspect_ratio", c_double),
        ("aspect_ratio_grow_factor", c_double),
        ("average_width", c_double),
        ("curvature_adapted_grid_spacing", c_int),
        ("grow_grid_outside", c_int),
        ("maximum_num_faces_in_uniform_part", c_int),
        ("nodes_on_top_of_each_other_tolerance", c_double),
        ("min_cosine_crossing_angles", c_double),
        ("check_front_collisions", c_int),
        ("remove_skinny_triangles", c_int),
    ]

    @staticmethod
    def from_splinesToCurvilinearParameters(
        splines_to_curvilinear_parameters: SplinesToCurvilinearParameters,
    ) -> CSplinesToCurvilinearParameters:
        """Creates a new `COrthogonalizationParameters` instance from the given OrthogonalizationParameters instance.

        Args:
            orthogonalization_parameters (OrthogonalizationParameters): The orthogonalization parameters.

        Returns:
            COrthogonalizationParameters: The created C-Structure for the given OrthogonalizationParameters.
        """

        c_splines_to_curvilinear_parameters = CSplinesToCurvilinearParameters()
        c_splines_to_curvilinear_parameters.aspect_ratio = (
            splines_to_curvilinear_parameters.aspect_ratio
        )
        c_splines_to_curvilinear_parameters.aspect_ratio_grow_factor = (
            splines_to_curvilinear_parameters.aspect_ratio_grow_factor
        )
        c_splines_to_curvilinear_parameters.average_width = (
            splines_to_curvilinear_parameters.average_width
        )
        c_splines_to_curvilinear_parameters.curvature_adapted_grid_spacing = (
            splines_to_curvilinear_parameters.curvature_adapted_grid_spacing
        )
        c_splines_to_curvilinear_parameters.grow_grid_outside = (
            splines_to_curvilinear_parameters.grow_grid_outside
        )
        c_splines_to_curvilinear_parameters.maximum_num_faces_in_uniform_part = (
            splines_to_curvilinear_parameters.maximum_num_faces_in_uniform_part
        )
        c_splines_to_curvilinear_parameters.nodes_on_top_of_each_other_tolerance = (
            splines_to_curvilinear_parameters.nodes_on_top_of_each_other_tolerance
        )
        c_splines_to_curvilinear_parameters.min_cosine_crossing_angles = (
            splines_to_curvilinear_parameters.min_cosine_crossing_angles
        )
        c_splines_to_curvilinear_parameters.check_front_collisions = (
            splines_to_curvilinear_parameters.check_front_collisions
        )
        c_splines_to_curvilinear_parameters.remove_skinny_triangles = (
            splines_to_curvilinear_parameters.remove_skinny_triangles
        )

        return c_splines_to_curvilinear_parameters


class CGriddedSamples(Structure):
    """C-structure intended for internal use only.
    It represents a GriddedSamples struct as described by the MeshKernel API.

    Used for communicating with the MeshKernel dll.

    Attributes:
        n_cols (c_int): Number of grid columns.
        n_rows (c_int): Number of grid rows.
        x_origin (c_double): X coordinate of the grid origin.
        y_origin (c_double): Y coordinate of the grid origin.
        cell_size (c_int):  Constant grid cell size.
        x_coordinates (POINTER(c_double)): If not nullptr, coordinates for non-uniform grid spacing in x direction.
        y_coordinates (POINTER(c_double)): If not nullptr, coordinates for non-uniform grid spacing in y direction.
        values (POINTER(c_double)): Sample values.
    """

    _fields_ = [
        ("n_cols", c_int),
        ("n_rows", c_int),
        ("x_origin", c_double),
        ("y_origin", c_double),
        ("cell_size", c_double),
        ("x_coordinates", POINTER(c_double)),
        ("y_coordinates", POINTER(c_double)),
        ("values", POINTER(c_double)),
    ]

    @staticmethod
    def from_griddedSamples(
        gridded_samples: GriddedSamples,
    ) -> CGriddedSamples:
        """Creates a new `CGriddedSamples` instance from the given GriddedSamples instance.

        Args:
            gridded_samples (GriddedSamples): The GriddedSamples samples.

        Returns:
            CGriddedSamples: The created C-Structure for the given CGriddedSamples.
        """

        c_gridded_samples = CGriddedSamples()

        if len(gridded_samples.x_coordinates) == 0:
            n_cols = gridded_samples.n_cols
            c_gridded_samples.x_coordinates = None
        else:
            n_cols = len(gridded_samples.x_coordinates) - 1
            c_gridded_samples.x_coordinates = as_ctypes(gridded_samples.x_coordinates)

        if len(gridded_samples.y_coordinates) == 0:
            n_rows = gridded_samples.n_rows
            c_gridded_samples.y_coordinates = None
        else:
            n_rows = len(gridded_samples.y_coordinates) - 1
            c_gridded_samples.y_coordinates = as_ctypes(gridded_samples.y_coordinates)

        c_gridded_samples.n_cols = n_cols
        c_gridded_samples.n_rows = n_rows
        c_gridded_samples.x_origin = gridded_samples.x_origin
        c_gridded_samples.y_origin = gridded_samples.y_origin
        c_gridded_samples.cell_size = gridded_samples.cell_size
        c_gridded_samples.values = as_ctypes(gridded_samples.values)

        return c_gridded_samples
