from __future__ import annotations

from ctypes import POINTER, Structure, c_double, c_int

import numpy as np

from meshkernel.py_structures import Mesh2d


class CMesh2d(Structure):
    """C-structure intended for internal use only.
    It represents a Mesh2D struct as described by the MeshKernel API.

    Used for communicating with the MeshKernel dll.
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
