from ctypes import Structure, POINTER, c_int, c_double
import numpy as np

from meshkernel import Mesh2d


class CMesh2d(Structure):
    """C-structure that represent a Mesh2D instance.

    Used for communicating with the MeshKernel dll
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

    @classmethod
    def from_mesh2d(cls, mesh2d: Mesh2d):
        """Creates a new CMesh instance from a given Mesh2d instance

        Args:
            mesh2d (Mesh2d): Class of numpy instances owning the state
        """

        cmesh2d = cls()

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
        cmesh2d.num_edges = mesh2d.edge_x.size
        cmesh2d.num_faces = mesh2d.face_x.size
        cmesh2d.num_face_nodes = mesh2d.face_nodes.size

        return cmesh2d
