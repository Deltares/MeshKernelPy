import logging
import os
import platform
import sys
from ctypes import CDLL, POINTER, byref, c_char_p, c_double, c_int
from enum import Enum, IntEnum, unique
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np

from meshkernel.c_structures import CMesh2d
from meshkernel.errors import InputError, MeshKernelError
from meshkernel.py_structures import Mesh2d

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

    def __init__(self, is_geographic: bool):

        # Determine OS
        if platform.system() == "Windows":
            lib_path = Path(__file__).parent.parent / "lib" / "MeshKernelApi.dll"
        elif platform.system() == "Linux":
            lib_path = Path(__file__).parent.parent / "lib" / "libMeshKernelApi.so"
        else:
            raise OSError("Unsupported operating system")

        # LoadLibraryEx flag: LOAD_WITH_ALTERED_SEARCH_PATH 0x08
        # -> uses the altered search path for resolving ddl dependencies
        # `winmode` has no effect while running on Linux or macOS
        self.lib = CDLL(str(lib_path), winmode=0x08)

        self.libname = os.path.basename(lib_path)
        self._allocate_state(is_geographic)

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

    def deallocate_state(self) -> None:
        """
        Deallocate mesh state.
        """

        self._execute_function(
            self.lib.mkernel_deallocate_state,
            self._meshkernelid,
        )

    def set_mesh2d(self, mesh2d: Mesh2d) -> None:
        """Sets the two-dimensional mesh state of the MeshKernel.

        Please note that this involves a copy of the data
        and should therefore not be called in hot loops.

        Args:
            mesh2d (Mesh2d): The input data used for setting the state
        """
        cmesh2d = CMesh2d.from_mesh2d(mesh2d)

        self._execute_function(
            self.lib.mkernel_set_mesh2d, self._meshkernelid, byref(cmesh2d)
        )

    def get_mesh2d(self) -> Mesh2d:
        """Gets the two-dimensional mesh state from the MeshKernel.

        Please note that this involves a copy of the data
        and should therefore not be called in hot loops.

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

    def insert_node_mesh2d(self, x: float, y: float, index: int):
        """Insert a new node at the specified coordinates

        Args:
            x (float): The x-coordinate of the new node
            y (float): The y-coordinate of the new node
            index (int): The index of the new node
        """
        self._execute_function(
            self.lib.mkernel_insert_node_mesh2d,
            self._meshkernelid,
            c_double(x),
            c_double(y),
            byref(c_int(index)),
        )

    def _execute_function(self, function, *args, detail=""):
        """
        Utility function to execute a BMI function in the kernel and checks its status
        """

        if function(*args) != Status.SUCCESS:
            msg = f"MeshKernel exception in: {function.__name__} ({detail})"
            # TODO: Report errors from MeshKernel
            raise MeshKernelError(msg)
