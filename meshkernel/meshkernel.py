import logging
import os
import platform
import sys
from ctypes import CDLL, byref, POINTER, c_char_p, c_double, c_int
from enum import Enum, IntEnum, unique
from typing import Iterable, Tuple

import numpy as np

from meshkernel.errors import InputError, MeshKernelError
from meshkernel.py_structures import Mesh2d
from meshkernel.c_structures import CMesh2d

logger = logging.getLogger(__name__)


@unique
class Status(IntEnum):
    SUCCESS = 0
    EXCEPTION = 1
    INVALID_GEOMETRY = 2


class MeshKernel:
    """
    Please document
    """

    def __init__(self, is_geographic: bool):
        lib_path = "lib/MeshKernelApi.dll"
        if sys.version_info[0:2] < (3, 8):
            # Python version < 3.8
            self.lib = CDLL(lib_path)
        else:
            # LoadLibraryEx flag: LOAD_WITH_ALTERED_SEARCH_PATH 0x08
            # -> uses the altered search path for resolving ddl dependencies
            # `winmode` has no effect while running on Linux or macOS
            self.lib = CDLL(lib_path, winmode=0x08)

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
        cmesh2d = CMesh2d.from_mesh2d(mesh2d)

        self._execute_function(
            self.lib.mkernel_set_mesh2d, self._meshkernelid, byref(cmesh2d)
        )

    def get_mesh2d(self) -> Mesh2d:
        cmesh2d = CMesh2d()
        self._execute_function(
            self.lib.mkernel_get_dimensions_mesh2d, self._meshkernelid, byref(cmesh2d)
        )

        mesh2d = cmesh2d.allocate_memory()
        self._execute_function(
            self.lib.mkernel_get_data_mesh2d, self._meshkernelid, byref(cmesh2d)
        )

        return mesh2d

    def _execute_function(self, function, *args, detail=""):
        """
        Utility function to execute a BMI function in the kernel and checks its status
        """

        if function(*args) != Status.SUCCESS:
            msg = f"MeshKernel exception in: {function.__name__} ({detail})"
            # TODO: Report errors from MeshKernel
            raise MeshKernelError(msg)
