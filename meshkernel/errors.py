from typing import Tuple

import meshkernel.py_structures as mps


class Error(Exception):
    """Base class for exceptions in this module."""


class InputError(Error):
    """Exception raised for errors in the input."""


class MeshKernelError(Error):
    """Exception raised for errors occurring in the MeshKernel library."""

    def __init__(self, category: str, message: str):
        super().__init__(category + ": " + message)


class MeshGeometryError(MeshKernelError):
    """Exception raised for mesh geometry errors occurring in the MeshKernel library."""

    def __init__(self, message: str, info: Tuple[int, mps.Mesh2dLocation]):
        super().__init__("MeshGeometryError", message)
        self.info = info

    def index(self) -> int:
        return self.info[0]

    def location(self) -> mps.Mesh2dLocation:
        return self.info[1]
