class Error(Exception):
    """Base class for exceptions in this module."""


class InputError(Error):
    """Exception raised for errors in the input."""


class MeshKernelError(Error):
    """Exception raised for errors coming from the MeshKernel library."""
