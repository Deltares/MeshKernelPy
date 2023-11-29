import pytest

from meshkernel import MeshKernel
from meshkernel.version import __backend_version__, __version__


def test_get_meshkernel_version():
    """Tests if we can get the version of MeshKernel through the API"""
    mk = MeshKernel()
    meshkernel_version = mk.get_meshkernel_version()
    last_dot = meshkernel_version.rfind(".")
    assert meshkernel_version[:last_dot] == __backend_version__


def test_get_meshkernelpy_version():
    """Tests if we can get the version of MeshKernelPy through the API"""
    mk = MeshKernel()
    meshkernelpy_version = mk.get_meshkernelpy_version()
    assert meshkernelpy_version == __version__
