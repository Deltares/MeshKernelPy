import math

import numpy as np
import pytest

from meshkernel import MeshKernel, MeshKernelError


def test_constructor():
    MeshKernel(False)


def test_deallocate():
    meshlib = MeshKernel(False)
    meshlib.deallocate_state()


def test_set_mesh():
    # TODO
    pass
