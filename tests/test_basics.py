import math

import numpy as np
import pytest

from meshkernel import MeshKernel


def test_initialize():
    meshlib = MeshKernel(lib_path="../lib/MeshKernelApi.dll")
    meshlib.allocate_state(False)
