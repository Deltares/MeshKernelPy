import pytest
from tests.mesh2d_factory import Mesh2dFactory

from meshkernel import MeshKernel


@pytest.fixture(scope="function")
def meshkernel_with_mesh2d():
    """Creates a new instance of 'meshkernel' and sets a Mesh2d with the specified dimensions.

    Args:
        rows (int): Number of node rows
        columns (int): Number of node columns

    Returns:
        MeshKernel: The created instance of `meshkernel`
    """

    def _create(rows: int, columns: int, spacing_x: int = 1.0, spacing_y: int = 1.0):
        mesh2d = Mesh2dFactory.create(
            rows=rows, columns=columns, spacing_x=spacing_x, spacing_y=spacing_y
        )
        mk = MeshKernel()

        mk.mesh2d_set(mesh2d)

        return mk

    return _create
