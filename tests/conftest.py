import pytest

from meshkernel import Mesh2dFactory, MeshKernel


@pytest.fixture(scope="function")
def meshkernel_with_mesh2d():
    """Creates a new instance of 'meshkernel' and sets a Mesh2d with the specified dimensions.

    Args:
        rows (int): Number of node rows
        columns (int): Number of node columns

    Returns:
        MeshKernel: The created instance of `meshkernel`
    """

    def _create(rows: int, columns: int):
        mesh2d = Mesh2dFactory.create(rows, columns)
        mk = MeshKernel()

        mk.mesh2d_set(mesh2d)

        return mk

    return _create
