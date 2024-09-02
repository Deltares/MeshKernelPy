import numpy as np

from meshkernel import (
    GeometryList,
    MakeGridParameters,
    MeshKernel
)


def test_mesh2d_casulli_refinement():
    """Test `mesh2d_casulli_refinement`."""
    mk = MeshKernel()
    mk.mesh2d_make_rectangular_mesh(
        MakeGridParameters(
            num_columns=10,
            num_rows=10,
            block_size_x=1,
            block_size_y=1,
        )
    )
    mesh2d = mk.mesh2d_get()
    mk.mesh2d_casulli_refinement()
    refined_mesh2d = mk.mesh2d_get()
    assert mesh2d.node_x.size < refined_mesh2d.node_x.size


def test_mesh2d_casulli_refinement_on_polygon():
    """Test `mesh2d_casulli_refinement_on_polygon`."""
    mk = MeshKernel()
    mk.mesh2d_make_rectangular_mesh(
        MakeGridParameters(
            num_columns=10,
            num_rows=10,
            block_size_x=1,
            block_size_y=1,
        )
    )
    mesh2d = mk.mesh2d_get()
    polygon_x_coordinates = np.array([2.5, 7.5, 5.5, 2.5], dtype=np.double)
    polygon_y_coordinates = np.array([2.5, 4.5, 8.5, 2.5], dtype=np.double)
    polygon = GeometryList(polygon_x_coordinates, polygon_y_coordinates)
    mk.mesh2d_casulli_refinement_on_polygon(polygon)
    refined_mesh2d = mk.mesh2d_get()
    assert mesh2d.node_x.size < refined_mesh2d.node_x.size


def test_mesh2d_casulli_derefinement():
    """Test `mesh2d_casulli_derefinement`."""

    mk = MeshKernel()
    mk.mesh2d_make_rectangular_mesh(
        MakeGridParameters(
            num_columns=10,
            num_rows=10,
            block_size_x=1,
            block_size_y=1,
        )
    )
    mesh2d = mk.mesh2d_get()
    mk.mesh2d_casulli_derefinement()
    derefined_mesh2d = mk.mesh2d_get()
    assert mesh2d.node_x.size > derefined_mesh2d.node_x.size


def test_mesh2d_casulli_derefinement_on_polygon():
    """Test `mesh2d_casulli_derefinement_on_polygon`."""
    mk = MeshKernel()
    mk.mesh2d_make_rectangular_mesh(
        MakeGridParameters(
            num_columns=10,
            num_rows=10,
            block_size_x=1,
            block_size_y=1,
        )
    )
    mesh2d = mk.mesh2d_get()
    polygon_x_coordinates = np.array([2.5, 7.5, 5.5, 2.5], dtype=np.double)
    polygon_y_coordinates = np.array([2.5, 4.5, 8.5, 2.5], dtype=np.double)
    polygon = GeometryList(polygon_x_coordinates, polygon_y_coordinates)
    mk.mesh2d_casulli_derefinement_on_polygon(polygon)
    derefined_mesh2d = mk.mesh2d_get()
    assert mesh2d.node_x.size > derefined_mesh2d.node_x.size
