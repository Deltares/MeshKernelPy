import numpy as np
import pytest
from numpy.ctypeslib import as_array
from numpy.testing import assert_array_equal

from meshkernel import (
    AveragingMethod,
    DeleteMeshOption,
    InterpolateToOption,
    ProjectToLandBoundaryOption,
    RefinementType,
)

cases_deletemeshoption_values = [
    (DeleteMeshOption.ALL_NODES, 0),
    (DeleteMeshOption.ALL_FACE_CIRCUMCENTERS, 1),
    (DeleteMeshOption.ALL_COMPLETE_FACES, 2),
]


@pytest.mark.parametrize("enum_val, exp_int", cases_deletemeshoption_values)
def test_deletemeshoption_values(enum_val: DeleteMeshOption, exp_int: int):
    assert enum_val == exp_int


cases_projecttolandboundaryoption_values = [
    (ProjectToLandBoundaryOption.DO_NOT_PROJECT_TO_LANDBOUNDARY, 0),
    (ProjectToLandBoundaryOption.TO_ORIGINAL_NETBOUNDARY, 1),
    (ProjectToLandBoundaryOption.OUTER_MESH_BOUNDARY_TO_LANDBOUNDARY, 2),
    (ProjectToLandBoundaryOption.INNER_AND_OUTER_MESH_BOUNDARY_TO_LANDBOUNDARY, 3),
    (ProjectToLandBoundaryOption.WHOLE_MESH, 4),
]


@pytest.mark.parametrize("enum_val, exp_int", cases_projecttolandboundaryoption_values)
def test_projecttolandboundaryoption_values(
    enum_val: ProjectToLandBoundaryOption, exp_int: int
):
    assert enum_val == exp_int


cases_averagingmethod_values = [
    (AveragingMethod.SIMPLE_AVERAGING, 1),
    (AveragingMethod.CLOSEST_POINT, 2),
    (AveragingMethod.MAX, 3),
    (AveragingMethod.MIN, 4),
    (AveragingMethod.INVERSE_WEIGHTED_DISTANCE, 5),
    (AveragingMethod.MINABS, 6),
    (AveragingMethod.KDTREE, 7),
]


@pytest.mark.parametrize("enum_val, exp_int", cases_averagingmethod_values)
def test_averagingmethod_values(enum_val: AveragingMethod, exp_int: int):
    assert enum_val == exp_int


cases_interpolatetooption_values = [
    (InterpolateToOption.BATHY, 1),
    (InterpolateToOption.ZK, 2),
    (InterpolateToOption.S1, 3),
    (InterpolateToOption.ZC, 4),
]


@pytest.mark.parametrize("enum_val, exp_int", cases_interpolatetooption_values)
def test_interpolatetooption_values(enum_val: InterpolateToOption, exp_int: int):
    assert enum_val == exp_int


cases_refinementtype_values = [
    (RefinementType.RIDGE_REFINEMENT, 1),
    (RefinementType.WAVE_COURANT, 2),
    (RefinementType.REFINEMENT_LEVELS, 3),
]


@pytest.mark.parametrize("enum_val, exp_int", cases_refinementtype_values)
def test_refinementtype_values(enum_val: RefinementType, exp_int: int):
    assert enum_val == exp_int
