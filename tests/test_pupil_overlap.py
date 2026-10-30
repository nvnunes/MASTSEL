"""Prepared pupil overlap preserves pair order, optics and cache ownership."""
import numpy as np
import pytest

import mastsel.pupil_tilt as tilt
from mastsel.mavisPsf import padOrCropCentered

pytestmark = pytest.mark.skipif(tilt._projected_overlap_numba is None,
                                reason='Optional Numba acceleration is unavailable')


@pytest.mark.parametrize('complex_pupil', [False, True])
@pytest.mark.parametrize('mask_kind', ['full', 'annular', 'irregular'])
def test_overlap_kernel_matches_original_pairs_exactly(complex_pupil, mask_kind):
    rng = np.random.default_rng(915)
    support = np.ones((8, 8), bool)
    if mask_kind == 'annular':
        y, x = np.mgrid[:8, :8]
        radius = (x-3.5)**2+(y-3.5)**2
        support = (radius < 16) & (radius > 2)
    elif mask_kind == 'irregular':
        support = rng.uniform(size=(8, 8)) > .4
    local_y, local_x = np.nonzero(support)
    ys, xs = local_y+3, local_x+4
    n = 16
    covariance = rng.uniform(0, 1, (n, n))
    cross = rng.uniform(0, .01, (2, len(xs)))
    tilt_covariance = np.eye(2)*.01
    pupil = rng.uniform(.1, 1., len(xs))
    if complex_pupil:
        pupil = pupil*np.exp(1j*rng.uniform(-.5, .5, len(xs)))
    geometry = tilt._pupil_overlap_runs(support)
    assert sum(array.nbytes for array in geometry) <= 32*1024**2
    arguments = (covariance, cross, tilt_covariance, ys, xs, pupil, n)
    expected = tilt._projected_coherence_numba(*arguments)
    actual = tilt._projected_overlap_numba(*arguments, *geometry)
    np.testing.assert_array_equal(actual, expected)


def test_overlap_run_list_obeys_its_allocation_bound():
    with pytest.raises(MemoryError, match='memory bound'):
        tilt._pupil_overlap_runs(np.ones((8, 8), bool), 1)


def test_support_cache_reuses_geometry_and_replaces_changed_support(monkeypatch):
    pupil = np.ones((6, 6))
    projector = tilt._PupilTiltProjector(pupil, 16, 6)
    coherent = padOrCropCentered(pupil, 16).astype(complex)
    power = np.ones((16, 16))*.001
    original = tilt._pupil_overlap_runs
    calls = []

    def prepare(support):
        calls.append(support.copy())
        return original(support)

    monkeypatch.setattr(tilt, '_pupil_overlap_runs', prepare)
    projector.psf(power, coherent)
    first_geometry = projector._overlap_geometry
    varied = coherent*.7*np.exp(1j*.2)
    actual = projector.psf(2*power, varied)
    assert len(calls) == 1
    assert projector._overlap_geometry is first_geometry
    fresh = tilt._PupilTiltProjector(pupil, 16, 6).psf(2*power, varied)
    np.testing.assert_array_equal(actual[0], fresh[0])
    assert actual[1] == fresh[1]
    varied[6:8, 7] = 0
    actual = projector.psf(power, varied)
    assert projector._overlap_geometry is not first_geometry
    # One build for the reused projector and one for the preceding fresh control.
    assert len(calls) == 3
    fresh = tilt._PupilTiltProjector(pupil, 16, 6).psf(power, varied)
    np.testing.assert_array_equal(actual[0], fresh[0])
    assert actual[1] == fresh[1]


def test_oversized_geometry_uses_original_kernel_without_repeated_preparation(monkeypatch):
    pupil = np.ones((6, 6))
    coherent = padOrCropCentered(pupil, 16)
    power = np.ones((16, 16))*.001
    projector = tilt._PupilTiltProjector(pupil, 16, 6)
    original_kernel = tilt._projected_coherence_numba
    with monkeypatch.context() as legacy:
        legacy.setattr(tilt, '_projected_coherence', lambda *args: original_kernel(*args))
        expected = projector.psf(power, coherent)
    calls = []

    def unavailable(support):
        calls.append(support.shape)
        raise MemoryError('test allocation limit')

    monkeypatch.setattr(tilt, '_pupil_overlap_runs', unavailable)
    for _ in range(2):
        actual = projector.psf(power, coherent)
        np.testing.assert_array_equal(actual[0], expected[0])
        assert actual[1] == expected[1]
    assert len(calls) == 1
    assert projector._overlap_geometry is None
