"""The public fit shares the original-pupil contract with existing optics."""

import numpy as np
import pytest

from mastsel import pupilTiltFit
from mastsel.pupil_tilt import _PupilTiltProjector


@pytest.mark.parametrize('side', [7, 8])
@pytest.mark.parametrize('obstructed', [False, True])
def test_public_fit_recovers_ramps_and_matches_projector(side, obstructed):
    q = 2*(np.arange(side)-(side-1)/2)/side
    x, y = np.meshgrid(q, q)
    pupil = np.exp(-.2*(x*x+y*y))
    if obstructed:
        pupil[x*x+y*y < .15] = 0
    original = pupil.copy()
    fit = pupilTiltFit(pupil)
    np.testing.assert_array_equal(pupil, original)
    assert fit.shape == (2, side, side) and fit.dtype == np.float64
    np.testing.assert_allclose(np.einsum('iyx,jyx->ij', fit, [x, y]),
                               np.eye(2), atol=2e-15)
    phase = 1.7*x-.4*y+.2*x*y
    design = np.stack((x, y)).reshape(2, -1).T*pupil.ravel()[:, None]
    expected = np.linalg.lstsq(design, (phase*pupil).ravel(), rcond=None)[0]
    np.testing.assert_allclose(np.einsum('iyx,yx->i', fit, phase), expected,
                               atol=2e-15)
    projector = _PupilTiltProjector(pupil, 32, 8)
    np.testing.assert_array_equal(fit, projector.fit)
    assert projector.fit_shape == pupil.shape


@pytest.mark.parametrize('pupil', [
    np.ones(4), np.ones((2, 3)), np.zeros((2, 2)),
    np.array([[1., -1.], [1., 1.]]),
    np.array([[1., np.nan], [1., 1.]]),
    np.array([[1., np.inf], [1., 1.]]),
    np.array([[0., 0.], [0., 1.]]),
])
def test_public_fit_rejects_invalid_or_unconstrained_pupils(pupil):
    with pytest.raises(ValueError):
        pupilTiltFit(pupil)
