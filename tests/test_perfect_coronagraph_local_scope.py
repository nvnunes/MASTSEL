"""The stationary model leaves existing fitted-tilt/full-flux output alone."""
import numpy as np
import pytest
from mastsel.mavisPsf import psdSetToPsfSet


@pytest.mark.parametrize('options', [{'tip_tilt_removal': 'pupil_fit'}, {'full_flux': True}])
def test_perfect_coronagraph_rejects_incompatible_optics(options):
    with pytest.raises(ValueError, match='stationary optics and full_flux=False'):
        psdSetToPsfSet([], None, 1.65e-6, perfectCoronagraph=True, **options)


def test_default_coronagraph_is_off_for_pupil_fit_and_detector_response():
    n = 64
    y, x = np.mgrid[-16:16, -16:16]
    pupil = (np.hypot(x, y) < 15).astype(float)
    psd = [[np.zeros((n, n))]]
    kw = dict(nPixPup=[24], freq_range=[4.], dk=[4. / n], nPixPsf=32,
              full_flux=True, tip_tilt_removal='pupil_fit', pixelResponse=True)
    implicit = psdSetToPsfSet(psd, pupil, [1.65e-6], **kw)[0]
    explicit = psdSetToPsfSet(psd, pupil, [1.65e-6], perfectCoronagraph=False, **kw)[0]
    np.testing.assert_array_equal(implicit.sampling, explicit.sampling)
