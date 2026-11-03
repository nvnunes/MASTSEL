"""Detector integration composes with projected coherence and full-flux output."""
import numpy as np
import pytest
from scipy import fft

from mastsel.mavisPsf import psdSetToPsfSet


@pytest.mark.parametrize("tilt", ["none", "pupil_fit"])
@pytest.mark.parametrize("static_map", [False, True])
def test_full_flux_pixel_response_integrates_once(tilt, static_map):
    n = 64
    y, x = np.mgrid[-16:16, -16:16]
    mask = (np.hypot(x, y) < 15).astype(float)
    opd = 60 * np.cos(x / 3) * np.sin(y / 4) if static_map else None
    fy, fx = np.mgrid[-32:32, -32:32]
    psd = 1e-16 * np.exp(-((fx / 9)**2 + (fy / 13)**2))
    kwargs = dict(nPixPup=[24], freq_range=[4.], dk=[4. / n], opdMap=opd,
                  full_flux=True, tip_tilt_removal=tilt)

    def run(size, **options):
        return psdSetToPsfSet([[psd]], mask, [1.65e-6], nPixPsf=size, **kwargs, **options)[0]

    native = run(n)
    default = run(n // 2)
    np.testing.assert_array_equal(default.sampling, run(n // 2, pixelResponse=False).sampling)
    np.testing.assert_array_equal(default.sampling, native.sampling.reshape(32, 2, 32, 2).sum((1, 3)))
    response = np.sinc(2 * fft.fftfreq(n))
    integrated = fft.ifft2(fft.fft2(native.sampling) * response[:, None] * response[None, :]).real
    expected = integrated[::2, ::2].copy()
    expected /= expected.sum()
    detector = run(n // 2, pixelResponse=True)
    assert detector.sampling.min() >= -1e-13
    np.testing.assert_allclose(detector.sampling, expected, atol=1e-14, rtol=1e-12)
    if tilt == "pupil_fit":
        assert detector.residual_phase_variance == native.residual_phase_variance
