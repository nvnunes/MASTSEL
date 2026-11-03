"""Integration checks for pupil registration and retained full-flux decimation."""
import numpy as np
import pytest

from mastsel.mavisPsf import psdSetToPsfSet, pupilOnGrid
from mastsel.pupil_tilt import _PupilTiltProjector


def _centred_coordinates(size):
    return 2 * (np.arange(size) - (size - 1) / 2) / size


@pytest.mark.parametrize('opd_side', [32, 48])
def test_fractional_optical_grid_removes_pure_static_tilt(opd_side):
    x, y = np.meshgrid(*[_centred_coordinates(48)] * 2)
    mask = ((np.hypot(x, y) < .90) & (np.hypot(x, y) > .18)).astype(float)
    x, y = np.meshgrid(*[_centred_coordinates(opd_side)] * 2)
    opd = 200 * x - 80 * y
    kwargs = dict(nPixPup=[20.7], freq_range=[4.], dk=[4. / 64],
                  nPixPsf=64, full_flux=True, tip_tilt_removal='pupil_fit')
    psd = [[np.zeros((64, 64))]]
    fitted = psdSetToPsfSet(psd, mask, [1.65e-6], opdMap=opd, **kwargs)[0]
    reference = psdSetToPsfSet(psd, mask, [1.65e-6], **kwargs)[0]
    np.testing.assert_allclose(fitted.sampling, reference.sampling, rtol=1e-11, atol=1e-14)
    assert fitted.residual_phase_variance == 0.


def test_fractional_projected_covariance_matches_independent_mode():
    n, side, width = 16, 12, 7.7
    source_x, source_y = np.meshgrid(*[_centred_coordinates(side)] * 2)
    pupil = (np.hypot(source_x, source_y) < .88).astype(float)
    amplitude = pupilOnGrid(pupil, width, n, np)
    projector = _PupilTiltProjector(pupil, n, width)

    # A stationary covariance from two quadratures of one real Fourier mode.
    power = np.zeros((n, n))
    power[1, 2] = power[-1, -2] = n * n / 2
    actual, variance = projector.psf(power, amplitude, normalize_static=True)
    native_coords = 2 * (np.arange(n) - n // 2) / width
    native_x, native_y = np.meshgrid(native_coords, native_coords)
    source_phase = np.pi * width / n * (2 * source_x + source_y)
    native_phase = np.pi * width / n * (2 * native_x + native_y)
    source_design = np.column_stack((source_x[pupil > 0], source_y[pupil > 0]))
    source_modes = np.stack((np.cos(source_phase), np.sin(source_phase)), axis=-1)
    fit = np.linalg.lstsq(source_design, source_modes[pupil > 0], rcond=None)[0]
    corrected = np.stack((np.cos(native_phase), np.sin(native_phase)), axis=-1)
    corrected -= np.stack((native_x, native_y), axis=-1) @ fit
    yy, xx = np.nonzero(amplitude)
    delta = corrected[yy, xx][:, None, :] - corrected[yy, xx][None, :, :]
    a = amplitude[yy, xx]
    weights = a[:, None] * a[None, :] * np.exp(-.5 * np.sum(delta**2, axis=-1))
    otf = np.zeros((n,n))
    np.add.at(otf, ((yy[:, None]-yy[None, :]) % n, (xx[:, None]-xx[None, :]) % n), weights)
    expected = np.fft.fftshift(np.fft.fft2(otf)).real / np.sum(a*a)
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-14)
    np.testing.assert_allclose(variance, np.average(np.sum(corrected**2, axis=-1), weights=amplitude**2))


@pytest.mark.parametrize('tilt_removal', ['none', 'pupil_fit'])
def test_full_flux_decimates_by_integrating_native_pixel_power(tilt_removal):
    n = 64
    x, y = np.meshgrid(*[_centred_coordinates(32)] * 2)
    mask = (np.hypot(x, y) < .9).astype(float)
    opd = 60 * np.cos(3 * x) * np.sin(2 * y)
    kwargs = dict(nPixPup=[24], freq_range=[4.], dk=[4. / n],
                  full_flux=True, tip_tilt_removal=tilt_removal, opdMap=opd)
    psd = [[np.zeros((n, n))]]
    native = psdSetToPsfSet(psd, mask, [1.65e-6], nPixPsf=n, **kwargs)[0]
    coarse = psdSetToPsfSet(psd, mask, [1.65e-6], nPixPsf=n//2, **kwargs)[0]
    np.testing.assert_array_equal(coarse.sampling,
                                  native.sampling.reshape(n//2, 2, n//2, 2).sum((1, 3)))


def test_fractional_binary_pupil_retains_interpolated_amplitude():
    mask = np.zeros((12, 12), dtype=bool)
    mask[3:9, 3:9] = True
    binary = pupilOnGrid(mask, 7.7, 16, np)
    reference = pupilOnGrid(mask.astype(float), 7.7, 16, np)
    assert np.any((reference > 0) & (reference < 1))
    np.testing.assert_array_equal(binary, reference)
