"""Optional full-plane normalization is independent of the requested crop."""

import numpy as np
import pytest

from mastsel.mavisPsf import psdSetToPsfSet


def _inputs(turbulence=False):
    y, x = np.mgrid[:32, :32] - 15.5
    radius = np.hypot(x, y)
    pupil = ((radius < 14) & (radius > 4)) * (0.8 + 0.1 * x / 16)
    k = np.fft.fftshift(np.fft.fftfreq(96))
    kx, ky = np.meshgrid(k, k)
    psd = 2e-15 * np.exp(-(kx*kx + ky*ky) / 0.03) if turbulence else np.zeros((96, 96))
    opd = 60 * np.sin(x / 4) * np.cos(y / 4)
    return pupil, psd, opd


@pytest.mark.parametrize('oversampling', [1.0, 1.5, 2.0, 3.0])
@pytest.mark.parametrize('wavelength', [0.865e-6, 1.215e-6, 2.2e-6])
@pytest.mark.parametrize('opd_on', [False, True])
@pytest.mark.parametrize('turbulence', [False, True])
def test_full_flux_crop_matches_wide_field(oversampling, wavelength, opd_on, turbulence):
    pupil, psd, opd = _inputs(turbulence)
    kwargs = dict(nPixPup=32, freq_range=8.0, dk=1.0, oversampling=oversampling,
                  opdMap=opd if opd_on else None, full_flux=True)
    wide = psdSetToPsfSet([psd], pupil, wavelength, nPixPsf=256, **kwargs)[0].sampling
    crop = psdSetToPsfSet([psd], pupil, wavelength, nPixPsf=16, **kwargs)[0].sampling
    np.testing.assert_allclose(wide.sum(), 1.0, rtol=2e-6)
    np.testing.assert_array_equal(crop, wide[120:136, 120:136])
    assert 0 < crop.sum() < 1


@pytest.mark.parametrize('internal_grid_mode', ['even_legacy', 'odd_internal'])
def test_full_flux_center_alignment_for_odd_and_even_outputs(internal_grid_mode):
    pupil, psd, _ = _inputs()
    kwargs = dict(nPixPup=32, freq_range=8.0, dk=1.0, oversampling=1.5,
                  full_flux=True, internal_grid_mode=internal_grid_mode)
    wide = psdSetToPsfSet([psd], pupil, 1.65e-6, nPixPsf=257, **kwargs)[0].sampling
    crop = psdSetToPsfSet([psd], pupil, 1.65e-6, nPixPsf=17, **kwargs)[0].sampling
    np.testing.assert_array_equal(crop, wide[120:137, 120:137])
    np.testing.assert_allclose(wide.sum(), 1.0, rtol=2e-6)


def test_full_flux_per_wavelength_exact_decimation():
    pupil, psd, opd = _inputs(True)
    images = psdSetToPsfSet([[psd], [psd]], pupil, [1.215e-6, 2.2e-6],
                            nPixPup=[32, 32], freq_range=[8.0, 8.0], dk=[1.0, 1.0],
                            nPixPsf=32, opdMap=opd, full_flux=True)
    for wavelength in images:
        np.testing.assert_allclose(wavelength[0].sampling.sum(), 1.0, rtol=2e-6)


@pytest.mark.parametrize('opd_on', [False, True])
def test_shared_grid_multiwavelength_crops_keep_the_same_scale(opd_on):
    pupil, psd, opd = _inputs(True)
    kwargs = dict(nPixPup=32, freq_range=8.0, dk=1.0, oversampling=1,
                  opdMap=opd if opd_on else None, full_flux=True)
    wide = psdSetToPsfSet([psd], pupil, [0.485e-6, 2.195e-6], nPixPsf=512, **kwargs)
    crop = psdSetToPsfSet([psd], pupil, [0.485e-6, 2.195e-6], nPixPsf=16, **kwargs)
    for full_images, cropped_images in zip(wide, crop):
        np.testing.assert_allclose(full_images[0].sampling.sum(), 1.0, rtol=2e-6)
        np.testing.assert_array_equal(cropped_images[0].sampling,
                                      full_images[0].sampling[248:264, 248:264])


def test_full_flux_option_leaves_default_output_unchanged():
    pupil, psd, _ = _inputs()
    kwargs = dict(nPixPup=32, freq_range=8.0, dk=1.0, nPixPsf=16)
    default = psdSetToPsfSet([psd], pupil, 1.65e-6, **kwargs)[0].sampling
    explicit = psdSetToPsfSet([psd], pupil, 1.65e-6, full_flux=False, **kwargs)[0].sampling
    np.testing.assert_array_equal(default, explicit)
