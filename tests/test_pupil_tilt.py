"""Pupil-fit optics against independent finite covariance and static limits."""

import numpy as np
import pytest
from scipy import fft

from mastsel.mavisPsf import psdSetToPsfSet, congrid, padOrCropCentered
from mastsel.pupil_tilt import _PupilTiltProjector
from test_full_flux import _inputs


@pytest.mark.parametrize('n,side', [(16, 6), (8, 8)])
@pytest.mark.parametrize('backend', ['numpy', 'numba'])
def test_projected_complex_pupil_matches_dense_covariance(n, side, backend, monkeypatch):
    import mastsel.pupil_tilt as tilt
    kernel = getattr(tilt, '_projected_coherence_' + backend)
    if kernel is None:
        pytest.skip('Optional Numba acceleration is unavailable')
    monkeypatch.setattr(tilt, '_projected_coherence', kernel)
    y, x = np.mgrid[:side, :side]
    center = (side-1)/2
    pupil = ((x-center)**2+(y-center)**2 < (side/2)**2).astype(float)
    pupil[1, 2] = .6
    projector = _PupilTiltProjector(pupil, n, side)
    rng = np.random.default_rng(42)
    power = rng.uniform(.1, 1., (n,n))
    reverse = (-np.arange(n)) % n
    power = .5*(power+power[reverse][:,reverse])
    static = .3*np.sin(x) + .2*y + .07*x*y
    native_phase = padOrCropCentered(static, n)
    coherent = padOrCropCentered(pupil,n)*np.exp(1j*projector.remove_static_tilt(native_phase,static))
    actual, _ = projector.psf(power, coherent, normalize_static=True)
    ys,xs = np.nonzero(coherent)
    covariance = fft.ifft2(power).real
    matrix = covariance[(ys[:,None]-ys[None,:])%n,(xs[:,None]-xs[None,:])%n]
    ramps = projector.ramps[:,ys,xs].T
    weights = pupil[pupil>0]**2
    fit = np.linalg.solve((ramps.T*weights)@ramps,ramps.T*weights)
    transform = np.eye(len(xs))-ramps@fit
    projected = transform@matrix@transform.T
    sf = np.diag(projected)[:,None]+np.diag(projected)[None,:]-2*projected
    amplitudes = coherent[ys,xs]
    terms = amplitudes[:,None]*np.conj(amplitudes[None,:])*np.exp(-.5*sf)
    expected_otf = np.zeros((n,n),complex)
    np.add.at(expected_otf,((ys[:,None]-ys[None,:])%n,(xs[:,None]-xs[None,:])%n),terms)
    expected = fft.fftshift(fft.fft2(expected_otf)).real/np.sum(abs(amplitudes)**2)
    np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)


def test_pure_static_tilt_is_removed_before_imaging():
    pupil, psd, _ = _inputs(False)
    y,x = np.mgrid[:32,:32]
    ramp = 40*(x-15.5)+25*(y-15.5)
    kwargs = dict(nPixPup=32,freq_range=8.,dk=1.,nPixPsf=96,full_flux=True)
    fitted = psdSetToPsfSet([psd],pupil,1.65e-6,opdMap=ramp,
                            tip_tilt_removal='pupil_fit',**kwargs)[0]
    reference = psdSetToPsfSet([psd],pupil,1.65e-6,**kwargs)[0]
    np.testing.assert_allclose(fitted.sampling,reference.sampling,rtol=1e-11,atol=1e-14)
    assert fitted.residual_phase_variance == 0.


@pytest.mark.parametrize('opd_on',[False,True])
def test_pupil_fit_conserves_flux_and_crop_pixels(opd_on):
    pupil,psd,opd = _inputs(True)
    kwargs=dict(nPixPup=32,freq_range=8.,dk=1.,oversampling=1.5,
                opdMap=opd if opd_on else None,full_flux=True,tip_tilt_removal='pupil_fit')
    wide=psdSetToPsfSet([psd],pupil,1.65e-6,nPixPsf=256,**kwargs)[0]
    crop=psdSetToPsfSet([psd],pupil,1.65e-6,nPixPsf=16,**kwargs)[0]
    np.testing.assert_allclose(wide.sampling.sum(),1.,rtol=1e-12)
    np.testing.assert_array_equal(crop.sampling,wide.sampling[120:136,120:136])
    assert 0<crop.sampling.sum()<1
    assert crop.residual_phase_variance == wide.residual_phase_variance


def test_pupil_fit_supports_exact_multiwavelength_decimation():
    pupil,psd,opd=_inputs(True)
    result=psdSetToPsfSet([[psd],[psd]],pupil,[1.2e-6,2.2e-6],nPixPup=[32,24],
                         freq_range=[8.,8.],dk=[1.,1.],nPixPsf=32,opdMap=opd,
                         tip_tilt_removal='pupil_fit',full_flux=True)
    for images in result:
        np.testing.assert_allclose(images[0].sampling.sum(),1.,rtol=1e-12)
        assert images[0].residual_phase_variance>0


def test_fit_pupil_can_differ_from_optical_subaperture():
    pupil,psd,_=_inputs(True)
    aperture=pupil.copy();aperture[:16]=0
    full=psdSetToPsfSet([psd],pupil,1.65e-6,nPixPup=32,freq_range=8.,dk=1.,nPixPsf=96,
                        tip_tilt_removal='pupil_fit',full_flux=True)[0]
    sub=psdSetToPsfSet([psd],aperture,1.65e-6,nPixPup=32,freq_range=8.,dk=1.,nPixPsf=96,
                       tip_tilt_removal='pupil_fit',tilt_fit_pupil=pupil,full_flux=True)[0]
    local=psdSetToPsfSet([psd],aperture,1.65e-6,nPixPup=32,freq_range=8.,dk=1.,nPixPsf=96,
                         tip_tilt_removal='pupil_fit',full_flux=True)[0]
    assert np.isfinite(sub.residual_phase_variance)
    assert not np.allclose(full.sampling,sub.sampling)
    assert not np.allclose(sub.sampling,local.sampling,rtol=1e-6,atol=1e-12)


def test_pupil_fit_respects_requested_output_precision():
    from mastsel.mavisPsf import mastselPsfPrecision
    pupil,psd,_=_inputs(True)
    try:
        mastselPsfPrecision('single')
        image=psdSetToPsfSet([psd],pupil,1.65e-6,nPixPup=32,freq_range=8.,dk=1.,nPixPsf=96,
                             tip_tilt_removal='pupil_fit',full_flux=True)[0]
        assert image.sampling.dtype == np.float32
        np.testing.assert_allclose(image.sampling.sum(),1.,rtol=2e-6)
    finally:
        mastselPsfPrecision('double')


def test_invalid_projection_contracts_are_rejected():
    with pytest.raises(ValueError,match='both fitted tilt'):
        _PupilTiltProjector(np.array([[0.,0.],[0.,1.]]),16,2)
    pupil,psd,_=_inputs(False)
    with pytest.raises(ValueError,match='even_legacy'):
        psdSetToPsfSet([psd],pupil,1.65e-6,nPixPup=32,freq_range=8.,dk=1.,nPixPsf=32,
                       tip_tilt_removal='pupil_fit',internal_grid_mode='odd_internal')


def test_real_pupil_kernel_matches_complex_arithmetic_exactly():
    from mastsel.pupil_tilt import _projected_coherence
    n = 8
    rng = np.random.default_rng(184)
    covariance = rng.uniform(0, 1, (n, n))
    ys, xs = np.nonzero(np.ones((n, n)))
    pupil = rng.uniform(.1, 1., len(xs))
    cross = rng.uniform(0, .01, (2, len(xs)))
    tilt_covariance = np.eye(2)*.01
    arguments = (covariance, cross, tilt_covariance, ys, xs)
    real = _projected_coherence(*arguments, pupil, n)
    complex_values = _projected_coherence(*arguments, pupil.astype(complex), n)
    np.testing.assert_array_equal(real, complex_values)


def test_pupil_fit_import_and_optics_do_not_require_numba_or_czt(monkeypatch):
    import builtins
    import importlib
    import mastsel.pupil_tilt as tilt
    from scipy import signal
    original_import = builtins.__import__

    def without_numba(name, *args, **kwargs):
        if name.split('.')[0] == 'numba':
            raise ImportError('Numba is unavailable in this dependency check')
        return original_import(name, *args, **kwargs)

    try:
        with monkeypatch.context() as unavailable:
            unavailable.setattr(builtins, '__import__', without_numba)
            unavailable.delattr(signal, 'CZT', raising=False)
            importlib.reload(tilt)
            assert tilt._projected_coherence_numba is None
            assert tilt._projected_coherence is tilt._projected_coherence_numpy
            pupil, psd, _ = _inputs(True)
            image = psdSetToPsfSet([psd], pupil, 1.65e-6, nPixPup=32, freq_range=8.,
                                   dk=1., nPixPsf=96, full_flux=True,
                                   tip_tilt_removal='pupil_fit')[0]
            assert np.all(np.isfinite(image.sampling))
            np.testing.assert_allclose(image.sampling.sum(), 1., atol=1e-12)
    finally:
        importlib.reload(tilt)


@pytest.mark.parametrize('opd_on', [False, True])
def test_optional_acceleration_matches_numpy_psf_and_variance(opd_on, monkeypatch):
    import mastsel.pupil_tilt as tilt
    if tilt._projected_coherence_numba is None:
        pytest.skip('Optional Numba acceleration is unavailable')
    pupil, psd, opd = _inputs(True)
    kwargs = dict(nPixPup=32, freq_range=8., dk=1., nPixPsf=96, full_flux=True,
                  oversampling=1.5, tip_tilt_removal='pupil_fit', opdMap=opd if opd_on else None)
    results = []
    for kernel in (tilt._projected_coherence_numpy, tilt._projected_coherence_numba):
        monkeypatch.setattr(tilt, '_projected_coherence', kernel)
        results.append(psdSetToPsfSet([psd], pupil, 1.65e-6, **kwargs)[0])
    np.testing.assert_allclose(results[0].sampling, results[1].sampling, rtol=1e-12, atol=1e-14)
    assert results[0].residual_phase_variance == results[1].residual_phase_variance


def test_disabled_numba_jit_selects_numpy(monkeypatch):
    import mastsel.pupil_tilt as tilt
    import importlib
    config = pytest.importorskip('numba').config
    try:
        with monkeypatch.context() as disabled:
            disabled.setattr(config, 'DISABLE_JIT', 1)
            importlib.reload(tilt)
            assert tilt._projected_coherence_numba is None
            assert tilt._projected_coherence is tilt._projected_coherence_numpy
    finally:
        importlib.reload(tilt)


@pytest.mark.parametrize('vary_mask', [False, True])
@pytest.mark.parametrize('opd_on', [False, True])
def test_shared_optics_match_individual_direction_and_wavelength_calls(vary_mask, opd_on):
    pupil, psd, opd = _inputs(True)
    other = pupil.copy()
    other[:16] = 0
    masks = [pupil, other if vary_mask else pupil]
    spectra = [psd, 2*psd]
    kwargs = dict(nPixPup=32, freq_range=8., dk=1., nPixPsf=96,
                  opdMap=opd if opd_on else None, full_flux=True,
                  tip_tilt_removal='pupil_fit', tilt_fit_pupil=pupil)
    wavelengths = [1.2e-6, 2.2e-6]
    together = psdSetToPsfSet(spectra, masks if vary_mask else pupil, wavelengths, **kwargs)
    for i, wavelength in enumerate(wavelengths):
        for j, (mask, spectrum) in enumerate(zip(masks, spectra)):
            separate = psdSetToPsfSet([spectrum], mask, wavelengths, **kwargs)[i][0]
            np.testing.assert_array_equal(together[i][j].sampling, separate.sampling)
            assert together[i][j].residual_phase_variance == separate.residual_phase_variance
