"""Explicit convolution normalization preserves amplitude without mutating inputs."""

import numpy as np
import pytest

from mastsel.mavisPsf import Field, KernelConvolve, convolve, residualToSpectrum


def _inputs(power=0.4):
    psf = Field(1.65e-6, 32, 320.0)
    y, x = np.mgrid[:32, :32] - 16
    psf.sampling = np.exp(-(x*x + y*y) / 18.0)
    psf.sampling *= power / psf.sampling.sum()
    kernel = residualToSpectrum([0.3, 4.0, 7.0], psf.wvl, psf.N, 1 / 320.0)
    return psf, kernel


@pytest.mark.parametrize('power', [0.4, 1.0, 5.0])
@pytest.mark.parametrize('gain', [1.0, 2.0])
def test_unnormalized_convolution_retains_amplitude_and_kernel_gain(power, gain):
    psf, kernel = _inputs(power)
    kernel.sampling *= gain
    original = psf.sampling.copy()
    original_kernel = kernel.sampling.copy()

    result = convolve(psf, kernel, normalize_input=False)

    expected = np.real(KernelConvolve(original, original_kernel))
    np.testing.assert_array_equal(result.sampling, expected)
    assert result.sampling.sum() == pytest.approx(power * gain, rel=1e-12)
    np.testing.assert_array_equal(psf.sampling, original)
    np.testing.assert_array_equal(kernel.sampling, original_kernel)


def test_default_convolution_still_normalizes_the_input():
    psf, kernel = _inputs()
    expected = np.real(KernelConvolve(psf.sampling / psf.sampling.sum(), kernel.sampling))
    implicit = convolve(psf, kernel)
    explicit = convolve(psf, kernel, np, normalize_input=True)
    np.testing.assert_array_equal(implicit.sampling, expected)
    np.testing.assert_array_equal(explicit.sampling, expected)


def test_repeated_unnormalized_convolution_keeps_the_input_power():
    psf, kernel = _inputs()
    first = convolve(psf, kernel, normalize_input=False)
    second = convolve(first, kernel, normalize_input=False)
    assert first.sampling.sum() == pytest.approx(0.4, rel=1e-12)
    assert second.sampling.sum() == pytest.approx(0.4, rel=1e-12)
