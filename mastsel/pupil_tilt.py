"""CPU pupil-pair optics for least-squares removal of two pupil tilts.

This private primitive consumes a stationary Gaussian phase spectrum. It
retains phase/tilt correlations and integrates coherence after projection.
Numba accelerates the serial pair sum when available with JIT enabled; NumPy provides the
fallback in bounded row blocks. Both retain every pair and share the optics.
No additional dependencies or global threading changes are required.
"""

import numpy as np
from numpy import fft

from .mavisPsf import pupilTiltFit

try:
    from numba import config as _numba_config, njit as _njit
except ImportError:
    _njit = None
else:
    if _numba_config.DISABLE_JIT:
        _njit = None


def _projected_coherence_numpy(covariance, cross, tilt_covariance, ys, xs, pupil, pupil_pixels):
    """Sum every pupil pair, sharing calculations for equal separations."""
    n = covariance.shape[0]
    result = np.zeros((n, n), np.complex128)
    y_start, x_start = ys.min(), xs.min()
    height, width = ys.max() - y_start + 1, xs.max() - x_start + 1
    amplitudes = np.zeros((height, width), pupil.dtype)
    cross_x = np.zeros((height, width))
    cross_y = np.zeros((height, width))
    amplitudes[ys - y_start, xs - x_start] = pupil
    cross_x[ys - y_start, xs - x_start], cross_y[ys - y_start, xs - x_start] = cross
    columns = np.arange(width)
    dx = columns[None, :] - columns[:, None]
    bins = (dx + width - 1).ravel()
    displacements = np.arange(1 - width, width)
    rx = 2.0 * dx / pupil_pixels
    # Each block contains all column pairs for a bounded number of pupil rows.
    block_rows = max(1, 262144 // (width * width))
    for dy in range(height):
        ry = 2.0 * dy / pupil_pixels
        offset = covariance[dy % n, dx % n] - covariance[0, 0]
        tilt = 0.5 * (rx * rx * tilt_covariance[0, 0]
                      + 2 * rx * ry * tilt_covariance[0, 1]
                      + ry * ry * tilt_covariance[1, 1])
        for start in range(0, height - dy, block_rows):
            stop = min(start + block_rows, height - dy)
            first, second = slice(start, stop), slice(start + dy, stop + dy)
            exponent = rx * (cross_x[second, None, :] - cross_x[first, :, None])
            exponent += ry * (cross_y[second, None, :] - cross_y[first, :, None])
            exponent += offset
            exponent -= tilt
            weights = amplitudes[second, None, :] * np.conj(amplitudes[first, :, None])
            valid = weights != 0
            np.exp(exponent, out=exponent, where=valid)
            np.multiply(weights, exponent, out=weights, where=valid)
            weights = weights.sum(axis=0).ravel()
            totals = np.bincount(bins, weights=weights.real, minlength=2 * width - 1)
            if np.iscomplexobj(weights):
                totals = totals + 1j * np.bincount(bins, weights=weights.imag,
                                                 minlength=2 * width - 1)
            shifts = displacements
            # Same-row pairs retain positive column displacements only.
            if dy == 0:
                totals, shifts = totals[width:], displacements[width:]
            np.add.at(result, (dy % n, shifts % n), totals)
            np.add.at(result, ((-dy) % n, (-shifts) % n), np.conj(totals))
    result[0, 0] = np.sum(np.abs(pupil) ** 2)
    return result


def _projected_coherence_numba(covariance, cross, tilt_covariance, ys, xs, pupil, pupil_pixels):
    """Sum every pupil pair, sharing calculations for equal separations."""
    n = covariance.shape[0]
    result = np.zeros((n, n), np.complex128)
    y_start, x_start = ys.min(), xs.min()
    height, width = ys.max() - y_start + 1, xs.max() - x_start + 1
    amplitudes = np.zeros((height, width), pupil.dtype)
    cross_x = np.zeros((height, width))
    cross_y = np.zeros((height, width))
    for i in range(len(xs)):
        y, x = ys[i] - y_start, xs[i] - x_start
        amplitudes[y, x] = pupil[i]
        cross_x[y, x], cross_y[y, x] = cross[0, i], cross[1, i]

    # Positive row displacements, plus positive columns on the same row,
    # enumerate each distinct pair once. Zero pupil pixels contribute nothing.
    for dy in range(height):
        for dx in range(1 if dy == 0 else 1 - width, width):
            rx, ry = 2.0 * dx / pupil_pixels, 2.0 * dy / pupil_pixels
            offset = covariance[dy % n, dx % n] - covariance[0, 0]
            tilt = 0.5 * (rx * rx * tilt_covariance[0, 0]
                          + 2 * rx * ry * tilt_covariance[0, 1]
                          + ry * ry * tilt_covariance[1, 1])
            total = 0.0
            for y in range(height - dy):
                for x in range(max(0, -dx), min(width, width - dx)):
                    first, second = amplitudes[y, x], amplitudes[y + dy, x + dx]
                    if first == 0.0j or second == 0.0j:
                        continue
                    exponent = offset
                    exponent += (rx * (cross_x[y + dy, x + dx] - cross_x[y, x])
                                 + ry * (cross_y[y + dy, x + dx] - cross_y[y, x]))
                    exponent -= tilt
                    total += second * np.conj(first) * np.exp(exponent)
            result[dy % n, dx % n] += total
            result[(-dy) % n, (-dx) % n] += np.conj(total)
    result[0, 0] = np.sum(np.abs(pupil) ** 2)
    return result

def _pupil_overlap_runs(support, max_bytes=32 * 1024**2):
    """Build ordered valid-pair row runs within a fixed allocation bound.

    Each displacement retains row/column traversal order. Two passes count and
    then fill runs without allocating individual pupil pairs. The limit covers
    the returned int32 shift, offset and run arrays.
    """
    height, width = support.shape
    n_shifts = (width - 1) + (height - 1) * (2 * width - 1)
    max_runs = (max_bytes - (12 * n_shifts + 4)) // 12
    if max_runs < 0:
        raise MemoryError('Pupil overlap run list exceeds its memory bound')
    count = 0
    for dy in range(height):
        for dx in range(1 if dy == 0 else 1 - width, width):
            for y in range(height - dy):
                inside = False
                for x in range(max(0, -dx), min(width, width - dx)):
                    valid = support[y, x] and support[y + dy, x + dx]
                    if valid and not inside:
                        count += 1
                        if count > max_runs:
                            raise MemoryError('Pupil overlap run list exceeds its memory bound')
                    inside = valid
    shifts = np.empty((n_shifts, 2), np.int32)
    offsets = np.empty(n_shifts + 1, np.int32)
    runs = np.empty((count, 3), np.int32)
    shift_index = 0
    run_index = 0
    for dy in range(height):
        for dx in range(1 if dy == 0 else 1 - width, width):
            shifts[shift_index, 0] = dy
            shifts[shift_index, 1] = dx
            offsets[shift_index] = run_index
            for y in range(height - dy):
                inside = False
                start = 0
                stop = min(width, width - dx)
                for x in range(max(0, -dx), stop):
                    valid = support[y, x] and support[y + dy, x + dx]
                    if valid and not inside:
                        start = x
                    if inside and not valid:
                        runs[run_index, 0] = y
                        runs[run_index, 1] = start
                        runs[run_index, 2] = x
                        run_index += 1
                    inside = valid
                if inside:
                    runs[run_index, 0] = y
                    runs[run_index, 1] = start
                    runs[run_index, 2] = stop
                    run_index += 1
            shift_index += 1
    offsets[shift_index] = run_index
    return shifts, offsets, runs


def _projected_overlap_numba(covariance, cross, tilt_covariance, ys, xs, pupil,
                             pupil_pixels, shifts, offsets, runs):
    """Sum the same ordered pupil pairs using prepared nonzero row runs."""
    n = covariance.shape[0]
    result = np.zeros((n, n), np.complex128)
    y_start, x_start = ys.min(), xs.min()
    height, width = ys.max() - y_start + 1, xs.max() - x_start + 1
    amplitudes = np.zeros((height, width), pupil.dtype)
    cross_x = np.zeros((height, width))
    cross_y = np.zeros((height, width))
    for i in range(len(xs)):
        y, x = ys[i] - y_start, xs[i] - x_start
        amplitudes[y, x] = pupil[i]
        cross_x[y, x], cross_y[y, x] = cross[0, i], cross[1, i]
    for d in range(len(shifts)):
        dy, dx = shifts[d]
        rx, ry = 2.0 * dx / pupil_pixels, 2.0 * dy / pupil_pixels
        offset = covariance[dy % n, dx % n] - covariance[0, 0]
        tilt = 0.5 * (rx * rx * tilt_covariance[0, 0]
                      + 2 * rx * ry * tilt_covariance[0, 1]
                      + ry * ry * tilt_covariance[1, 1])
        total = 0.0
        for r in range(offsets[d], offsets[d + 1]):
            y, start, stop = runs[r]
            for x in range(start, stop):
                first, second = amplitudes[y, x], amplitudes[y + dy, x + dx]
                exponent = offset
                exponent += (rx * (cross_x[y + dy, x + dx] - cross_x[y, x])
                             + ry * (cross_y[y + dy, x + dx] - cross_y[y, x]))
                exponent -= tilt
                total += second * np.conj(first) * np.exp(exponent)
        result[dy % n, dx % n] += total
        result[(-dy) % n, (-dx) % n] += np.conj(total)
    result[0, 0] = np.sum(np.abs(pupil) ** 2)
    return result


_pupil_overlap_runs = (
    _njit(cache=True, nogil=True)(_pupil_overlap_runs) if _njit is not None else None
)
_projected_overlap_numba = (
    _njit(cache=True, nogil=True)(_projected_overlap_numba) if _njit is not None else None
)

_projected_coherence_numba = (
    _njit(cache=True, nogil=True)(_projected_coherence_numba) if _njit is not None else None
)
_projected_coherence = (
    _projected_coherence_numba if _projected_coherence_numba is not None else _projected_coherence_numpy
)


class _PupilTiltProjector:
    """A two-ramp fit on the original pupil, evaluated on an even optical grid.

    ``fit_pupil`` is the real, nonnegative pupil amplitude before native optical
    resampling. Its intensity supplies fit weights (uniform on a binary pupil).
    ``pupil_pixels`` is the native resampled full-telescope pupil width, even
    when the optical aperture is an NGS subaperture. Coordinates follow the
    existing MASTSEL congrid convention. The static phase and the Gaussian
    residual are projected by the same two-mode operator.
    """

    def __init__(self, fit_pupil, n, pupil_pixels):
        self.fit = pupilTiltFit(fit_pupil)
        if n < 2 or n % 2 or pupil_pixels <= 0:
            raise ValueError('pupil_fit requires a positive even native optical grid')
        self.n = int(n)
        self.pupil_pixels = int(pupil_pixels)
        # One compact support cache per projector; amplitudes and phase stay live.
        self._overlap_positions = None
        self._overlap_geometry = None
        self.fit_shape = self.fit.shape[1:]
        fine = self.fit_shape[0]
        positions = self.n / 2 - self.pupil_pixels / 2 + np.arange(fine) * self.pupil_pixels / fine
        transform = np.exp(2j * np.pi * np.fft.fftfreq(self.n)[:, None] * positions[None, :])
        self.kernels = np.asarray([transform @ fit @ transform.T / self.n ** 2 for fit in self.fit])
        yy, xx = np.mgrid[:self.n, :self.n]
        self.ramps = np.stack((2 * (xx - self.n / 2) / self.pupil_pixels + 1 / fine,
                               2 * (yy - self.n / 2) / self.pupil_pixels + 1 / fine))

    def _coherence(self, covariance, cross, tilt_covariance, ys, xs, amplitudes):
        """Reuse bounded overlap runs only for the native Numba kernel.

        The key is the exact optical-pupil support, independent of amplitudes,
        static phase and residual power. A changed support replaces the single
        cache. Oversized geometry uses the original bounded-workspace kernel;
        the NumPy path and explicit kernel selections retain their own behavior.
        """
        if _projected_coherence is _projected_coherence_numba:
            positions = self._overlap_positions
            if (positions is None or not np.array_equal(ys, positions[0])
                    or not np.array_equal(xs, positions[1])):
                self._overlap_geometry = None
                self._overlap_positions = (ys.copy(), xs.copy())
                support = np.zeros((ys.max()-ys.min()+1, xs.max()-xs.min()+1), bool)
                support[ys-ys.min(), xs-xs.min()] = True
                try:
                    self._overlap_geometry = _pupil_overlap_runs(support)
                except MemoryError:
                    pass
            if self._overlap_geometry is not None:
                return _projected_overlap_numba(covariance, cross, tilt_covariance,
                    ys, xs, amplitudes, self.pupil_pixels, *self._overlap_geometry)
        return _projected_coherence(covariance, cross, tilt_covariance,
                                    ys, xs, amplitudes, self.pupil_pixels)

    def remove_static_tilt(self, native_phase, fit_phase):
        """Subtract the original-pupil static fit on the native optical grid.

        Both inputs are radians, on the native grid and fit pupil respectively.
        This matches fitting the total residual after a static OPD is added.
        The supplied maps are read only.
        """
        if np.shape(native_phase) != (self.n, self.n) or np.shape(fit_phase) != self.fit_shape:
            raise ValueError('Static phase must match the native and fit-pupil grids')
        coefficients = np.einsum('iyx,yx->i', self.fit, fit_phase)
        return np.asarray(native_phase) - np.einsum('i,iyx->yx', coefficients, self.ramps)

    def psf(self, phase_spectrum, coherent_pupil, normalize_static=False):
        """Return native PSF pixels and pupil-weighted dynamic phase variance.

        ``phase_spectrum`` is an unshifted real-phase FFT spectrum: its inverse
        FFT is the covariance in radians squared. ``coherent_pupil`` includes
        the static phase after its fit has been removed. Output scaling matches
        native longExposurePsf: total pupil power without an OPD map, or N²
        with the normalized static OTF. Subsequent full_flux handling is shared.
        """
        power = np.asarray(phase_spectrum, dtype=np.float64)
        pupil = np.asarray(coherent_pupil, dtype=np.complex128)
        if power.shape != (self.n, self.n) or pupil.shape != power.shape:
            raise ValueError('Spectrum and coherent pupil must match the native optical grid')
        if not np.all(np.isfinite(power)) or not np.all(np.isfinite(pupil)):
            raise ValueError('Pupil-fit optics require finite phase power and pupil values')
        if power.min() < -1e-12 * max(float(power.max()), np.finfo(float).tiny):
            raise ValueError('Pupil-fit phase power must be nonnegative')
        power = np.maximum(power, 0)
        reverse = (-np.arange(self.n)) % self.n
        power = 0.5 * (power + power[reverse][:, reverse])
        covariance = fft.ifft2(power).real
        cross = np.asarray([self.n ** 2 * fft.ifft2(power * np.conj(k)).real for k in self.kernels])
        tilt_covariance = self.n ** 2 * np.einsum('yx,iyx,jyx->ij', power, self.kernels,
                                                 np.conj(self.kernels)).real
        ys, xs = np.nonzero(pupil)
        if not len(xs):
            raise ValueError('Optical pupil has zero power')
        amplitudes = pupil[ys, xs]
        # A pupil with exactly zero phase needs only real pair weights.
        if not np.any(amplitudes.imag):
            amplitudes = amplitudes.real
        coherence = self._coherence(covariance, cross[:, ys, xs], tilt_covariance,
                                    ys, xs, amplitudes)
        pupil_power = float(np.sum(np.abs(amplitudes) ** 2))
        normalization = pupil_power if normalize_static else self.n ** 2
        image = fft.fftshift(fft.fft2(coherence)).real / normalization
        ramps = self.ramps[:, ys, xs]
        residual_variance = (covariance[0, 0] - 2 * np.sum(ramps * cross[:, ys, xs], axis=0)
                             + np.einsum('iy,ij,jy->y', ramps, tilt_covariance, ramps))
        variance = float(np.average(residual_variance, weights=np.abs(amplitudes) ** 2))
        if variance < -1e-10 * max(float(covariance[0, 0]), 1.0):
            raise ValueError('Projected phase variance is negative')
        return image, max(variance, 0.0)
