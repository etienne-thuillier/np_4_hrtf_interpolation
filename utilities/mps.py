""" python version of Julius O. Smith's mps
    https://ccrma.stanford.edu/~jos/fp/Matlab_listing_mps_m.html
"""
import numpy as np
import numpy.fft as fft

ft = fft.fft
ift = fft.ifft


def mps(s, cutoff_dB, axis):
    """ create minimum-phase spectrum from complex spectrum s

        adapted from https://ccrma.stanford.edu/~jos/fp/Matlab_listing_mps_m.html
    """

    n = len(s.shape)
    permutation = list(range(n))
    permutation[n - 1] = axis
    permutation[axis] = n - 1
    sm = s.transpose(permutation)

    sm = clipdb(sm, cutoff_dB)
    sm = np.log(sm)
    sm = ift(sm, axis=-1)
    sm = fold(sm)
    sm = ft(sm, axis=-1)
    sm = np.exp(sm)

    return sm.transpose(permutation)


def clipdb(s, cutoff_dB):
    """
        floors magnitude of spectrum s according to cutoff in dB defined relative to peak magnitude of s

        adapted from https://ccrma.stanford.edu/~jos/fp/Matlab_listing_clipdb_m.html
    """

    assert cutoff_dB < 0

    clipped = s
    as_ = np.abs(s)
    mas = np.max(as_.flatten())

    if mas == 0:
        return clipped

    thresh = mas * 10 ** (cutoff_dB / 20)
    clipped[as_ < thresh] = thresh

    return clipped


def fold(r):
    """
        [rw] = fold(r)
        Fold left wing of vector in "FFT buffer format"
        onto right wing
        J.O. Smith, 1982-2002

        Adapted from https://ccrma.stanford.edu/~jos/fp/Matlab_listing_fold_m.html#sec:fold
    """

    n = r.shape[-1]

    if n < 3:
        return r

    assert np.mod(n, 2) == 0

    rf = np.concatenate((r[..., 1:r.shape[-1] // 2], np.zeros((*r.shape[:-1], 1), dtype=r.dtype)), axis=-1)
    rf += np.conj(r[..., -1:r.shape[-1] // 2 - 1:-1])
    rw = np.concatenate((r[..., :1], rf, np.zeros((*r.shape[:-1], r.shape[-1] // 2 - 1), dtype=r.dtype)), axis=-1)

    return rw


if __name__ == "__main__":
    pass
