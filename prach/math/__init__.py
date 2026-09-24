import math
import cmath
import numpy as np
from typing import List

__all__ = ["zadoff_chu"]


def zadoff_chu(root: int, n_zc: int) -> List[complex]:
    """Zadoff-Chu sequence is a complex-valued mathematical sequence which, when applied to a signal,
    gives rise to a new signal of constant amplitude. When cyclically shifted versions
    of a Zadoff-Chu sequence are imposed upon a signal the resulting set of signals
    detected at the receiver are uncorrelated with one another

    1. Periodic with period 'n_zc'.
    2. If 'n_zc' is prime DFT of a Zadoff-Chu sequence is another Zadoff-Chu sequence conjugated,scaled and time scaled.
    3. The auto correlation of a Zadoff-Chu sequence with a cyclically shifted version of itself is zero
    """

    if n_zc <= 0:
        raise ValueError("'n_zc' must be larger than 0")
    if root <= 0:
        raise ValueError("'root' must be larger than 0")
    if math.gcd(n_zc, root) != 1:
        raise ValueError("'root' must be coprime with 'n_zc'")

    sequence = [0j] * n_zc
    phase = 0

    for n in range(n_zc):
        phase = (-1j * math.pi * root * n * (n + 1)) / n_zc
        sequence[n] = cmath.exp(phase)

    return sequence


def dft(x):

    N = len(x)
    X = np.zeros(N, dtype=complex)

    for k in range(N):
        s = 0j
        for n in range(N):
            angle = -2 * math.pi * k * n / N
            s += x[n] * cmath.exp(1j * angle)
        X[k] = s

    return X


def idft(numbers):
    N = len(numbers)
    result = np.zeros(N, dtype=complex)
    for n in range(N):
        sum_value = 0
        for k in range(N):
            angle = 2 * math.pi * k * n / N
            sum_value += numbers[k] * complex(math.cos(angle), math.sin(angle))
        sum_value /= N
        result[n] = sum_value
    return result


def multi_bef_detect(waveform: np.ndarray, reference: np.ndarray) -> np.ndarray:
    waveform = np.asarray(waveform, dtype=complex)
    reference = np.asarray(reference, dtype=complex)

    if waveform.shape != reference.shape:
        raise ValueError("waveform and reference must have the same shape")

    return waveform * np.conj(reference)


def check_for_power2(n: int) -> bool:
    if n <= 0:
        return False
    while n % 2 == 0:
        n //= 2
    return n == 1


def next_power2(n: int) -> int:
    if n <= 2:
        return 2
    power = 2
    while power < n:
        power *= 2
    return power


def fft_butterfly(waveform: np.ndarray) -> np.ndarray:
    n = len(waveform)
    if n <= 1:
        return waveform

    even = fft_butterfly(waveform[0::2])
    odd = fft_butterfly(waveform[1::2])

    half_n = n // 2
    angles = -2j * math.pi * np.arange(half_n) / n
    twiddle_factors = np.exp(angles)

    odd_twiddled = odd * twiddle_factors

    result = np.zeros(n, dtype=complex)
    result[:half_n] = even + odd_twiddled
    result[half_n:] = even - odd_twiddled
    return result


def ifft_butterfly(waveform: np.ndarray) -> np.ndarray:
    n = len(waveform)
    if n <= 1:
        return waveform

    even = ifft_butterfly(waveform[0::2])
    odd = ifft_butterfly(waveform[1::2])

    half_n = n // 2
    angles = 2j * math.pi * np.arange(half_n) / n
    twiddle_factors = np.exp(angles)

    odd_twiddled = odd * twiddle_factors

    result = np.zeros(n, dtype=complex)
    result[:half_n] = even + odd_twiddled
    result[half_n:] = even - odd_twiddled
    return result


def fft(waveform: np.ndarray) -> np.ndarray:
    n = len(waveform)
    if n <= 1:
        return waveform

    if check_for_power2(n):
        return fft_butterfly(waveform)

    m = next_power2(2 * n - 1)

    angles = np.pi * (np.arange(n) ** 2) / n
    chirp = np.exp(-1j * angles)

    a = np.zeros(m, dtype=complex)
    a[:n] = waveform * chirp

    b = np.zeros(m, dtype=complex)
    b[:n] = np.conj(chirp)
    for i in range(1, n):
        b[m - i] = np.conj(chirp[i])

    fa = fft_butterfly(a)
    fb = fft_butterfly(b)
    fc = ifft_butterfly(fa * fb) / m

    return fc[:n] * chirp


def ifft(waveform: np.ndarray) -> np.ndarray:
    n = len(waveform)
    if n <= 1:
        return waveform

    if check_for_power2(n):
        return ifft_butterfly(waveform) / n

    m = next_power2(2 * n - 1)

    angles = np.pi * (np.arange(n) ** 2) / n
    chirp = np.exp(1j * angles)

    a = np.zeros(m, dtype=complex)
    a[:n] = waveform * chirp

    b = np.zeros(m, dtype=complex)
    b[:n] = np.conj(chirp)
    for i in range(1, n):
        b[m - i] = np.conj(chirp[i])

    fa = fft_butterfly(a)
    fb = fft_butterfly(b)
    fc = ifft_butterfly(fa * fb) / m

    return (fc[:n] * chirp) / n
