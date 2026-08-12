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


def fft(waveform: np.ndarray):
    n = len(waveform)

    if n <= 1:
        return waveform

    power = 2
    while power * power <= n:
        if n % power == 0:
            break
        power += 1
    else:
        power = n

    if power == n:
        return dft(waveform)

    rem = n // power
    div_fft = []
    for r in range(power):
        div_fft.append(fft(waveform[r::power]))
    result = np.zeros(n, dtype=complex)

    for k1 in range(rem):
        for k0 in range(power):
            k = k0 * rem + k1
            s = 0j
            for r in range(power):
                angel = np.exp(-2j * math.pi * k * r / n)
                s += div_fft[r][k1] * angel
            result[k] = s
    return result

def ifft (waveform: np.ndarray, top_level_flag = True):
    n = len(waveform)
    if n <= 1:
        return waveform

    power = 2
    while power * power <= n:
        if n % power == 0:
            break
        power += 1
    else:
        power = n

    if power == n:
        return idft(waveform)

    rem = n // power
    div_ifft = []
    for r in range(power):
        div_ifft.append(ifft(waveform[r::power], top_level_flag = False))

    result = np.zeros(n, dtype = complex)
    for k1 in range(rem):
        for k0 in range(power):
            k = k0 * rem + k1
            s = 0j
            for r in range(power):
                angle = np.exp(2j * math.pi * k * r / n)
                s += div_ifft[r][k1] * angle
            result[k] = s
    if top_level_flag:
        return result / n
    else:
        return result