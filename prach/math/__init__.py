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


def idft(numbers: complex):
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


def Multi_Bef_Detect(numbers: complex, format: int = 0):
    N = len(numbers)
    result = np.zeros((N, 64), dtype=complex)
    cp_inside = np.zeros((N, 64), dtype=complex)
    # Here will be a reference signal from generate preamble block, when I will be able to use it outside of ue
    # for i in range(64):
    # generate preambles here
    # here will be fft function of reference signal
    # plug for reference signal so that none errors would occure
    reference_fft = np.zeros((N, 64), dtype=complex)
    if format == 2 or format == 3:
        n_corr = 2
    else:
        n_corr = 1
    for i in range(n_corr):
        for j in range(64):
            ref_fft_column = reference_fft[:, j]
            cp_inside[:, j] += numbers * np.conj(ref_fft_column)
    result = cp_inside / math.sqrt(n_corr)
    return result
