import math
import cmath
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
    result = []
    N = len(numbers)
    for n in range(numbers):
        sum_value = 0
        for k in range(numbers):
            angle = 2 * math.pi * k * n / N
            sum_value += numbers[k] * complex(math.cos(angle), math.sin(angle))
        sum_value /= N
        result.append(sum_value)
    return result
