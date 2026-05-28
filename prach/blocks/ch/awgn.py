import math
import random
import numpy as np


snr_table = {
    2: {
        0: -14.2,
        1: -14.2,
        2: -16.4,
        3: -16.5,
        4: -7.2
    },

    4: {
        0: -16.9,
        1: -16.7,
        2: -19.0,
        3: -18.8,
        4: -9.8
    }
}


def awgn(signal, rx_antennas, burst_format):

    snr_db = snr_table[rx_antennas][burst_format]

    signal_power = 0
    for x in signal:

        signal_power += abs(x) ** 2

    signal_power = signal_power / len(signal)

    snr_linear = 10**(snr_db / 10)

    noise_power = signal_power / snr_linear
    noise_std = math.sqrt(noise_power / 2)

    rx_signal = np.zeros(len(signal), dtype=complex)

    for n, x in enumerate(signal):

        noise_real = random.gauss(0, noise_std)

        noise_imag = random.gauss(0, noise_std)

        noise = complex(noise_real, noise_imag)

        rx_signal[n] = x + noise

    return rx_signal
