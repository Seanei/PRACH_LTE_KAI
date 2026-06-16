import random
import numpy as np


def awgn(signal, rx_antennas, burst_format, snr_db):

    signal_power = np.sum(np.abs(signal) ** 2)

    signal_power = signal_power / len(signal)

    snr_linear = 10**(snr_db / 10)

    noise_power = signal_power / snr_linear
    noise_std = np.sqrt(noise_power / 2)

    rx_signal = np.zeros(len(signal), dtype=complex)

    for n, x in enumerate(signal):

        noise_real = random.gauss(0, noise_std)

        noise_imag = random.gauss(0, noise_std)

        noise = complex(noise_real, noise_imag)

        rx_signal[n] = x + noise

    return rx_signal
