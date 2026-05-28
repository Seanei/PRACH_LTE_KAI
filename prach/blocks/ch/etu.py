import numpy as np
import math
import cmath
import random


snr_table = {
    2: {
        0: -8.0,
        1: -7.8,
        2: -10.0,
        3: -10.1,
        4: -0.1
    },

    4: {
        0: -12.1,
        1: -11.7,
        2: -14.1,
        3: -13.9,
        4: -5.1
    }
}


def etu70_channel(signal, rx_antennas, burst_format, sample_rate):

    snr_db = snr_table[rx_antennas][burst_format]

    path_delays_ns = [
        0,
        50,
        120,
        200,
        230,
        500,
        1600,
        2300,
        5000
    ]

    path_gains_db = [
        -1.0,
        -1.0,
        -1.0,
        0.0,
        0.0,
        0.0,
        -3.0,
        -5.0,
        -7.0
    ]

    path_delays_samples = np.zeros(len(path_delays_ns), dtype=int)
    for i, delay_ns in enumerate(path_delays_ns):

        delay_seconds = delay_ns * 1e-9

        delay_samples = int(delay_seconds * sample_rate)

        path_delays_samples[i] = delay_samples

    faded_signal = np.zeros(len(signal), dtype=complex)
    doppler_freq = 70

    for path_index in range(len(path_delays_samples)):

        delay = path_delays_samples[path_index]

        gain_db = path_gains_db[path_index]

        gain = 10 ** (gain_db / 20)

        phi = random.uniform(0, 2 * math.pi)

        h_real = random.gauss(0, 1)
        h_imag = random.gauss(0, 1)
        h = complex(h_real, h_imag) / math.sqrt(2)

        for n in range(len(signal)):

            delayed_index = n - delay
            if delayed_index >= 0:
                t = n / sample_rate

                doppler_phase = (2 * math.pi * doppler_freq * t + phi)

                h_doppler = h * cmath.exp(1j * doppler_phase)

                faded_signal[n] += (gain * h_doppler * signal[delayed_index])

    freq_offset = 270
    shifted_signal = np.zeros(len(signal), dtype=complex)

    for n in range(len(signal)):

        t = n / sample_rate

        phase = 2 * math.pi * freq_offset * t

        shifted = faded_signal[n] * cmath.exp(1j * phase)

        shifted_signal[n] = shifted

    signal_power = 0
    for x in shifted_signal:

        signal_power += abs(x) ** 2

    signal_power = signal_power / len(shifted_signal)

    snr_linear = 10**(snr_db / 10)

    noise_power = signal_power / snr_linear
    noise_std = math.sqrt(noise_power / 2)

    rx_signal = np.zeros(len(shifted_signal), dtype=complex)
    for n, x in enumerate(shifted_signal):

        noise_real = random.gauss(0, noise_std)

        noise_imag = random.gauss(0, noise_std)

        noise = complex(noise_real, noise_imag)

        rx_signal[n] = x + noise

    return rx_signal
