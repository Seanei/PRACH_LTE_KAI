import numpy as np
import math
import cmath

from .awgn import awgn


def etu_channel(signal, sample_rate, doppler_freq, freq_offset, snr_db): 

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

    n_terms = 16

    alpha = np.array([
        math.pi * m / (n_terms + 1)
        for m in range(1, n_terms + 1)
    ])

    doppler_freqs = doppler_freq * np.cos(alpha)

    faded_signal = np.zeros(len(signal), dtype=complex)

    for path_index in range(len(path_delays_samples)):

        delay = path_delays_samples[path_index]

        gain_db = path_gains_db[path_index]

        gain = 10 ** (gain_db / 20)

        phi = np.random.uniform(0, 2 * math.pi, n_terms)

        for n in range(len(signal)):

            delayed_index = n - delay
            if delayed_index >= 0:
                t = n / sample_rate

                h_gmeds = (1.0 / np.sqrt(n_terms)) * np.sum(
                    np.exp(1j * (2 * math.pi * doppler_freqs * t + phi))
                )

                faded_signal[n] += (gain * h_gmeds * signal[delayed_index])

    shifted_signal = np.zeros(len(signal), dtype=complex)

    for n in range(len(signal)):

        t = n / sample_rate

        phase = 2 * math.pi * freq_offset * t

        shifted_signal[n] = faded_signal[n] * cmath.exp(1j * phase)

    return awgn(shifted_signal, snr_db)
