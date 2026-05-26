import numpy as np
import math
import cmath
import random


snr_table = {
    2: {
        0: -8.0,
        1: -10.0,
        2: -11.2,
        3: -11.4,
        4: -4.2
    },

    4: {
        0: -10.4,
        1: -12.2,
        2: -13.5,
        3: -13.7,
        4: -5.8
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

    path_delays_samples = []
    for delay_ns in path_delays_ns:

        delay_seconds = delay_ns * 1e-9

        delay_samples = int(
            delay_seconds * sample_rate
        )

        path_delays_samples.append(delay_samples)

    freq_offset = 270
    shifted_signal = []

    for n in range(len(signal)):

        t = n / sample_rate

        phase = 2 * math.pi * freq_offset * t

        shifted = signal[n] * cmath.exp(1j * phase)

        shifted_signal.append(shifted)

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

                faded_signal[n] += (gain * h_doppler * shifted_signal[delayed_index])

    signal_power = 0
    for x in faded_signal:

        signal_power += abs(x) ** 2

    signal_power = signal_power / len(faded_signal)

    snr_linear = 10**(snr_db / 10)

    noise_power = signal_power / snr_linear
    noise_std = math.sqrt(noise_power / 2)

    rx_signal = []
    for x in faded_signal:

        noise_real = random.gauss(0, noise_std)

        noise_imag = random.gauss(0, noise_std)

        noise = complex(noise_real, noise_imag)

        rx_signal.append(x + noise)

    return rx_signal
