import unittest

import numpy as np

from prach.blocks.enb import PRACHDetector
from prach.pipeline import PRACHConfiguration
from prach.pipeline.spec import (
    DELTA_F_RA,
    F_S,
    N_ZC_FDD,
    T_A_GRANULARITY,
    build_preamble_map,
    n_cs_from_config,
)

# The receiver zero pads the 839 correlated bins to this before transforming back
N_IFFT = 2048


def make_config(zero_correlation_config=1, high_speed_flag=0, config_index=3):
    # zero_correlation_config=1 -> n_cs=13 -> all 64 preambles sit on one root,
    # which is the simplest case that exercises preamble identification
    return PRACHConfiguration(
        config_index=config_index,
        zero_correlation_config=zero_correlation_config,
        high_speed_flag=high_speed_flag,
    )


def expected_timing_advance(delay_bins):
    delay_seconds = delay_bins / (N_IFFT * DELTA_F_RA)
    return round(delay_seconds * F_S / T_A_GRANULARITY)


def profile_with_peak(config, preamble_index, delay_bins=0, height=1.0):
    """One root's power profile holding a single peak for one preamble.

    The peak sits where a signature detector expects that preamble's cyclic
    shift, moved up by `delay_bins` of propagation delay.
    """
    n_cs = n_cs_from_config(
        config.zero_correlation_config, config.high_speed_flag
    )
    slot = build_preamble_map(
        config.root_sequence_index, n_cs, config.high_speed_flag
    )[preamble_index]

    oversampling = N_IFFT / N_ZC_FDD
    start = (N_IFFT - round(slot.c_v * oversampling)) % N_IFFT
    peak_bin = (start + delay_bins) % N_IFFT

    profile = np.zeros(N_IFFT)
    profile[peak_bin] = height
    return profile


class TestSignatureDetector(unittest.TestCase):

    def test_detects_transmitted_preamble_index(self):
        config = make_config()
        pdps = [profile_with_peak(config, preamble_index=7)]

        found = PRACHDetector(config).detect(pdps)

        self.assertEqual([d.preamble_index for d in found], [7])

    def test_measures_timing_advance(self):
        config = make_config()
        delay_bins = 12
        pdps = [profile_with_peak(config, preamble_index=3, delay_bins=delay_bins)]

        found = PRACHDetector(config).detect(pdps)

        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].preamble_index, 3)
        self.assertEqual(
            found[0].timing_advance, expected_timing_advance(delay_bins)
        )

    def test_below_threshold_reports_nothing(self):
        config = make_config()
        # a flat profile: no bin stands above the threshold over the mean
        pdps = [np.ones(N_IFFT)]

        found = PRACHDetector(config, threshold_factor=2.0).detect(pdps)

        self.assertEqual(found, [])

    def test_root_count_matches_preamble_map(self):
        # every preamble of this configuration is one root's cyclic shift
        self.assertEqual(PRACHDetector(make_config()).root_count, 1)


if __name__ == "__main__":
    unittest.main()
