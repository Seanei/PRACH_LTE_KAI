"""Subframe mapping and demapping have to be exact inverses of each other."""

import unittest

import numpy as np

from prach.blocks.enb import SubframeDemappingBlock
from prach.blocks.ue import SubframeMappingBlock
from prach.pipeline import PRACHConfiguration
from prach.pipeline.spec import (
    CP_LENGTH,
    SAMPLES_PER_SUBFRAME,
    SEQUENCE_LENGTH,
)


def make_preamble(preamble_format, num_sf, rng):
    """A preamble laid out as the UE builds it: CP, sequence, then guard."""
    cp_length = CP_LENGTH[preamble_format]
    sequence_length = SEQUENCE_LENGTH[preamble_format]

    sequence = rng.standard_normal(sequence_length) + 1j * rng.standard_normal(
        sequence_length
    )
    guard = np.zeros(
        num_sf * SAMPLES_PER_SUBFRAME - cp_length - sequence_length, dtype=np.complex128
    )
    preamble = np.concatenate(
        [np.zeros(cp_length, dtype=np.complex128), sequence, guard]
    )
    return preamble, sequence


class TestSubframeRoundTrip(unittest.TestCase):

    def test_preamble_survives_mapping_and_demapping(self):
        config = PRACHConfiguration(config_index=0)

        rng = np.random.default_rng(42)
        preamble, sequence = make_preamble(config.preamble_format, 1, rng)

        frame_signal, carry_over = SubframeMappingBlock(config).map(preamble, sf_n=0)
        self.assertIsNone(carry_over)

        windows = SubframeDemappingBlock(config).demap(frame_signal, sf_n=0)

        self.assertEqual(len(windows), 1)
        start_sf, extracted = windows[0]
        self.assertEqual(start_sf, 1)
        np.testing.assert_allclose(extracted, sequence, atol=1e-12)

    def test_preamble_split_over_two_frames_survives(self):
        # index 31 puts a format 1 opportunity in subframe 9; the preamble
        # spans two subframes and so runs into the next frame
        config = PRACHConfiguration(config_index=31)

        rng = np.random.default_rng(7)
        preamble, sequence = make_preamble(config.preamble_format, 2, rng)

        first_frame, carry_over = SubframeMappingBlock(config).map(preamble, sf_n=0)
        self.assertIsNotNone(carry_over)

        demapper = SubframeDemappingBlock(config)
        self.assertEqual(demapper.demap(first_frame, sf_n=0), [])

        # the tail of the preamble opens the next frame
        second_frame = np.zeros_like(first_frame)
        second_frame[0, : len(carry_over)] = carry_over

        windows = demapper.demap(second_frame, sf_n=1)

        self.assertEqual(len(windows), 1)
        start_sf, extracted = windows[0]
        self.assertEqual(start_sf, 9)
        np.testing.assert_allclose(extracted, sequence, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
