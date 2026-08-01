import unittest

import numpy as np

from prach.blocks.enb import SubframeDemappingBlock
from prach.pipeline import PRACHConfiguration
from prach.pipeline.spec import (
    CP_LENGTH,
    NUM_SUBFRAMES,
    SAMPLES_PER_SUBFRAME,
    SEQUENCE_LENGTH,
)


def make_config(config_index):
    config = PRACHConfiguration()
    config.config_index = config_index
    return config


def noise_frame(rng):
    return rng.standard_normal(
        (NUM_SUBFRAMES, SAMPLES_PER_SUBFRAME)
    ) + 1j * rng.standard_normal((NUM_SUBFRAMES, SAMPLES_PER_SUBFRAME))


class TestSubframeDemapping(unittest.TestCase):

    def test_preamble_inside_one_frame(self):
        """Формат 0, конфигурация 0: преамбула целиком в одном субкадре"""
        block = SubframeDemappingBlock(make_config(0))
        preamble_format = block.config.preamble_format
        cp_length = CP_LENGTH[preamble_format]
        sequence_length = SEQUENCE_LENGTH[preamble_format]

        rng = np.random.default_rng(0)
        frame = noise_frame(rng)
        expected = np.ones(sequence_length, dtype=np.complex128) * (5 + 5j)
        frame[1, cp_length: cp_length + sequence_length] = expected

        windows = block.demap(frame)

        self.assertEqual(len(windows), 1)
        start_sf, extracted = windows[0]
        self.assertEqual(start_sf, 1)
        np.testing.assert_array_equal(extracted, expected)
        self.assertIsNone(block.carry_over)

    def test_preamble_across_a_frame_boundary(self):
        """Формат 1, конфигурация 31: преамбула переносится в следующий фрейм"""
        config = make_config(31)
        block = SubframeDemappingBlock(config)
        preamble_format = config.preamble_format
        cp_length = CP_LENGTH[preamble_format]
        sequence_length = SEQUENCE_LENGTH[preamble_format]

        rng = np.random.default_rng(1)
        first, second = noise_frame(rng), noise_frame(rng)

        expected = np.ones(sequence_length, dtype=np.complex128) * (7 + 7j)
        window = np.concatenate([first[9], second[0]])
        window[cp_length: cp_length + sequence_length] = expected
        first[9] = window[:SAMPLES_PER_SUBFRAME]
        second[0] = window[SAMPLES_PER_SUBFRAME:]

        self.assertEqual(block.demap(first, sf_n=0), [])
        self.assertIsNotNone(block.carry_over)

        windows = block.demap(second, sf_n=1)

        self.assertEqual(len(windows), 1)
        start_sf, extracted = windows[0]
        self.assertEqual(start_sf, 9)
        np.testing.assert_array_equal(extracted, expected)
        self.assertIsNone(block.carry_over)

    def test_frame_without_an_opportunity(self):
        """Конфигурация 0 требует чётный фрейм, нечётный не даёт окон"""
        block = SubframeDemappingBlock(make_config(0))

        rng = np.random.default_rng(2)
        self.assertEqual(block.demap(noise_frame(rng), sf_n=1), [])
        self.assertIsNone(block.carry_over)

    def test_configuration_without_prach_rejected(self):
        # index 30 holds no PRACH opportunity at all
        block = SubframeDemappingBlock(make_config(30))

        rng = np.random.default_rng(3)
        with self.assertRaises(ValueError):
            block.demap(noise_frame(rng))


if __name__ == "__main__":
    unittest.main()
