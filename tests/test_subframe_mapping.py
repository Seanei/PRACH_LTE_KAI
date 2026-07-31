import unittest

import numpy as np

from prach.blocks.ue import SubframeMappingBlock
from prach.pipeline import PRACHConfiguration
from prach.pipeline.spec import NUM_SUBFRAMES, SAMPLES_PER_SUBFRAME

# (config_index, ожидаемый формат, номер фрейма, стартовый сабфрейм, длина)
VALID_CONFIGS = [
    (3, 0, 0, 1, 1),  # формат 0
    (19, 1, 1, 1, 2),  # формат 1
    (35, 2, 0, 1, 2),  # формат 2
    (48, 3, 0, 1, 3),  # формат 3
    (31, 1, 0, 9, 2),  # случай с переносом в следующий фрейм для формата 1
    (42, 2, 0, 2, 2),  # формат 2
    # (57, 3, 1, 8, 3),  # случай с переполнением для формата 3
]

# TS 36.211 Table 5.7.1-2 помечает эти индексы как N/A
CONFIGS_WITHOUT_PRACH = [30, 46, 60, 61, 62]


def make_config(config_index):
    config = PRACHConfiguration()
    config.config_index = config_index
    return config


class TestSubframeMapping(unittest.TestCase):

    def test_valid_configs(self):
        for config_index, preamble_format, sf_n, start_sf, num_sf in VALID_CONFIGS:
            with self.subTest(config_index=config_index, sf_n=sf_n):
                config = make_config(config_index)

                # формат задаётся индексом конфигурации, а не отдельно
                self.assertEqual(config.preamble_format, preamble_format)

                preamble = np.ones(num_sf * SAMPLES_PER_SUBFRAME, dtype=np.complex128)
                frame_signal, carry_over = SubframeMappingBlock(config).map(
                    preamble, sf_n
                )

                fit_in_current = num_sf
                if start_sf + num_sf > NUM_SUBFRAMES:
                    fit_in_current = NUM_SUBFRAMES - start_sf
                    self.assertIsNotNone(carry_over, "не сработал перенос")
                else:
                    self.assertIsNone(carry_over)

                filled = range(start_sf, start_sf + fit_in_current)
                for sf_idx in range(NUM_SUBFRAMES):
                    chunk = frame_signal[sf_idx]
                    if sf_idx in filled:
                        self.assertTrue(
                            np.all(chunk == 1.0 + 0j),
                            f"Сабфрейм {sf_idx} должен быть заполнен",
                        )
                    else:
                        self.assertTrue(
                            np.all(chunk == 0j),
                            f"Сабфрейм {sf_idx} должен быть пустым",
                        )

    def test_odd_system_frame_rejected(self):
        # config_index 0 requires an even system frame
        block = SubframeMappingBlock(make_config(0))

        with self.assertRaises(ValueError):
            block.map(np.ones(SAMPLES_PER_SUBFRAME, dtype=np.complex128), sf_n=1)

    def test_configuration_without_prach_rejected(self):
        for config_index in CONFIGS_WITHOUT_PRACH:
            with self.subTest(config_index=config_index):
                with self.assertRaises(ValueError):
                    make_config(config_index).preamble_format


if __name__ == "__main__":
    unittest.main()
