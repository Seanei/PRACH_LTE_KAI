import unittest
import numpy as np
from prach.pipeline import CommonData
from prach.blocks.enb.subframe_demapping import (
    SubframeDemappingBlock,
    F_S,
    NUM_SUBFRAMES,
)


class TestSubframeDemapping(unittest.TestCase):

    def test_simple_subframe_demapping_without_carry_over(self):
        """Тест 1: Извлечение преамбулы без переноса (формат 0, конфигурация 0)"""
        cp_len = 3168
        seq_len = 24576
        samples_per_subframe = int(F_S * 1e-3)

        config = {
            "sf_n": 0,
            "config_index": 0,
            "preamble_format": 0,
            "cp_length": cp_len,
            "sequence_length": seq_len,
        }
        block = SubframeDemappingBlock(config=config)

        data = CommonData()
        data.meta = {}  # <--- ВОТ НАШЕ СПАСЕНИЕ! Создаем пустой словарь
        data.meta["sf_n"] = 0
        data.meta["config_index"] = 0
        data.meta["preamble_format"] = 0
        data.meta["cp_length"] = cp_len
        data.meta["sequence_length"] = seq_len

        frame_signal = np.random.randn(
            NUM_SUBFRAMES, samples_per_subframe
        ) + 1j * np.random.randn(NUM_SUBFRAMES, samples_per_subframe)
        expected_preamble = np.ones(seq_len, dtype=np.complex128) * (5 + 5j)
        frame_signal[1, cp_len : cp_len + seq_len] = expected_preamble
        data.meta["frame_signal"] = frame_signal

        output_data = block.process(data)

        self.assertIsNotNone(output_data)
        prach_windows = output_data.meta.get("prach_windows", [])
        self.assertEqual(len(prach_windows), 1, "Должно быть найдено ровно 1 окно")

        start_sf, extracted_preamble = prach_windows[0]
        self.assertEqual(start_sf, 1, "Преамбула должна начинаться в 1-м субкадре")

        np.testing.assert_array_equal(extracted_preamble, expected_preamble)
        self.assertIsNone(output_data.meta.get("carry_over_prach"))

    def test_subframe_demapping_with_carry_over(self):
        """Тест 2: Преамбула на границе фреймов (формат 1, конфигурация 15)"""
        cp_len = 3168
        seq_len = 24576
        samples_per_subframe = int(F_S * 1e-3)

        config = {
            "sf_n": 0,
            "config_index": 15,
            "preamble_format": 1,
            "cp_length": cp_len,
            "sequence_length": seq_len,
        }
        block = SubframeDemappingBlock(config=config)

        expected_preamble = np.ones(seq_len, dtype=np.complex128) * (7 + 7j)

        frame_1 = np.random.randn(
            NUM_SUBFRAMES, samples_per_subframe
        ) + 1j * np.random.randn(NUM_SUBFRAMES, samples_per_subframe)
        frame_2 = np.random.randn(
            NUM_SUBFRAMES, samples_per_subframe
        ) + 1j * np.random.randn(NUM_SUBFRAMES, samples_per_subframe)

        combined_window = np.concatenate([frame_1[9], frame_2[0]])
        combined_window[cp_len : cp_len + seq_len] = expected_preamble

        frame_1[9] = combined_window[:samples_per_subframe]
        frame_2[0] = combined_window[samples_per_subframe:]

        data1 = CommonData()
        data1.meta = {}
        data1.meta["sf_n"] = 0
        data1.meta["config_index"] = 15
        data1.meta["preamble_format"] = 1
        data1.meta["cp_length"] = cp_len
        data1.meta["sequence_length"] = seq_len
        data1.meta["frame_signal"] = frame_1

        block.process(data1)
        self.assertIsNotNone(
            block._carry_over, "Хвост сигнала должен сохраниться в carry_over"
        )

        data2 = CommonData()
        data2.meta = {}
        data2.meta["sf_n"] = 1
        data2.meta["config_index"] = 15
        data2.meta["preamble_format"] = 1
        data2.meta["cp_length"] = cp_len
        data2.meta["sequence_length"] = seq_len
        data2.meta["frame_signal"] = frame_2

        output_data2 = block.process(data2)

        self.assertIsNotNone(output_data2)
        prach_windows = output_data2.meta.get("prach_windows", [])
        self.assertEqual(len(prach_windows), 1, "1 интервал")

        start_sf, extracted_preamble = prach_windows[0]
        self.assertEqual(start_sf, 9)

        np.testing.assert_array_equal(extracted_preamble, expected_preamble)
        self.assertIsNone(
            block._carry_over, "После склейки carry_over должен очиститься"
        )


if __name__ == "__main__":
    unittest.main()
