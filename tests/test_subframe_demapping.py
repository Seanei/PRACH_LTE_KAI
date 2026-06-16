import numpy as np
import pytest
from prach.pipeline import CommonData
from prach.blocks.enb.subframe_demapping import SubframeDemappingBlock, F_S, NUM_SUBFRAMES

def test_simple_subframe_demapping_without_carry_over():
    cp_len = 100
    seq_len = 500
    samples_per_subframe = int(F_S * 1e-3) 
    
    block = SubframeDemappingBlock(
        config_index=0,       
        preamble_format=0,    
        cp_length=cp_len,
        sequence_length=seq_len
    )
    
    frame_signal = np.random.randn(NUM_SUBFRAMES, samples_per_subframe) + 1j * np.random.randn(NUM_SUBFRAMES, samples_per_subframe)
    
    expected_preamble = np.ones(seq_len) * (5 + 5j)
    frame_signal[1, cp_len : cp_len + seq_len] = expected_preamble
    
    data = CommonData()
    data.meta["frame_signal"] = frame_signal
    data.meta["sf_n"] = 0 
    
    result = block.process(data)
    
    assert result is not None, "Блок вернул None, хотя должен был найти окно"
    assert "prach_windows" in result.meta
    
    windows = result.meta["prach_windows"]
    assert len(windows) == 1, "Должно быть найдено ровно 1 окно"
    
    start_sf, extracted_preamble = windows[0]
    assert start_sf == 1, "Окно должно было начаться в субфрейме 1"
    assert len(extracted_preamble) == seq_len, f"Длина преамбулы должна быть {seq_len}"
    
    np.testing.assert_array_equal(extracted_preamble, expected_preamble)


def test_subframe_demapping_with_carry_over():
    cp_len = 200
    seq_len = 1200
    samples_per_subframe = int(F_S * 1e-3)
    
    block = SubframeDemappingBlock(
        config_index=15, 
        preamble_format=1, 
        cp_length=cp_len,
        sequence_length=seq_len
    )

    frame_1 = np.zeros((NUM_SUBFRAMES, samples_per_subframe), dtype=np.complex128)
    frame_1[9, :] = np.ones(samples_per_subframe) * 2
    
    data_1 = CommonData()
    data_1.meta["frame_signal"] = frame_1
    data_1.meta["sf_n"] = 0
    
    result_1 = block.process(data_1)
    
    assert result_1.meta.get("prach_windows") is None or len(result_1.meta["prach_windows"]) == 0
    assert block._carry_over is not None, "Кусочек должен был сохраниться в памяти блока!"
    
    frame_2 = np.zeros((NUM_SUBFRAMES, samples_per_subframe), dtype=np.complex128)
    frame_2[0, :] = np.ones(samples_per_subframe) * 3
    
    data_2 = CommonData()
    data_2.meta["frame_signal"] = frame_2
    data_2.meta["sf_n"] = 1
    
    result_2 = block.process(data_2)
    
    assert result_2 is not None
    windows = result_2.meta["prach_windows"]
    assert len(windows) == 1, "Должно собраться склеенное окно"
    
    start_sf, extracted_preamble = windows[0]
    assert start_sf == 9, "Стартовый субфрейм должен остаться равным 9 (из прошлого кадра)"
    assert len(extracted_preamble) == seq_len, "Длина преамбулы должна строго соответствовать sequence_length"
    
    full_mock_window = np.concatenate([frame_1[9], frame_2[0]])
    expected_clean = full_mock_window[cp_len : cp_len + seq_len]
    
    np.testing.assert_array_equal(extracted_preamble, expected_clean)
    assert block._carry_over is None, "После склеивания carry-over должен очиститься!"
    