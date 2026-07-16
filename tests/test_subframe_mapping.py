import numpy as np
import pytest

from prach.pipeline import PRACHConfiguration
from prach.blocks.ue.subframe_mapping import SubframeMappingBlock

F_S = 30.72e6
NUM_SUBFRAMES = 10
SAMPLES_PER_SF = int(F_S * 1e-3)


@pytest.mark.parametrize(
    "config_index, preamble_format, sf_n, start_sf, num_sf",
    [
        # (config_index, preamble_format, номер фрейма, с какого сабфрейма начинается, длина)
        (3, 0, 0, 1, 1),  # формат 0
        (19, 1, 1, 1, 2),  # формат 1
        (35, 2, 0, 1, 2),  # формат 2
        (48, 3, 0, 1, 3),  # формат 3 (только чётные SFN)
        (31, 1, 0, 9, 2),  # случай с переносом в следующий фрейм для формата 1
        (42, 2, 0, 2, 2),  # формат 2
        # (57, 3, 1, 8, 3),  # случай с переполнением для формата 3
    ]
)
def test_subframe_mapping_valid_configs(config_index, preamble_format, sf_n, start_sf, num_sf):
    config = PRACHConfiguration()
    config.sf_n = sf_n
    config.config_index = config_index
    config.preamble_format = preamble_format

    block = SubframeMappingBlock(config)

    expected_len = num_sf * SAMPLES_PER_SF
    test_preamble = np.ones(expected_len, dtype=np.complex128)

    frame_signal, carry_over = block.map(test_preamble)
    fit_in_current = num_sf

    if start_sf + num_sf > NUM_SUBFRAMES:
        fit_in_current = NUM_SUBFRAMES - start_sf
        assert carry_over is not None, "не сработал перенос"
    else:
        assert carry_over is None

    for i in range(fit_in_current):
        sf_idx = start_sf + i
        chunk = frame_signal[sf_idx]
        assert np.all(chunk == 1.0 + 0j), f"Сабфрейм {sf_idx} должен быть заполнен"

    for sf_idx in range(NUM_SUBFRAMES):
        if not (start_sf <= sf_idx < start_sf + fit_in_current):
            chunk = frame_signal[sf_idx]
            assert np.all(chunk == 0j), f"Сабфрейм {sf_idx} должен быть пустым"


def test_subframe_mapping_invalid_sfn():
    # config_index 0 requires an even system frame
    config = PRACHConfiguration()
    config.sf_n = 1
    config.config_index = 0
    config.preamble_format = 0

    block = SubframeMappingBlock(config)

    with pytest.raises(ValueError):
        block.map(np.ones(SAMPLES_PER_SF, dtype=np.complex128))
