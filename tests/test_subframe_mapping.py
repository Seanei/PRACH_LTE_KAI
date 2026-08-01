import numpy as np
import pytest

from prach.blocks.ue import SubframeMappingBlock
from prach.pipeline import PRACHConfiguration
from prach.pipeline.spec import NUM_SUBFRAMES, SAMPLES_PER_SUBFRAME


def make_config(config_index):
    config = PRACHConfiguration()
    config.config_index = config_index
    return config


@pytest.mark.parametrize(
    "config_index, preamble_format, sf_n, start_sf, num_sf",
    [
        # (config_index, ожидаемый формат, номер фрейма, стартовый сабфрейм, длина)
        (3, 0, 0, 1, 1),  # формат 0
        (19, 1, 1, 1, 2),  # формат 1
        (35, 2, 0, 1, 2),  # формат 2
        (48, 3, 0, 1, 3),  # формат 3
        (31, 1, 0, 9, 2),  # случай с переносом в следующий фрейм для формата 1
        (42, 2, 0, 2, 2),  # формат 2
        # (57, 3, 1, 8, 3),  # случай с переполнением для формата 3
    ]
)
def test_subframe_mapping_valid_configs(config_index, preamble_format, sf_n, start_sf, num_sf):
    config = make_config(config_index)

    # формат задаётся индексом конфигурации, а не отдельно
    assert config.preamble_format == preamble_format

    preamble = np.ones(num_sf * SAMPLES_PER_SUBFRAME, dtype=np.complex128)
    frame_signal, carry_over = SubframeMappingBlock(config).map(preamble, sf_n)

    fit_in_current = num_sf
    if start_sf + num_sf > NUM_SUBFRAMES:
        fit_in_current = NUM_SUBFRAMES - start_sf
        assert carry_over is not None, "не сработал перенос"
    else:
        assert carry_over is None

    filled = range(start_sf, start_sf + fit_in_current)
    for sf_idx in range(NUM_SUBFRAMES):
        chunk = frame_signal[sf_idx]
        if sf_idx in filled:
            assert np.all(chunk == 1.0 + 0j), f"Сабфрейм {sf_idx} должен быть заполнен"
        else:
            assert np.all(chunk == 0j), f"Сабфрейм {sf_idx} должен быть пустым"


def test_subframe_mapping_invalid_sfn():
    # config_index 0 requires an even system frame
    block = SubframeMappingBlock(make_config(0))

    with pytest.raises(ValueError):
        block.map(np.ones(SAMPLES_PER_SUBFRAME, dtype=np.complex128), sf_n=1)


# TS 36.211 Table 5.7.1-2 помечает эти индексы как N/A
@pytest.mark.parametrize("config_index", [30, 46, 60, 61, 62])
def test_configuration_without_prach_rejected(config_index):
    with pytest.raises(ValueError):
        make_config(config_index).preamble_format
