from dataclasses import dataclass, field
from typing import Dict, Any
import numpy as np

from prach.pipeline import Pipeline, CommonData
from prach.blocks.ue.subframe_mapping import SubframeMappingBlock
from prach.blocks.enb.subframe_demapping import SubframeDemappingBlock

cfg = {
    "config": {
        "SubframeMappingBlock": {
            "sf_n": 0,
            "config_index": 0,
            "preamble_format": 0
        },
        "SubframeDemappingBlock": {
            "sf_n": 0,
            "config_index": 0,
            "preamble_format": 0,
            "cp_length": 3168,
            "sequence_length": 24576
        }
    },
    "chain": ["SubframeMappingBlock", "SubframeDemappingBlock"],
}

pipeline = Pipeline(cfg)
data = CommonData()
data.meta = {}

F_S = 30_720_000
samples_per_subframe = int(F_S * 1e-3)  # 30720 отсчетов
cp_length = 3168
sequence_length = 24576

np.random.seed(42)
reference_sequence = np.random.randn(sequence_length) + 1j * np.random.randn(sequence_length)

cp = np.zeros(cp_length, dtype=np.complex128)
guard_size = samples_per_subframe - cp_length - sequence_length
guard_period = np.zeros(guard_size, dtype=np.complex128)

ready_preamble = np.concatenate([cp, reference_sequence, guard_period])

data.meta["ready_preamble"] = ready_preamble
data.meta["sf_n"] = 0
data.meta["config_index"] = 0
data.meta["preamble_format"] = 0

print("Запуск сквозного пайплайна Передатчик -> Приемник...")
result = pipeline.run(data)

output_windows = result.meta.get("prach_windows")

print("\n--- Результаты mapper-теста ---")
if output_windows:
    extracted_sf, extracted_sequence = output_windows[0]
    print(f"Сигнал успешно передан в сабфрейме: {extracted_sf}")
    
    max_diff = np.max(np.abs(reference_sequence - extracted_sequence))
    print(f"Максимальное различие отсчетов: {max_diff}")
    
    assert max_diff < 1e-12, "Ошибка: Извлеченный сигнал искажен"
    print("Сигнал на выходе демаппера идентичен исходному сигналу на входе")
else:
    print("Ошибка: Приемник не смог выделить временные окна")
