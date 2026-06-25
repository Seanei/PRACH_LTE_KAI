from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from prach.pipeline import CommonData, Block, BlockRegistry
from prach.blocks.ue.preamble_generator import N_CS_FDD, N_ZC_FDD

F_S = 30_720_000
T_A_GRANULARITY = 16
T_A_MAX = 1282
TOTAL_PREAMBLES = 64


@dataclass
class Detection:
    preamble_index: int
    peak_index: int
    peak_value: float
    delay_samples: int
    delay_seconds: float
    timing_advance: int
    root_index: int = 0


@BlockRegistry.register
class DetectorBlock(Block):
    zero_correlation_config: int = 0
    high_speed_flag: int = 0
    threshold_factor: float = 5.0
    delta_f_ra: int = 1250

    def process(self, data: CommonData) -> CommonData:
        self.zero_correlation_config = data.meta.get(
            "zero_correlation_config", self.zero_correlation_config
        )
        self.high_speed_flag = data.meta.get("high_speed_flag", self.high_speed_flag)
        self.threshold_factor = data.meta.get("threshold_factor", self.threshold_factor)

        pdps = data.meta.get("pdps")

        n_cs = N_CS_FDD[self.zero_correlation_config][self.high_speed_flag]
        shifts_per_root = N_ZC_FDD // n_cs if n_cs else 1

        detections: List[Detection] = []
        noise_floors: List[float] = []
        thresholds: List[float] = []

        for root_index, raw in enumerate(pdps):
            pdp = self._to_power(raw)
            root_detections, noise_floor, threshold = self._detect(pdp, n_cs)
            noise_floors.append(noise_floor)
            thresholds.append(threshold)

            for det in root_detections:
                global_index = root_index * shifts_per_root + det.preamble_index
                if global_index >= TOTAL_PREAMBLES:
                    continue
                det.preamble_index = global_index
                det.root_index = root_index
                detections.append(det)

        data.meta["detected_preambles"] = detections
        data.meta["detected_indices"] = [d.preamble_index for d in detections]
        data.meta["timing_advances"] = [d.timing_advance for d in detections]
        data.meta["pdp_noise_floor"] = noise_floors
        data.meta["pdp_threshold"] = thresholds

        return data

    @staticmethod
    def _to_power(pdp) -> np.ndarray:
        arr = np.asarray(pdp)
        if arr.size == 0:
            return np.zeros(0, dtype=float)
        if np.iscomplexobj(arr):
            return np.abs(arr) ** 2
        return arr.astype(float)

    def _windows(self, length: int, n_cs: int) -> List[Tuple[int, int, int]]:
        if n_cs == 0:
            return [(0, 0, length)]

        oversampling = length / N_ZC_FDD
        width = max(int(round(n_cs * oversampling)), 1)
        num_shifts = N_ZC_FDD // n_cs

        windows = []
        for v in range(num_shifts):
            start = int(round(v * width)) % length
            start = (length - start) % length
            windows.append((v, start, width))
        return windows

    def _detect(
        self, pdp: np.ndarray, n_cs: int
    ) -> Tuple[List[Detection], float, float]:
        length = pdp.size
        if length == 0:
            return [], 0.0, 0.0

        noise_floor = float(np.mean(pdp))
        threshold = self.threshold_factor * noise_floor

        detections: List[Detection] = []
        for v, start, width in self._windows(length, n_cs):
            idx = (start + np.arange(width)) % length
            window_power = pdp[idx]

            local_peak = int(np.argmax(window_power))
            peak_value = float(window_power[local_peak])
            if peak_value <= threshold:
                continue

            delay_seconds = local_peak / (length * self.delta_f_ra)
            timing_advance = round(delay_seconds * F_S / T_A_GRANULARITY)
            timing_advance = max(0, min(timing_advance, T_A_MAX))

            detections.append(
                Detection(
                    preamble_index=v,
                    peak_index=int(idx[local_peak]),
                    peak_value=peak_value,
                    delay_samples=local_peak,
                    delay_seconds=delay_seconds,
                    timing_advance=timing_advance,
                )
            )

        return detections, noise_floor, threshold
