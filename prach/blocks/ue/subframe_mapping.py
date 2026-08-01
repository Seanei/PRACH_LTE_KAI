from typing import Optional, Tuple

import numpy as np

from prach.pipeline.block import Block
from prach.pipeline.spec import (
    NUM_SF,
    NUM_SUBFRAMES,
    SAMPLES_PER_SUBFRAME,
    SUBFRAME_CONFIG,
)


class SubframeMappingBlock(Block):
    def map(
        self, preamble: np.ndarray, sf_n: int = 0
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        config = self.config

        preamble = np.asarray(preamble, dtype=np.complex128)
        sf_n_cond, subframes = SUBFRAME_CONFIG[config.config_index]

        if sf_n_cond != 1 and sf_n % 2 != 0:
            raise ValueError(
                f"PRACH configuration index {config.config_index} allows only "
                f"even system frames, got sf_n={sf_n}"
            )

        start_sf = subframes[0]
        num_sf = NUM_SF[config.preamble_format]
        frame_signal = np.zeros(
            (NUM_SUBFRAMES, SAMPLES_PER_SUBFRAME), dtype=np.complex128
        )

        carry_over: Optional[np.ndarray] = None
        if start_sf + num_sf > NUM_SUBFRAMES:
            fit_in_current = NUM_SUBFRAMES - start_sf
            carry_over = preamble[fit_in_current * SAMPLES_PER_SUBFRAME:]
            num_sf = fit_in_current

        for i in range(num_sf):
            start_idx = i * SAMPLES_PER_SUBFRAME
            end_idx = start_idx + SAMPLES_PER_SUBFRAME
            chunk = preamble[start_idx:end_idx]
            frame_signal[start_sf + i, : len(chunk)] = chunk

        return frame_signal, carry_over
