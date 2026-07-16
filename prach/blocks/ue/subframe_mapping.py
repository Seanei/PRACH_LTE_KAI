from typing import Optional, Tuple

import numpy as np

from prach.pipeline.block import Block
from prach.pipeline.spec import PRACHSpecification


class SubframeMappingBlock(Block):
    def map(self, preamble: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        config = self.config
        spec = PRACHSpecification

        preamble = np.asarray(preamble, dtype=np.complex128)
        samples_per_subframe = int(spec.F_S * 1e-3)
        sf_n_cond = spec.SUBFRAME_CONFIG[config.config_index][0]

        if sf_n_cond != 1 and config.sf_n % 2 != 0:
            raise ValueError(
                f"PRACH configuration index {config.config_index} allows only "
                f"even system frames, got sf_n={config.sf_n}"
            )

        start_sf = spec.SUBFRAME_CONFIG[config.config_index][1][0]
        num_sf = spec.NUM_SF[config.preamble_format]
        frame_signal = np.zeros(
            (spec.NUM_SUBFRAMES, samples_per_subframe), dtype=np.complex128
        )

        carry_over: Optional[np.ndarray] = None
        if start_sf + num_sf > spec.NUM_SUBFRAMES:
            fit_in_current = spec.NUM_SUBFRAMES - start_sf
            carry_over = preamble[fit_in_current * samples_per_subframe :]
            num_sf = fit_in_current

        for i in range(num_sf):
            start_idx = i * samples_per_subframe
            end_idx = start_idx + samples_per_subframe
            chunk = preamble[start_idx:end_idx]
            frame_signal[start_sf + i, : len(chunk)] = chunk

        return frame_signal, carry_over
