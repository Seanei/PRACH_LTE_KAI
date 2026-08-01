from typing import List, Optional, Tuple

import numpy as np

from prach.pipeline.block import Block
from prach.pipeline.config import PRACHConfiguration
from prach.pipeline.spec import (
    CP_LENGTH,
    NUM_SF,
    NUM_SUBFRAMES,
    SEQUENCE_LENGTH,
    SUBFRAME_CONFIG,
)


class SubframeDemappingBlock(Block):
    """Cuts the PRACH preambles out of a radio frame.

    A preamble may start late enough to run past the end of its frame. The
    leading part is then held until the next frame arrives and the two halves
    are joined, so the block carries state between calls and frames have to be
    fed to it in order.
    """

    def __init__(self, config: PRACHConfiguration):
        super().__init__(config)
        self._carry_over: Optional[np.ndarray] = None
        self._carry_over_start_sf: int = 0

    @property
    def carry_over(self) -> Optional[np.ndarray]:
        """Head of a preamble waiting for the rest of it, if any."""
        return self._carry_over

    def demap(
        self, frame_signal: np.ndarray, sf_n: int = 0
    ) -> List[Tuple[int, np.ndarray]]:
        config = self.config

        frame_signal = np.asarray(frame_signal, dtype=np.complex128)
        # raises when the configuration index carries no PRACH opportunity
        preamble_format = config.preamble_format
        num_sf = NUM_SF[preamble_format]
        cp_length = CP_LENGTH[preamble_format]
        sequence_length = SEQUENCE_LENGTH[preamble_format]

        sf_n_cond, subframes = SUBFRAME_CONFIG[config.config_index]

        prach_windows: List[Tuple[int, np.ndarray]] = []

        # a preamble started in the previous frame is completed first, whether
        # or not this frame carries an opportunity of its own
        if self._carry_over is not None:
            remaining_sf = num_sf - (NUM_SUBFRAMES - self._carry_over_start_sf)
            tail = frame_signal[:remaining_sf].flatten()
            window = np.concatenate([self._carry_over, tail])

            prach_windows.append(
                (
                    self._carry_over_start_sf,
                    window[cp_length: cp_length + sequence_length],
                )
            )

            self._carry_over = None
            self._carry_over_start_sf = 0

        if sf_n_cond == 1 or sf_n % 2 == 0:
            for start_sf in subframes:
                end_sf = start_sf + num_sf

                if end_sf <= NUM_SUBFRAMES:
                    window = frame_signal[start_sf:end_sf].flatten()
                    prach_windows.append(
                        (start_sf, window[cp_length: cp_length + sequence_length])
                    )
                else:
                    self._carry_over = frame_signal[start_sf:].flatten()
                    self._carry_over_start_sf = start_sf

        return prach_windows
