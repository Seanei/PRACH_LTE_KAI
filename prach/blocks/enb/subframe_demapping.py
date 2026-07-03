import numpy as np
from typing import Optional
from prach.pipeline import CommonData, Block, BlockRegistry
from prach.constants import (
    NUM_SUBFRAMES_PER_FRAME,
    PREAMBLE_DURATION_SUBFRAMES,
    SUBFRAME_CONFIG,
)


@BlockRegistry.register
class SubframeDemappingBlock(Block):
    sf_n: int = 0
    config_index: int = 0
    preamble_format: int = 0
    cp_length: int = 3168
    sequence_length: int = 24576

    def process(self, data: CommonData) -> Optional[CommonData]:
        if not hasattr(self, "_carry_over"):
            self._carry_over: Optional[np.ndarray] = None
            self._carry_over_start_sf: int = 0

        sf_n = data.meta.get("sf_n", self.sf_n)
        config_index = data.meta.get("config_index", self.config_index)
        preamble_format = data.meta.get("preamble_format", self.preamble_format)
        cp_length = data.meta.get("cp_length", self.cp_length)
        sequence_length = data.meta.get("sequence_length", self.sequence_length)

        frame_signal = np.array(data.meta.get("frame_signal"), dtype=np.complex128)
        num_sf = PREAMBLE_DURATION_SUBFRAMES[preamble_format]

        prach_windows = []

        if self._carry_over is not None:
            remaining_sf = num_sf - (NUM_SUBFRAMES_PER_FRAME - self._carry_over_start_sf)
            tail = frame_signal[:remaining_sf].flatten()
            full_window = np.concatenate([self._carry_over, tail])

            clean_preamble = full_window[cp_length:cp_length + sequence_length]
            prach_windows.append((self._carry_over_start_sf, clean_preamble))

            self._carry_over = None
            self._carry_over_start_sf = 0

        sf_n_cond = SUBFRAME_CONFIG[config_index][0]
        if sf_n_cond is not None:
            if sf_n_cond == 1 or (sf_n_cond == 0 and sf_n % 2 == 0):
                subframes = SUBFRAME_CONFIG[config_index][1]

                for start_sf in subframes:
                    end_sf = start_sf + num_sf

                    if end_sf <= NUM_SUBFRAMES_PER_FRAME:
                        full_window = frame_signal[start_sf:end_sf].flatten()
                        clean_preamble = full_window[cp_length:cp_length + sequence_length]
                        prach_windows.append((start_sf, clean_preamble))
                    else:
                        partial = frame_signal[start_sf:].flatten()
                        self._carry_over = partial
                        self._carry_over_start_sf = start_sf

        data.meta["prach_windows"] = prach_windows
        data.meta["carry_over_prach"] = self._carry_over
        return data
    