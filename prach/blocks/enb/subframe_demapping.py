import numpy as np
from typing import Optional
from prach.pipeline import CommonData, Block, BlockRegistry

F_S = 30_720_000
NUM_SUBFRAMES = 10

# TS 136 211 v10.0.0 - Table 5.7.1-2
SUBFRAME_CONFIG = [
    [0, [1]],
    [0, [4]],
    [0, [7]],
    [1, [1]],
    [1, [4]],
    [1, [7]],
    [1, [1, 6]],
    [1, [2, 7]],
    [1, [3, 8]],
    [1, [1, 4, 7]],
    [1, [2, 5, 8]],
    [1, [3, 6, 9]],
    [1, [0, 2, 4, 6, 8]],
    [1, [1, 3, 5, 7, 9]],
    [1, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]],
    [0, [9]],
    [0, [1]],
    [0, [4]],
    [0, [7]],
    [1, [1]],
    [1, [4]],
    [1, [7]],
    [1, [1, 6]],
    [1, [2, 7]],
    [1, [3, 8]],
    [1, [1, 4, 7]],
    [1, [2, 5, 8]],
    [1, [3, 6, 9]],
    [1, [0, 2, 4, 6, 8]],
    [1, [1, 3, 5, 7, 9]],
    [None, None],
    [0, [9]],
    [0, [1]],
    [0, [4]],
    [0, [7]],
    [1, [1]],
    [1, [4]],
    [1, [7]],
    [1, [1, 6]],
    [1, [2, 7]],
    [1, [3, 8]],
    [1, [1, 4, 7]],
    [1, [2, 5, 8]],
    [1, [3, 6, 9]],
    [1, [0, 2, 4, 6, 8]],
    [1, [1, 3, 5, 7, 9]],
    [None, None],
    [0, [9]],
    [0, [1]],
    [0, [4]],
    [0, [7]],
    [1, [1]],
    [1, [4]],
    [1, [7]],
    [1, [1, 6]],
    [1, [2, 7]],
    [1, [3, 8]],
    [1, [1, 4, 7]],
    [1, [2, 5, 8]],
    [1, [3, 6, 9]],
    [None, None],
    [None, None],
    [None, None],
    [0, [9]],
]

NUM_SF = [1, 2, 2, 3]


@BlockRegistry.register
class SubframeDemappingBlock(Block):
    sf_n: int = 0
    config_index: int = 0
    preamble_format: int = 0

    cp_length: int = 3168
    sequence_length: int = 24576

    def process(self, data: CommonData) -> CommonData:
        if not hasattr(self, "_carry_over"):
            self._carry_over: Optional[np.ndarray] = None
            self._carry_over_start_sf: int = 0

        self.sf_n = data.meta.get("sf_n", self.sf_n)
        self.config_index = data.meta.get("config_index", self.config_index)
        self.preamble_format = data.meta.get("preamble_format", self.preamble_format)

        self.cp_length = data.meta.get("cp_length", self.cp_length)
        self.sequence_length = data.meta.get("sequence_length", self.sequence_length)

        frame_signal = np.array(data.meta.get("frame_signal"), dtype=np.complex128)
        num_sf = NUM_SF[self.preamble_format]

        prach_windows = []

        if self._carry_over is not None:
            remaining_sf = num_sf - (NUM_SUBFRAMES - self._carry_over_start_sf)
            tail = frame_signal[:remaining_sf].flatten()
            full_window = np.concatenate([self._carry_over, tail])

            clean_preamble = full_window[
                self.cp_length:self.cp_length + self.sequence_length
            ]
            prach_windows.append((self._carry_over_start_sf, clean_preamble))

            self._carry_over = None
            self._carry_over_start_sf = 0

        sf_n_cond = SUBFRAME_CONFIG[self.config_index][0]
        if sf_n_cond is not None:
            if sf_n_cond == 1 or (sf_n_cond == 0 and self.sf_n % 2 == 0):
                subframes = SUBFRAME_CONFIG[self.config_index][1]

                for start_sf in subframes:
                    end_sf = start_sf + num_sf

                    if end_sf <= NUM_SUBFRAMES:
                        full_window = frame_signal[start_sf:end_sf].flatten()
                        clean_preamble = full_window[
                            self.cp_length:self.cp_length + self.sequence_length
                        ]
                        prach_windows.append((start_sf, clean_preamble))

                    else:
                        partial = frame_signal[start_sf:].flatten()
                        self._carry_over = partial
                        self._carry_over_start_sf = start_sf

        if not prach_windows and self._carry_over is None:
            return None

        data.meta["prach_windows"] = prach_windows
        data.meta["carry_over_prach"] = self._carry_over
        return data
