from prach.pipeline import CommonData, Block, BlockRegistry


F_S = 30_720_000  # Hz
NUM_SUBFRAMES = 10
# TS 136 211 v10.0.0 — Table 5.7.1-2 data for frame structure type 1
# system frame (even = 0, any = 1), subframes
# fmt: off
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
# fmt: on


@BlockRegistry.register
class SubframeMappingBlock(Block):
    sf_n: int = 0
    config_index: int = 0

    def process(self, data: CommonData) -> CommonData:
        self.sf_n = data.meta.get("sf_n", self.sf_n)
        self.config_index = data.meta.get("config_index", self.config_index)
        self.preamble_format = data.meta.get("preamble_format", self.preamble_format)

        preamble = data.meta.get("ready_preamble", [])

        samples_per_subframe = int(F_S * 1e-3)

        sf_n_cond = SUBFRAME_CONFIG[self.config_index][0]
        if sf_n_cond != 1 and self.sf_n % 2 != 0:
            return None

        start_sf = SUBFRAME_CONFIG[self.config_index][1][0]
        num_sf = NUM_SF[self.preamble_format]

        if start_sf + num_sf > NUM_SUBFRAMES:
            # TODO: too fat?
            return None

        expected_len = num_sf * samples_per_subframe
        if len(preamble) > expected_len:
            # TODO: too fat
            return None

        preamble_chunks = [
            preamble[i * samples_per_subframe:(i + 1) * samples_per_subframe]
            for i in range(num_sf)
        ]

        frame_signal = [[0j] * samples_per_subframe for _ in range(NUM_SUBFRAMES)]

        for i in range(num_sf):
            frame_signal[start_sf + i] = preamble_chunks[i]

        data.meta["frame_signal"] = frame_signal

        return data