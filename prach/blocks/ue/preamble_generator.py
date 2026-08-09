import random

from prach.pipeline.block import Block
from prach.pipeline.spec import (
    N_ZC_FDD,
    TOTAL_PREAMBLES,
    build_preamble_map,
    n_cs_from_config,
)
from prach.math import zadoff_chu


class PreambleGeneratorBlock(Block):
    _random_system = random.SystemRandom()

    def generate(self) -> list[complex]:
        config = self.config

        # PREAMBLE_FORMAT holds frame structure type 1 only, so the format can
        # never be 4 here; reading it validates the configuration index
        config.preamble_format

        preamble_index = config.preamble_index
        if preamble_index == -1:
            preamble_index = self._random_system.randrange(TOTAL_PREAMBLES)

        n_cs = n_cs_from_config(
            config.zero_correlation_config, config.high_speed_flag
        )
        slots = build_preamble_map(
            config.root_sequence_index,
            n_cs,
            config.high_speed_flag,
            total_preambles=preamble_index + 1,  # small optimization :)
        )

        slot = slots[preamble_index]
        base_sequence = zadoff_chu(slot.u_zc, N_ZC_FDD)

        return base_sequence[slot.c_v:] + base_sequence[: slot.c_v]
