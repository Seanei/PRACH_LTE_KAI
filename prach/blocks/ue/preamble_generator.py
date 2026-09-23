import random
from typing import Optional

from prach.pipeline.block import Block
from prach.pipeline.config import AccessAttempt
from prach.pipeline.spec import (
    N_ZC_FDD,
    TOTAL_PREAMBLES,
    build_preamble_map,
    n_cs_from_config,
)
from prach.math import zadoff_chu


class PreambleGeneratorBlock(Block):
    _random_system = random.SystemRandom()

    def generate(self, attempt: Optional[AccessAttempt] = None) -> list[complex]:
        """The preamble of one access attempt; None draws a random one."""
        config = self.config

        # PREAMBLE_FORMAT holds frame structure type 1 only, so the format can
        # never be 4 here; reading it validates the configuration index
        config.preamble_format

        preamble_index = (
            self._random_system.randrange(TOTAL_PREAMBLES)
            if attempt is None
            else attempt.preamble_index
        )

        n_cs = n_cs_from_config(
            config.zero_correlation_config, config.high_speed_flag
        )
        slots = build_preamble_map(
            config.root_sequence_index,
            n_cs,
            config.high_speed_flag,
        )

        slot = slots[preamble_index]
        base_sequence = zadoff_chu(slot.u_zc, N_ZC_FDD)

        return base_sequence[slot.c_v:] + base_sequence[: slot.c_v]
