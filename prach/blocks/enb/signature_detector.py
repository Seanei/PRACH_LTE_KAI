from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from prach.pipeline.block import Block
from prach.pipeline.config import PRACHConfiguration
from prach.pipeline.spec import (
    DELTA_F_RA,
    F_S,
    N_ZC_FDD,
    T_A_GRANULARITY,
    TOTAL_PREAMBLES,
    PreambleSlot,
    build_preamble_map,
    n_cs_from_config,
)


@dataclass
class PRACHDetection:
    """One preamble a signature detector reports in a PRACH window."""

    preamble_index: int
    root_offset: int
    cyclic_shift: int
    peak_index: int  # the bin the peak sits at in the profile
    peak_energy: float
    delay_bins: int  # the peak's offset inside its cyclic-shift zone
    timing_advance: int


class PRACHDetector(Block):
    def __init__(self, config: PRACHConfiguration, *, threshold_factor: float = 10.0):
        """
        threshold_factor
            How many times the profile mean a peak must clear to count. The
            mean stands in for the noise floor; a plain multiple of it is the
            simplest threshold that does not depend on the signal's scale.
        """
        super().__init__(config)

        if threshold_factor < 0:
            raise ValueError(
                f"threshold_factor must be >= 0, got {threshold_factor}"
            )

        self.threshold_factor = threshold_factor

    def _preamble_map(self) -> List[PreambleSlot]:
        config = self.config

        # raises when the configuration index carries no PRACH opportunity
        config.preamble_format

        n_cs = n_cs_from_config(
            config.zero_correlation_config, config.high_speed_flag
        )
        return build_preamble_map(
            config.root_sequence_index,
            n_cs,
            config.high_speed_flag,
            total_preambles=TOTAL_PREAMBLES,
        )

    @property
    def root_count(self) -> int:
        """Roots the cell spreads its preambles over, one profile each."""
        return self._preamble_map()[-1].root_offset + 1

    @staticmethod
    def _zone(slot: PreambleSlot, n_cs: int, length: int) -> np.ndarray:
        """The bins one preamble's cyclic-shift zone covers, in delay order."""
        if n_cs == 0:
            # a single preamble per root: the whole profile belongs to it
            return np.arange(length)

        oversampling = length / N_ZC_FDD
        width = max(round(n_cs * oversampling), 1)
        start = (length - round(slot.c_v * oversampling)) % length
        return (start + np.arange(width)) % length

    @staticmethod
    def _timing_advance(delay_bins: int, length: int) -> int:
        delay_seconds = delay_bins / (length * DELTA_F_RA)
        return round(delay_seconds * F_S / T_A_GRANULARITY)

    def detect(self, pdps: Sequence[np.ndarray]) -> List[PRACHDetection]:
        """Preambles present in one power delay profile per root.

        `pdps` is one profile per root, already summed over the receive
        branches. Real input is taken as power; complex input is squared, so
        the block can be driven straight from a correlation profile too.
        """
        config = self.config
        n_cs = n_cs_from_config(
            config.zero_correlation_config, config.high_speed_flag
        )

        profiles = [
            np.abs(profile) ** 2
            if np.iscomplexobj(profile)
            else np.asarray(profile, dtype=float)
            for profile in pdps
        ]

        detections: List[PRACHDetection] = []
        for slot in self._preamble_map():
            profile = profiles[slot.root_offset]

            zone = self._zone(slot, n_cs, profile.size)
            energies = profile[zone]

            local_peak = int(np.argmax(energies))
            peak_energy = float(energies[local_peak])

            if peak_energy <= self.threshold_factor * float(profile.mean()):
                continue

            detections.append(
                PRACHDetection(
                    preamble_index=slot.preamble_index,
                    root_offset=slot.root_offset,
                    cyclic_shift=slot.c_v,
                    peak_index=int(zone[local_peak]),
                    peak_energy=peak_energy,
                    delay_bins=local_peak,
                    timing_advance=self._timing_advance(local_peak, profile.size),
                )
            )

        return detections
