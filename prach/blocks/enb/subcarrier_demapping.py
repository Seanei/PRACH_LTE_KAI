import numpy as np

from prach.pipeline.block import Block
from prach.pipeline.spec import N_ZC_FDD, prach_subcarrier_start


class SubcarrierDemappingBlock(Block):
    def demap(self, spectrum: np.ndarray) -> np.ndarray:
        config = self.config

        spectrum = np.asarray(spectrum, dtype=complex)
        k_start = prach_subcarrier_start(config.n_ul_rb, config.n_ra_prb_offset)

        return spectrum[k_start: k_start + N_ZC_FDD]
