import numpy as np

from prach.pipeline.block import Block
from prach.pipeline.spec import N_FFT, N_ZC_FDD, prach_subcarrier_start


class SubcarrierMappingBlock(Block):
    def map(self, dft_bins: np.ndarray) -> np.ndarray:
        config = self.config

        frequencies = np.asarray(dft_bins, dtype=complex)
        k_start = prach_subcarrier_start(config.n_ul_rb, config.n_ra_prb_offset)

        # Insert PRACH subcarriers into LTE grid
        spectrum = np.zeros(N_FFT, dtype=complex)
        spectrum[k_start: k_start + N_ZC_FDD] = frequencies

        return spectrum
