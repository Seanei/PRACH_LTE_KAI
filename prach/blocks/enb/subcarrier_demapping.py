import numpy as np

from prach.pipeline.block import Block
from prach.pipeline.spec import PRACHSpecification


class SubcarrierDemappingBlock(Block):
    def demap(self, spectrum: np.ndarray) -> np.ndarray:
        config = self.config
        spec = PRACHSpecification

        spectrum = np.asarray(spectrum, dtype=complex)

        # Frequency scaling factor
        K = config.delta_f // config.delta_f_ra

        n_fft = int(spec.F_S / config.delta_f_ra)

        # Relative PRACH offset from central carrier
        k0 = config.n_ra_prb_offset * spec.N_RB_SC - (config.n_ul_rb * spec.N_RB_SC) / 2

        # Start index of PRACH extraction
        k_start = int(n_fft // 2 + config.phi + K * (k0 + 0.5))

        return spectrum[k_start : k_start + spec.N_ZC_FDD]
