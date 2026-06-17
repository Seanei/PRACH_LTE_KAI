import numpy as np

from prach.pipeline import CommonData, Block, BlockRegistry

N_RB_SC = 12  # Number of subcarriers in one LTE resource block
F_S = 30_720_000  # LTE sampling frequency, Hz
N_ZC_FDD = 839  # PRACH Zadoff-Chu sequence length for formats 0-3


@BlockRegistry.register
class SubcarrierDemappingBlock(Block):
    n_ul_rb: int = 100  # Uplink LTE bandwidth configuration
    phi: int = 7  # Standardized PRACH frequency offset
    n_ra_prb_offset: int = 0  # PRACH resource block offset

    delta_f_ra: int = 1250  # PRACH subcarrier spacing, Hz
    delta_f: int = 15_000  # LTE uplink subcarrier spacing, Hz

    def process(self, data: CommonData) -> CommonData:
        # Load configurable parameters
        self.n_ul_rb = data.meta.get(
            "n_ul_rb",
            self.n_ul_rb
        )

        self.n_ra_prb_offset = data.meta.get(
            "n_ra_prb_offset",
            self.n_ra_prb_offset
        )
        # Input LTE FFT spectrum
        spectrum = np.array(data.meta.get("fft", []), dtype=complex)

        # Frequency scaling factor
        K = self.delta_f // self.delta_f_ra

        n_fft = int(F_S / self.delta_f_ra)

        # Relative PRACH offset from central carrier
        k0 = ( self.n_ra_prb_offset * N_RB_SC
            - (self.n_ul_rb * N_RB_SC) / 2
        )

        # Start index of PRACH extraction
        k_start = int(
            n_fft // 2 + self.phi + K * (k0 + 0.5)
        )

        prach_bins = spectrum[
            k_start:k_start + N_ZC_FDD
        ]

        data.meta["Subcarrier_Demapping"] = prach_bins

        return data
    