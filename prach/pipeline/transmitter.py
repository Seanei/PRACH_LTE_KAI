import numpy as np

from .config import PRACHConfiguration


class Transmitter:
    def __init__(self, config: PRACHConfiguration):
        # imported here to avoid a circular import at package load
        from prach.blocks.ue import (
            PreambleGeneratorBlock,
            DFTBlock,
            SubcarrierMappingBlock,
            IDFTBlock,
            SubframeMappingBlock,
        )

        self.config = config
        self.preamble_generator = PreambleGeneratorBlock(config)
        self.dft = DFTBlock(config)
        self.subcarrier_mapping = SubcarrierMappingBlock(config)
        self.idft = IDFTBlock(config)
        self.subframe_mapping = SubframeMappingBlock(config)

    def transmit(self) -> np.ndarray:
        preamble = self.preamble_generator.generate()
        bins = self.dft.transform(preamble)
        spectrum = self.subcarrier_mapping.map(bins)
        signal = self.idft.transform(spectrum)
        frame_signal, _carry_over = self.subframe_mapping.map(signal)
        return frame_signal
