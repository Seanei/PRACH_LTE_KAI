import numpy as np

from .config import PRACHConfiguration
from .spec import CP_LENGTH, SEQUENCE_LENGTH


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

    def transmit(self, sf_n: int = 0) -> np.ndarray:
        preamble_format = self.config.preamble_format

        preamble = self.preamble_generator.generate()
        bins = self.dft.transform(preamble)
        spectrum = self.subcarrier_mapping.map(bins)
        signal = self.idft.transform(spectrum)

        # TS 36.211 5.7.1: formats 2 and 3 send the sequence twice
        sequence = np.tile(signal, SEQUENCE_LENGTH[preamble_format] // len(signal))
        # the cyclic prefix repeats the tail of the sequence ahead of it, so a
        # delayed multipath copy still lands inside the receiver's window
        cp = sequence[-CP_LENGTH[preamble_format]:]
        signal = np.concatenate([cp, sequence])

        # TODO: a preamble starting late enough runs into the next frame; the
        #       tail is dropped here until transmit walks a stream of frames
        frame_signal, _carry_over = self.subframe_mapping.map(signal, sf_n)
        return frame_signal
