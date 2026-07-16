import numpy as np

from .config import PRACHConfiguration


class Receiver:
    def __init__(self, config: PRACHConfiguration):
        # imported here to avoid a circular import at package load
        from prach.blocks.ue import DFTBlock
        from prach.blocks.enb import SubcarrierDemappingBlock

        self.config = config
        self.dft = DFTBlock(config)
        self.subcarrier_demapping = SubcarrierDemappingBlock(config)

    def receive(self, signal: np.ndarray) -> np.ndarray:
        pass
