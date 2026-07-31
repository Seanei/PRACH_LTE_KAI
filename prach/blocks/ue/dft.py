import numpy as np

from prach.pipeline.block import Block
from prach.math import dft


class DFTBlock(Block):
    def transform(self, signal: np.ndarray) -> np.ndarray:
        return dft(np.asarray(signal, dtype=complex))
