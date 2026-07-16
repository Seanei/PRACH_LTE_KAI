import numpy as np

from prach.pipeline.block import Block
from prach.math import idft


class IDFTBlock(Block):
    def transform(self, spectrum: np.ndarray) -> np.ndarray:
        return idft(np.asarray(spectrum, dtype=complex))
