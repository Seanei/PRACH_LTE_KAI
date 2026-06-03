from prach.pipeline import CommonData, Block, BlockRegistry
import numpy as np


@BlockRegistry.register
class MultiplierBlock(Block):
    def multiplication_for_detection(self, data: CommonData) -> CommonData:
        After_demapping = data.meta.get("After_demapping")
        preambules = data.meta.get("preambules")
        result = After_demapping * preambules
        return result
