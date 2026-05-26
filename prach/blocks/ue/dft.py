from prach.pipeline import CommonData, Block, BlockRegistry
from prach.math import dft


@BlockRegistry.register
class DFTBlock(Block):
    def process(self, data: CommonData) -> CommonData:
        preamble = data.meta.get("generated_preamble", [])

        data.meta["after_dft"] = dft(preamble)

        return data
