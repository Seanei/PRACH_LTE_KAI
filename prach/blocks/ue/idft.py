from prach.pipeline import CommonData, Block, BlockRegistry
from prach.math import idft


@BlockRegistry.register
class IDFTBlock(Block):
    def process(self, data: CommonData) -> CommonData:
        After_submap = data.meta.get("Before_IDFT")
        result = idft(After_submap)
        data.meta["After_IDFT"] = result
        return data
