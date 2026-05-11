from prach.pipeline import CommonData, Block, BlockRegistry
from prach.math import idft


@BlockRegistry.register
class IDFTBlock(Block):
        def process(data:list):
                data.meta.get("Before_IDFT")
                result = idft(data)
                data.meta["After_IDFT"] = result
                return data