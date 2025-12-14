from typing import Any, Dict, Optional

from .common_data import CommonData
from .block import BlockRegistry


class Pipeline:
    def __init__(self, config: Dict[str, Any]):
        cfg = config.get("config", {})
        self.chain = config.get("chain", list(cfg.keys()))
        self.cfg = cfg

        self.blocks = []
        for block_name in self.chain:
            block_cfg = cfg.get(block_name, {})
            block_cls = BlockRegistry.get(block_name)
            block_instance = block_cls(block_cfg)
            self.blocks.append(block_instance)

    def run(self, data: Optional[CommonData] = None) -> CommonData:
        data = data or CommonData()
        for b in self.blocks:
            data = b.process(data)
            if not isinstance(data, CommonData):
                raise RuntimeError(
                    f"Block {b.__class__.__name__} returned non-CommonData"
                )
        return data
