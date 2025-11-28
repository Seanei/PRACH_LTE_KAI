"""Example usage:
`python -m prach --base-config configs/showcase_settings_base.yaml --override configs/showcase_settings.yaml`
"""

import argparse
import pathlib
import pprint
from enum import Enum
from dataclasses import dataclass, field
from copy import deepcopy
from typing import Any, Dict, Optional

import yaml
from prach.pipeline import CommonData, Block, BlockRegistry, Pipeline


def load_yaml(path: pathlib.Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f.read()) or {}


def deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Return new dict that's a deep merge of a and b (b overrides a)"""
    merged = deepcopy(a)
    for k, v in b.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = deep_merge(merged[k], v)
        else:
            merged[k] = deepcopy(v)
    return merged


def build_config_from_files(
    base_config_path: pathlib.Path, override_config_path: Optional[pathlib.Path]
) -> Dict[str, Any]:
    base = load_yaml(base_config_path)
    over = {}
    if override_config_path is not None:
        over = load_yaml(override_config_path)

    print("Base config:")
    pprint.pprint(base)
    print("")
    print("Override config:")
    pprint.pprint(over)

    return deep_merge(base, over)


@dataclass(kw_only=True)
class CommonDataEx(CommonData):
    meta: Dict[str, Any] = field(default_factory=dict)


class Mode(Enum):
    FAST = "fast"
    ACCURATE = "accurate"


@BlockRegistry.register
class BlockTest(Block):
    field1: str
    mode: Mode
    _field2: Optional[str] = None

    def process(self, data: CommonData) -> CommonData:
        data.meta["BlockTest.field1"] = self.field1
        data.meta["BlockTest.mode"] = self.mode.value

        prev = data.meta.get("combined", [])
        prev.append(self.field1)
        data.meta["combined"] = prev

        self._field2 = f"computed-{self.field1}"
        data.meta["_field2"] = self._field2
        return data


@BlockRegistry.register
class TestBlock(Block):
    field1: str

    def process(self, data: CommonData) -> CommonData:
        prev = data.meta.get("combined", [])
        prev.append(self.field1)
        data.meta["combined"] = prev
        return data


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-config",
        "-b",
        required=True,
        help="YAML settings file (can be overriden)",
    )
    parser.add_argument(
        "--override",
        "-o",
        required=False,
        help="Optional YAML configuration to merge from",
    )

    args = parser.parse_args(argv)

    base_config_path = pathlib.Path(args.base_config)
    override_config_path = pathlib.Path(args.override) if args.override else None

    config = build_config_from_files(base_config_path, override_config_path)

    print()
    print("Merged config:")
    pprint.pprint(config)

    pipeline = Pipeline(config)
    result = pipeline.run(CommonDataEx())

    print()
    print("Pipeline result meta:")
    pprint.pprint(result.meta)


if __name__ == "__main__":
    main()
