"""Example usage:
`python -m prach --base-config configs/showcase_settings_base.yaml --override configs/showcase_settings.yaml`
"""

import argparse
import pathlib
import pprint
from copy import deepcopy
from typing import Any, Dict, Optional

import yaml
from prach.pipeline import PRACHConfiguration, Transmitter, Receiver


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

    merged = build_config_from_files(base_config_path, override_config_path)

    config = PRACHConfiguration.from_dict(merged)

    print()
    print("PRACH configuration:")
    pprint.pprint(vars(config))

    transmitter = Transmitter(config)
    receiver = Receiver(config)

    print()
    print(
        f"Transmitter blocks: {[type(b).__name__ for b in (
        transmitter.preamble_generator, transmitter.dft,
        transmitter.subcarrier_mapping, transmitter.idft,
        transmitter.subframe_mapping)]}"
    )
    print(
        f"Receiver blocks: {[type(b).__name__ for b in (
        receiver.dft, receiver.subcarrier_demapping)]}"
    )


if __name__ == "__main__":
    main()
