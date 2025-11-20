from __future__ import annotations
from typing import Any, Dict, Type, get_type_hints
from enum import Enum

from .common_data import CommonData


class BlockRegistry:
    _registry: Dict[str, Type["Block"]] = {}

    @classmethod
    def register(cls, block_cls: Type["Block"]):
        cls._registry[block_cls.__name__] = block_cls
        return block_cls

    @classmethod
    def get(cls, name: str) -> Type["Block"]:
        return cls._registry[name]


class Block:
    """Base class for all blocks.

    Subclasses should define typed attributes (annotations). Any annotation
    whose name does not start with '_' is required and must be present
    in the YAML config for that block. Attributes starting with '_' are
    internal and are not parsed from YAML.

    The constructor will validate types, convert Enum values, and raise
    on missing or invalid values.
    """

    def __init__(self, config: Dict[str, Any]):
        cls = self.__class__
        hints = get_type_hints(cls, include_extras=False)

        yaml_fields = [k for k in hints.keys() if not k.startswith("_")]

        for name in yaml_fields:
            if name not in config:
                raise ValueError(
                    f"Missing required field '{name}' for block {cls.__name__}"
                )

        for name, expected_type in hints.items():
            if name.startswith("_"):
                continue
            raw = config.get(name)
            value = self._validate_and_convert(name, raw, expected_type)
            setattr(self, name, value)

    def _validate_and_convert(self, name: str, value: Any, expected_type: Type) -> Any:
        # handle Optional[T] (i.e., Union[T, None])
        origin = getattr(expected_type, "__origin__", None)
        if origin is None:
            base_type = expected_type
            allow_none = False
        elif origin is getattr(__import__("typing"), "Union"):
            args = expected_type.__args__
            if type(None) in args and len(args) == 2:
                allow_none = True
                base_type = args[0] if args[1] is type(None) else args[1]
            else:
                base_type = expected_type
                allow_none = False
        else:
            base_type = expected_type
            allow_none = False

        if value is None:
            if allow_none:
                return None
            raise ValueError(f"Field '{name}' is None but not Optional")

        if isinstance(base_type, type) and issubclass(base_type, Enum):
            if isinstance(value, base_type):
                return value
            try:
                return base_type[value]
            except Exception:
                # try by value
                for member in base_type:
                    if member.value == value:
                        return member
                raise ValueError(f"Invalid enum value for field '{name}': {value}")

        if base_type in (int, float, str, bool, dict, list):
            if not isinstance(value, base_type):
                try:
                    if base_type is int:
                        return int(value)
                    if base_type is float:
                        return float(value)
                    if base_type is str:
                        return str(value)
                    if base_type is bool:
                        if isinstance(value, str):
                            if value.lower() in "true":
                                return True
                            if value.lower() == "false":
                                return False
                        return bool(value)
                    if base_type is dict:
                        if isinstance(value, dict):
                            return value
                        raise
                    if base_type is list:
                        if isinstance(value, list):
                            return value
                        raise
                except Exception:
                    raise ValueError(
                        f"Field '{name}' expected {base_type} but got value {value} ({type(value)})"
                    )
            return value

        if not isinstance(value, base_type):
            raise ValueError(
                f"Field '{name}' expected {base_type} but got {type(value)}"
            )
        return value

    def process(self, data: CommonData) -> CommonData:
        """Override in subclass. Should return CommonData"""
        raise NotImplementedError
