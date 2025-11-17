from __future__ import annotations
from typing import Any, Dict, Type, Optional, get_type_hints
from enum import Enum


class BlockRegistry:
    _registry: Dict[str, Type['Block']] = {}

    @classmethod
    def register(cls, block_cls: Type['Block']):
        cls._registry[block_cls.__name__] = block_cls
        return block_cls

    @classmethod
    def get(cls, name: str) -> Type['Block']:
        return cls._registry[name]


class Block:
    """Base class for all blocks.

    Subclasses should define typed attributes (annotations). Any annotation
    whose name does not start with '_' is required and must be present
    in the YAML config for that block. Attributes starting with '_' are
    internal and are not parsed from YAML

    The constructor will validate types, convert Enum values, and raise
    on missing or invalid values
    """

    def __init__(self, config: Dict[str, Any]):
        cls = self.__class__
        hints = get_type_hints(cls, include_extras=False)

        # build set of fields that should be read from YAML (exclude private)
        yaml_fields = [k for k in hints.keys() if not k.startswith('_')]

        # check missing keys
        for name in yaml_fields:
            if name not in config:
                raise ValueError(f"Missing required field '{name}' for block {cls.__name__}")

        # set attributes with validation/conversion
        for name, expected_type in hints.items():
            if name.startswith('_'):
                # leave internal fields alone or use provided default
                continue
            raw = config.get(name)
            value = self._validate_and_convert(name, raw, expected_type)
            setattr(self, name, value)

    def _validate_and_convert(self, name: str, value: Any, expected_type: Type) -> Any:
        # handle Optional[T] (i.e., Union[T, None])
        origin = getattr(expected_type, '__origin__', None)
        if origin is None:
            base_type = expected_type
            allow_none = False
        elif origin is getattr(__import__('typing'), 'Union'):
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

        # enum handling
        if isinstance(base_type, type) and issubclass(base_type, Enum):
            if isinstance(value, base_type):
                return value
            # allow YAML to give string or int to map to enum
            try:
                return base_type[value] # by enum name
            except Exception:
                # try by value
                for member in base_type:
                    if member.value == value:
                        return member
                raise ValueError(f"Invalid enum value for field '{name}': {value}")
            return value

        # generic fallback: check isinstance
        if not isinstance(value, base_type):
            raise ValueError(f"Field '{name}' expected {base_type} but got {type(value)}")
        
        # we expect yaml parser to handle primitive types itself
        return value

    def process(self, data: CommonData) -> CommonData:
        """Override in subclass. Should return CommonData"""
        raise NotImplementedError
    