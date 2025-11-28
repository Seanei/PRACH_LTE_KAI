from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict


@dataclass(kw_only=True)
class CommonData:
    """Shared data object passed between blocks. Add arbitrary attributes to it

    Stores everything in __dict__ so blocks can set/get arbitrary fields.
    """

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)
