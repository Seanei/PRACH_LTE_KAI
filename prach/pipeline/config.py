from dataclasses import dataclass, fields
from typing import Any, Dict, Iterable, Optional, Set, Tuple

from .spec import PREAMBLE_FORMAT, TOTAL_PREAMBLES


def _arguments(cls: type, values: Dict[str, Any]) -> Dict[str, Any]:
    arguments: Dict[str, Any] = {}
    for field in fields(cls):
        if field.name not in values:
            continue
        value = values[field.name]
        try:
            arguments[field.name] = field.type(value)
        except (TypeError, ValueError):
            raise ValueError(
                f"Parameter '{field.name}' expected {field.type.__name__}, "
                f"got {value!r}"
            )
    return arguments


def _reject_unknown(values: Iterable[str], taken: Set[str]) -> None:
    unknown = sorted(set(values) - taken)
    if unknown:
        raise ValueError(f"Unknown PRACH configuration parameter '{unknown[0]}'")


@dataclass(frozen=True)
class PRACHConfiguration:
    config_index: int = 0
    root_sequence_index: int = 0
    zero_correlation_config: int = 0  # zeroCorrelationZoneConfig
    high_speed_flag: int = 0
    n_ra_prb_offset: int = 0  # prach-FreqOffset
    n_ul_rb: int = 100  # uplink bandwidth, signalled in SIB2

    @property
    def preamble_format(self) -> int:
        preamble_format = PREAMBLE_FORMAT[self.config_index]
        if preamble_format is None:
            raise ValueError(
                f"PRACH configuration index {self.config_index} carries no "
                f"PRACH opportunity"
            )
        return preamble_format

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "PRACHConfiguration":
        arguments = _arguments(cls, values)
        _reject_unknown(values, set(arguments))
        return cls(**arguments)


class AccessAttempt:
    """Preamble index chosen by one terminal. No attempt is None."""

    def __init__(self, preamble_index: int):
        if not 0 <= preamble_index < TOTAL_PREAMBLES:
            raise ValueError(
                f"preamble_index must be in 0..{TOTAL_PREAMBLES - 1}, got "
                f"{preamble_index}"
            )
        self.preamble_index = preamble_index


@dataclass(frozen=True)
class Deployment:
    max_timing_advance: int = -1  # timing advance units; -1 = the whole zone
    delay_spread: float = 0.0  # seconds


def settings_from_dict(
    values: Dict[str, Any]
) -> Tuple[PRACHConfiguration, Optional[AccessAttempt], Deployment]:
    config_arguments = _arguments(PRACHConfiguration, values)
    deployment_arguments = _arguments(Deployment, values)

    taken: Set[str] = set(config_arguments) | set(deployment_arguments)

    attempt = None
    if "preamble_index" in values:
        attempt = AccessAttempt(int(values["preamble_index"]))
        taken.add("preamble_index")

    _reject_unknown(values, taken)

    return (
        PRACHConfiguration(**config_arguments),
        attempt,
        Deployment(**deployment_arguments),
    )
