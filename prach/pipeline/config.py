from typing import Any, Dict

from .spec import PREAMBLE_FORMAT


class PRACHConfiguration:
    def __init__(self):
        # 3GPP TS 136.331: PRACH-ConfigInfo / PRACH-ConfigSIB
        self.config_index: int = 0  # prach-ConfigIndex
        self.root_sequence_index: int = 0  # rootSequenceIndex
        self.zero_correlation_config: int = 0  # zeroCorrelationZoneConfig
        self.high_speed_flag: int = 0  # highSpeedFlag
        self.n_ra_prb_offset: int = 0  # prach-FreqOffset

        # 3GPP TS 136.331: SystemInformationBlockType2, uplink bandwidth
        self.n_ul_rb: int = 100

        # Picked by the UE for one access attempt, -1 draws a random one
        self.preamble_index: int = -1

    @property
    def preamble_format(self) -> int:
        """Preamble format of this configuration index (TS 36.211 5.7.1-2)."""
        preamble_format = PREAMBLE_FORMAT[self.config_index]
        if preamble_format is None:
            raise ValueError(
                f"PRACH configuration index {self.config_index} carries no "
                f"PRACH opportunity"
            )
        return preamble_format

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "PRACHConfiguration":
        config = cls()
        for name, value in values.items():
            if name not in config.__dict__:
                raise ValueError(f"Unknown PRACH configuration parameter '{name}'")
            default = getattr(config, name)
            try:
                setattr(config, name, type(default)(value))
            except (TypeError, ValueError):
                raise ValueError(
                    f"Parameter '{name}' expected {type(default).__name__}, "
                    f"got {value!r}"
                )
        return config
