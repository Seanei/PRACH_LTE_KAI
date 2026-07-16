from typing import Any, Dict


class PRACHConfiguration:
    def __init__(self):
        self.root_sequence_index: int = 0
        self.preamble_format: int = 0
        self.preamble_index: int = -1
        self.zero_correlation_config: int = 0
        self.high_speed_flag: int = 0

        self.n_ul_rb: int = 100 
        self.phi: int = 7
        self.n_ra_prb_offset: int = 0
        self.delta_f_ra: int = 1250
        self.delta_f: int = 15_000

        self.sf_n: int = 0
        self.config_index: int = 0

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "PRACHConfiguration":
        config = cls()
        for name, value in values.items():
            if not hasattr(config, name):
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
