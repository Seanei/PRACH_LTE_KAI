import math
import random

from prach.pipeline.block import Block
from prach.pipeline.spec import PRACHSpecification
from prach.math import zadoff_chu


def get_shifts(n_zc: int, n_cs: int, u_zc: int = 0) -> list[int]:
    """Cyclic shifts C_v of one root (TS 36.211 5.7.2).

    u_zc = 0 selects the unrestricted set, the actual root value selects
    the restricted set.
    """
    return (
        _get_shifts_unrestricted(n_zc, n_cs)
        if u_zc == 0
        else _get_shifts_restricted(n_zc, n_cs, u_zc)
    )


# TODO: cache it?
def _get_shifts_unrestricted(n_zc: int, n_cs: int) -> list[int]:
    if n_cs == 0:
        return [0]

    num_shifts = math.floor(n_zc / n_cs)
    return [v * n_cs for v in range(num_shifts)]


# TODO: cache it?
def _get_shifts_restricted(n_zc: int, n_cs: int, u_zc: int) -> list[int]:
    if n_cs == 0:
        return [0]

    # (p * u_zc) mod n_zc == 1
    p = 1
    while (p * u_zc) % n_zc != 1:
        p += 1

    d_u = p if p < n_zc / 2 else n_zc - p

    if d_u >= n_zc / 3:
        if d_u > (n_zc - n_cs) / 2:
            return []

        n_shift = math.floor((n_zc - 2 * d_u) / n_cs)
        d_start = n_zc - 2 * d_u + n_shift * n_cs
        n_group = math.floor(d_u / d_start)
        n_shift_avg = min(
            max(math.floor((d_u - n_group * d_start) / n_cs), 0),
            n_shift,
        )
    elif n_cs <= d_u:
        n_shift = math.floor(d_u / n_cs)
        d_start = 2 * d_u + n_shift * n_cs
        n_group = math.floor(n_zc / d_start)
        n_shift_avg = max(math.floor((n_zc - 2 * d_u - n_group * d_start) / n_cs), 0)
    else:
        # no preamble from this root in restricted mode
        return []

    total_shifts = n_shift * n_group + n_shift_avg
    shifts = []
    for v in range(total_shifts):
        c_v = d_start * math.floor(v / n_shift) + (v % n_shift) * n_cs
        shifts.append(c_v)

    return shifts


class PreambleGeneratorBlock(Block):
    _random_system = random.SystemRandom()

    def _generate_preambles(
        self,
        root_sequence_index: int,
        zero_correlation_config: int,
        high_speed_flag: int,
        n_zc: int,
        u_zc_table: list[int],
        n_cs_table: list[list[int]],
        total_preambles: int = 64,
    ) -> list[list[complex]]:
        n_cs = n_cs_table[zero_correlation_config][high_speed_flag]
        preambles = []
        current_root_index = root_sequence_index

        while len(preambles) < total_preambles:
            u_zc = u_zc_table[current_root_index]
            base_seq = zadoff_chu(u_zc, n_zc)

            shifts = get_shifts(n_zc, n_cs, u_zc * high_speed_flag)  # hs = 0 or 1

            for c_v in shifts:
                if len(preambles) >= total_preambles:
                    break
                c = c_v % n_zc
                shifted_seq = base_seq[c:] + base_seq[:c]
                preambles.append(shifted_seq)

            current_root_index += 1
            current_root_index = current_root_index % len(u_zc_table)

        if len(preambles) < total_preambles:
            # TODO: handle as runtime error?
            pass
        return preambles

    def generate(self) -> list[complex]:
        config = self.config
        spec = PRACHSpecification

        fdd = config.preamble_format != 4

        assert fdd, "TDD is currently unsupported"

        preamble_index = config.preamble_index

        if preamble_index == -1:
            preamble_index = self._random_system.randrange(64)

        all_preambles = self._generate_preambles(
            config.root_sequence_index,
            config.zero_correlation_config,
            config.high_speed_flag,
            spec.N_ZC_FDD,
            spec.U_ZC_FDD,
            spec.N_CS_FDD,
            preamble_index + 1,  # small optimization :)
        )

        return all_preambles[preamble_index]
