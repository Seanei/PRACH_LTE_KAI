import math
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Sequence

# 3GPP TS 136.211: Table 5.7.2-1
N_ZC_FDD = 839  # PRACH Zadoff-Chu sequence length for formats 0-3
N_ZC_TDD = 139  # format 4

# Preambles a cell offers
TOTAL_PREAMBLES = 64

# 3GPP TS 136.211: Table 5.7.2-2
N_CS_FDD = [
    (0, 15),
    (13, 18),
    (15, 22),
    (18, 26),
    (22, 32),
    (26, 38),
    (32, 46),
    (38, 55),
    (46, 68),
    (59, 82),
    (76, 100),
    (93, 128),
    (119, 158),
    (167, 202),
    (279, 237),
    (419, None),
]

# 3GPP TS 136.211: Table 5.7.2-2
N_CS_TDD = [
    2,
    4,
    6,
    8,
    10,
    12,
    15,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
]

# Zaddof-Chu sequence root number
# 3GPP TS 136.211: Table 5.7.2-4
# fmt: off
U_ZC_FDD = [
    129, 710, 140, 699, 120, 719, 210, 629, 168, 671, 84, 755, 105, 734, 93, 746, 70, 769, 60, 779,
    2, 837, 1, 838,
    56, 783, 112, 727, 148, 691,
    80, 759, 42, 797, 40, 799,
    35, 804, 73, 766, 146, 693,
    31, 808, 28, 811, 30, 809, 27, 812, 29, 810,
    24, 815, 48, 791, 68, 771, 74, 765, 178, 661, 136, 703,
    86, 753, 78, 761, 43, 796, 39, 800, 20, 819, 21, 818,
    95, 744, 202, 637, 190, 649, 181, 658, 137, 702, 125, 714, 151, 688,
    217, 622, 128, 711, 142, 697, 122, 717, 203, 636, 118, 721, 110, 729, 89, 750, 103, 736, 61,
    778, 55, 784, 15, 824, 14, 825,
    12, 827, 23, 816, 34, 805, 37, 802, 46, 793, 207, 632, 179, 660, 145, 694, 130, 709, 223, 616,
    228, 611, 227, 612, 132, 707, 133, 706, 143, 696, 135, 704, 161, 678, 201, 638, 173, 666, 106,
    733, 83, 756, 91, 748, 66, 773, 53, 786, 10, 829, 9, 830,
    7, 832, 8, 831, 16, 823, 47, 792, 64, 775, 57, 782, 104, 735, 101, 738, 108, 731, 208, 631, 184,
    655, 197, 642, 191, 648, 121, 718, 141, 698, 149, 690, 216, 623, 218, 621,
    152, 687, 144, 695, 134, 705, 138, 701, 199, 640, 162, 677, 176, 663, 119, 720, 158, 681, 164,
    675, 174, 665, 171, 668, 170, 669, 87, 752, 169, 670, 88, 751, 107, 732, 81, 758, 82, 757, 100,
    739, 98, 741, 71, 768, 59, 780, 65, 774, 50, 789, 49, 790, 26, 813, 17, 822, 13, 826, 6, 833,
    5, 834, 33, 806, 51, 788, 75, 764, 99, 740, 96, 743, 97, 742, 166, 673, 172, 667, 175, 664, 187,
    652, 163, 676, 185, 654, 200, 639, 114, 725, 189, 650, 115, 724, 194, 645, 195, 644, 192, 647,
    182, 657, 157, 682, 156, 683, 211, 628, 154, 685, 123, 716, 139, 700, 212, 627, 153, 686, 213,
    626, 215, 624, 150, 689,
    225, 614, 224, 615, 221, 618, 220, 619, 127, 712, 147, 692, 124, 715, 193, 646, 205, 634, 206,
    633, 116, 723, 160, 679, 186, 653, 167, 672, 79, 760, 85, 754, 77, 762, 92, 747, 58, 781, 62,
    777, 69, 770, 54, 785, 36, 803, 32, 807, 25, 814, 18, 821, 11, 828, 4, 835,
    3, 836, 19, 820, 22, 817, 41, 798, 38, 801, 44, 795, 52, 787, 45, 794, 63, 776, 67, 772, 72,
    767, 76, 763, 94, 745, 102, 737, 90, 749, 109, 730, 165, 674, 111, 728, 209, 630, 204, 635, 117,
    722, 188, 651, 159, 680, 198, 641, 113, 726, 183, 656, 180, 659, 177, 662, 196, 643, 155, 684,
    214, 625, 126, 713, 131, 708, 219, 620, 222, 617, 226, 613,
    230, 609, 232, 607, 262, 577, 252, 587, 418, 421, 416, 423, 413, 426, 411, 428, 376, 463, 395,
    444, 283, 556, 285, 554, 379, 460, 390, 449, 363, 476, 384, 455, 388, 451, 386, 453, 361, 478,
    387, 452, 360, 479, 310, 529, 354, 485, 328, 511, 315, 524, 337, 502, 349, 490, 335, 504, 324,
    515,
    323, 516, 320, 519, 334, 505, 359, 480, 295, 544, 385, 454, 292, 547, 291, 548, 381, 458, 399,
    440, 380, 459, 397, 442, 369, 470, 377, 462, 410, 429, 407, 432, 281, 558, 414, 425, 247, 592,
    277, 562, 271, 568, 272, 567, 264, 575, 259, 580,
    237, 602, 239, 600, 244, 595, 243, 596, 275, 564, 278, 561, 250, 589, 246, 593, 417, 422, 248,
    591, 394, 445, 393, 446, 370, 469, 365, 474, 300, 539, 299, 540, 364, 475, 362, 477, 298, 541,
    312, 527, 313, 526, 314, 525, 353, 486, 352, 487, 343, 496, 327, 512, 350, 489, 326, 513, 319,
    520, 332, 507, 333, 506, 348, 491, 347, 492, 322, 517,
    330, 509, 338, 501, 341, 498, 340, 499, 342, 497, 301, 538, 366, 473, 401, 438, 371, 468, 408,
    431, 375, 464, 249, 590, 269, 570, 238, 601, 234, 605,
    257, 582, 273, 566, 255, 584, 254, 585, 245, 594, 251, 588, 412, 427, 372, 467, 282, 557, 403,
    436, 396, 443, 392, 447, 391, 448, 382, 457, 389, 450, 294, 545, 297, 542, 311, 528, 344, 495,
    345, 494, 318, 521, 331, 508, 325, 514, 321, 518,
    346, 493, 339, 500, 351, 488, 306, 533, 289, 550, 400, 439, 378, 461, 374, 465, 415, 424, 270,
    569, 241, 598,
    231, 608, 260, 579, 268, 571, 276, 563, 409, 430, 398, 441, 290, 549, 304, 535, 308, 531, 358,
    481, 316, 523,
    293, 546, 288, 551, 284, 555, 368, 471, 253, 586, 256, 583, 263, 576,
    242, 597, 274, 565, 402, 437, 383, 456, 357, 482, 329, 510,
    317, 522, 307, 532, 286, 553, 287, 552, 266, 573, 261, 578,
    236, 603, 303, 536, 356, 483,
    355, 484, 405, 434, 404, 435, 406, 433,
    235, 604, 267, 572, 302, 537,
    309, 530, 265, 574, 233, 606,
    367, 472, 296, 543,
    336, 503, 305, 534, 373, 466, 280, 559, 279, 560, 419, 420, 240, 599, 258, 581, 229, 610
]
# fmt: on

# 3GPP TS 136.211: Table 5.7.2-5
# fmt: off
U_ZC_TDD = [
    1, 138, 2, 137, 3, 136, 4, 135, 5, 134, 6, 133, 7, 132, 8, 131, 9, 130, 10, 129,
    11, 128, 12, 127, 13, 126, 14, 125, 15, 124, 16, 123, 17, 122, 18, 121, 19, 120, 20, 119,
    21, 118, 22, 117, 23, 116, 24, 115, 25, 114, 26, 113, 27, 112, 28, 111, 29, 110, 30, 109,
    31, 108, 32, 107, 33, 106, 34, 105, 35, 104, 36, 103, 37, 102, 38, 101, 39, 100, 40, 99,
    41, 98, 42, 97, 43, 96, 44, 95, 45, 94, 46, 93, 47, 92, 48, 91, 49, 90, 50, 89,
    51, 88, 52, 87, 53, 86, 54, 85, 55, 84, 56, 83, 57, 82, 58, 81, 59, 80, 60, 79,
    61, 78, 62, 77, 63, 76, 64, 75, 65, 74, 66, 73, 67, 72, 68, 71, 69, 70,
    # 138-837 - N/A
]
# fmt: on

N_RB_SC = 12  # Number of subcarriers in one LTE resource block
F_S = 30_720_000  # LTE sampling frequency, Hz

NUM_SUBFRAMES = 10
# TS 136 211 v10.0.0 — Table 5.7.1-2 data for frame structure type 1
# system frame (even = 0, any = 1), subframes
# fmt: off
SUBFRAME_CONFIG = [
    [0, [1]],
    [0, [4]],
    [0, [7]],
    [1, [1]],
    [1, [4]],
    [1, [7]],
    [1, [1, 6]],
    [1, [2, 7]],
    [1, [3, 8]],
    [1, [1, 4, 7]],
    [1, [2, 5, 8]],
    [1, [3, 6, 9]],
    [1, [0, 2, 4, 6, 8]],
    [1, [1, 3, 5, 7, 9]],
    [1, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]],
    [0, [9]],
    [0, [1]],
    [0, [4]],
    [0, [7]],
    [1, [1]],
    [1, [4]],
    [1, [7]],
    [1, [1, 6]],
    [1, [2, 7]],
    [1, [3, 8]],
    [1, [1, 4, 7]],
    [1, [2, 5, 8]],
    [1, [3, 6, 9]],
    [1, [0, 2, 4, 6, 8]],
    [1, [1, 3, 5, 7, 9]],
    [None, None],
    [0, [9]],
    [0, [1]],
    [0, [4]],
    [0, [7]],
    [1, [1]],
    [1, [4]],
    [1, [7]],
    [1, [1, 6]],
    [1, [2, 7]],
    [1, [3, 8]],
    [1, [1, 4, 7]],
    [1, [2, 5, 8]],
    [1, [3, 6, 9]],
    [1, [0, 2, 4, 6, 8]],
    [1, [1, 3, 5, 7, 9]],
    [None, None],
    [0, [9]],
    [0, [1]],
    [0, [4]],
    [0, [7]],
    [1, [1]],
    [1, [4]],
    [1, [7]],
    [1, [1, 6]],
    [1, [2, 7]],
    [1, [3, 8]],
    [1, [1, 4, 7]],
    [1, [2, 5, 8]],
    [1, [3, 6, 9]],
    [None, None],
    [None, None],
    [None, None],
    [0, [9]],
]

NUM_SF = [1, 2, 2, 3]

# 3GPP TS 136.211: Table 5.7.1-2, preamble format column
# The format follows from the PRACH configuration index, it is not signalled
# on its own. None marks an index that carries no PRACH opportunity: the
# density it asks for cannot hold a preamble of that length without overlap.
PREAMBLE_FORMAT = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, None, 1,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, None, 2,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, None, None, None, 3,
]
# fmt: on

# 3GPP TS 136.211: Table 5.7.1-1, preamble formats 0-3
# Cyclic prefix and sequence durations, in samples of F_S
CP_LENGTH = [3168, 21024, 6240, 21024]
SEQUENCE_LENGTH = [24576, 24576, 49152, 49152]

# 3GPP TS 136.211: 5.7.3
# Uplink subcarrier spacing
DELTA_F = 15_000
# PRACH subcarrier spacing, Hz. Follows from the preamble format:
# 7500 for format 4
DELTA_F_RA = 1250
# Fixed frequency offset within a resource block. 2 for format 4
PHI = 7

# Uplink grid sampled at the PRACH subcarrier spacing
N_FFT = int(F_S / DELTA_F_RA)

SAMPLES_PER_SUBFRAME = int(F_S * 1e-3)

# 3GPP TS 136.213: 4.2.3
# Timing advance is reported as a count of T_A_GRANULARITY samples of F_S
T_A_GRANULARITY = 16
T_A_MAX = 1282


def prach_subcarrier_start(n_ul_rb: int, n_ra_prb_offset: int) -> int:
    """Index of the first PRACH subcarrier in the uplink grid (TS 36.211 5.7.3).

    Both the mapping and the demapping side have to agree on where the 839
    PRACH tones sit, so the placement is derived in one place
    """
    # Frequency scaling factor between the uplink and the PRACH grid
    K = DELTA_F // DELTA_F_RA

    # Relative PRACH offset from the central carrier
    k0 = n_ra_prb_offset * N_RB_SC - (n_ul_rb * N_RB_SC) / 2

    return int(N_FFT // 2 + PHI + K * (k0 + 0.5))


def n_cs_from_config(
    zero_correlation_config: int,
    high_speed_flag: int,
    n_cs_table: Sequence[Sequence[int]] = N_CS_FDD,
) -> int:
    """N_cs from zeroCorrelationZoneConfig and highSpeedFlag (Table 5.7.2-2)."""
    if not 0 <= zero_correlation_config < len(n_cs_table):
        raise ValueError(
            f"zero_correlation_config must be in 0..{len(n_cs_table) - 1}, "
            f"got {zero_correlation_config}"
        )
    if high_speed_flag not in (0, 1):
        raise ValueError(f"high_speed_flag must be 0 or 1, got {high_speed_flag}")

    n_cs = n_cs_table[zero_correlation_config][high_speed_flag]
    if n_cs is None:
        raise ValueError(
            f"zero_correlation_config={zero_correlation_config} is not defined "
            f"for high_speed_flag={high_speed_flag}"
        )
    return int(n_cs)


@lru_cache(maxsize=None)
def root_distance(u_zc: int, n_zc: int = N_ZC_FDD) -> int:
    """d_u, the shift a one subcarrier frequency offset causes (TS 36.211 5.7.2).
    """
    p = pow(u_zc, -1, n_zc)
    return p if p < n_zc / 2 else n_zc - p


def get_shifts(n_zc: int, n_cs: int, u_zc: int = 0) -> List[int]:
    """Cyclic shifts C_v of one root (TS 36.211 5.7.2).

    u_zc = 0 selects the unrestricted set, the actual root value selects
    the restricted set

    Worked out once per (sequence, zone, root) and kept - a cell offers the
    same shifts for as long as it is configured the same way, and a detector
    asks after them on every frame. What is kept is a tuple, so that a caller
    cannot alter what the next one is handed; this returns a list of its own
    either way.
    """
    return list(_shifts(n_zc, n_cs, u_zc))


@lru_cache(maxsize=None)
def _shifts(n_zc: int, n_cs: int, u_zc: int) -> tuple:
    return tuple(
        _get_shifts_unrestricted(n_zc, n_cs)
        if u_zc == 0
        else _get_shifts_restricted(n_zc, n_cs, u_zc)
    )


def _get_shifts_unrestricted(n_zc: int, n_cs: int) -> List[int]:
    if n_cs == 0:
        return [0]

    num_shifts = math.floor(n_zc / n_cs)
    return [v * n_cs for v in range(num_shifts)]


def _get_shifts_restricted(n_zc: int, n_cs: int, u_zc: int) -> List[int]:
    if n_cs == 0:
        return [0]

    d_u = root_distance(u_zc, n_zc)

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
    return [
        d_start * math.floor(v / n_shift) + (v % n_shift) * n_cs
        for v in range(total_shifts)
    ]


@dataclass(frozen=True)
class PreambleSlot:
    """One (root, cyclic shift) pair and the preamble index it carries"""

    preamble_index: int
    root_offset: int  # offset from root_sequence_index, not an absolute root
    u_zc: int
    c_v: int
    d_u: int


def build_preamble_map(
    root_sequence_index: int,
    n_cs: int,
    high_speed_flag: int,
    n_zc: int = N_ZC_FDD,
    u_zc_table: Sequence[int] = U_ZC_FDD,
    total_preambles: int = TOTAL_PREAMBLES,
) -> List[PreambleSlot]:
    """Which (root, cyclic shift) carries each preamble a cell offers.

    The map follows from the configuration alone, and a detector walks it on
    every frame, so the answer for the table this project runs on is kept.
    A caller naming a table of its own is not a case worth keeping - it is
    tests asking what happens at the edges - and is worked out afresh.
    """
    if u_zc_table is U_ZC_FDD:
        return list(_preamble_map(
            root_sequence_index, n_cs, high_speed_flag, n_zc, total_preambles
        ))

    return _build_preamble_map(
        root_sequence_index, n_cs, high_speed_flag, n_zc, u_zc_table,
        total_preambles,
    )


@lru_cache(maxsize=None)
def _preamble_map(
    root_sequence_index: int,
    n_cs: int,
    high_speed_flag: int,
    n_zc: int,
    total_preambles: int,
) -> tuple:
    return tuple(_build_preamble_map(
        root_sequence_index, n_cs, high_speed_flag, n_zc, U_ZC_FDD,
        total_preambles,
    ))


def _build_preamble_map(
    root_sequence_index: int,
    n_cs: int,
    high_speed_flag: int,
    n_zc: int,
    u_zc_table: Sequence[int],
    total_preambles: int,
) -> List[PreambleSlot]:
    if total_preambles <= 0:
        raise ValueError(f"total_preambles must be positive, got {total_preambles}")
    if not 0 <= root_sequence_index < len(u_zc_table):
        raise ValueError(
            f"root_sequence_index must be in 0..{len(u_zc_table) - 1}, "
            f"got {root_sequence_index}"
        )

    slots: List[PreambleSlot] = []
    for root_offset in range(len(u_zc_table)):
        if len(slots) >= total_preambles:
            break

        u_zc = u_zc_table[(root_sequence_index + root_offset) % len(u_zc_table)]

        for c_v in get_shifts(n_zc, n_cs, u_zc * high_speed_flag):
            if len(slots) >= total_preambles:
                break
            slots.append(
                PreambleSlot(
                    preamble_index=len(slots),
                    root_offset=root_offset,
                    u_zc=u_zc,
                    c_v=c_v % n_zc,
                    d_u=root_distance(u_zc, n_zc),
                )
            )

    if len(slots) < total_preambles:
        # every root was walked and the cell still cannot offer that many
        raise ValueError(
            f"configuration yields only {len(slots)} preambles out of "
            f"{total_preambles} (n_cs={n_cs}, high_speed_flag={high_speed_flag})"
        )

    return slots
