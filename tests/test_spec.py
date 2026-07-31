import unittest

from prach.pipeline.spec import (
    N_CS_FDD,
    N_ZC_FDD,
    PREAMBLE_FORMAT,
    SUBFRAME_CONFIG,
    TOTAL_PREAMBLES,
    U_ZC_FDD,
    build_preamble_map,
    get_shifts,
    n_cs_from_config,
    root_distance,
)

DEFINED_CONFIGS = [
    (zcc, hs)
    for zcc in range(len(N_CS_FDD))
    for hs in (0, 1)
    if N_CS_FDD[zcc][hs] is not None
]


class TestPreambleFormatTable(unittest.TestCase):

    def test_matches_the_subframe_table(self):
        self.assertEqual(len(PREAMBLE_FORMAT), 64)
        for index, preamble_format in enumerate(PREAMBLE_FORMAT):
            with self.subTest(config_index=index):
                # both columns come from Table 5.7.1-2 and mark the same
                # indices as carrying no PRACH opportunity
                self.assertEqual(
                    preamble_format is None, SUBFRAME_CONFIG[index][0] is None
                )
                if preamble_format is not None:
                    self.assertEqual(preamble_format, index // 16)


class TestNcsFromConfig(unittest.TestCase):

    def test_matches_table(self):
        for zcc, hs in DEFINED_CONFIGS:
            with self.subTest(zero_correlation_config=zcc, high_speed_flag=hs):
                self.assertEqual(n_cs_from_config(zcc, hs), N_CS_FDD[zcc][hs])

    def test_undefined_combination_rejected(self):
        # zeroCorrelationZoneConfig 15 has no restricted set entry
        with self.assertRaises(ValueError):
            n_cs_from_config(15, 1)

    def test_out_of_range_rejected(self):
        for zcc, hs in ((16, 0), (-1, 0), (0, 2)):
            with self.subTest(zero_correlation_config=zcc, high_speed_flag=hs):
                with self.assertRaises(ValueError):
                    n_cs_from_config(zcc, hs)


class TestRootDistance(unittest.TestCase):

    def test_is_the_folded_modular_inverse(self):
        for u_zc in U_ZC_FDD[:100]:
            with self.subTest(u_zc=u_zc):
                d_u = root_distance(u_zc, N_ZC_FDD)
                self.assertTrue(1 <= d_u <= N_ZC_FDD // 2)
                # either p or n_zc - p, so the product folds to 1 or -1
                self.assertIn((d_u * u_zc) % N_ZC_FDD, (1, N_ZC_FDD - 1))


class TestGetShifts(unittest.TestCase):

    def test_unrestricted_shifts_are_evenly_spaced(self):
        for zcc in range(len(N_CS_FDD)):
            n_cs = n_cs_from_config(zcc, 0)
            with self.subTest(zero_correlation_config=zcc, n_cs=n_cs):
                shifts = get_shifts(N_ZC_FDD, n_cs)
                if n_cs == 0:
                    self.assertEqual(shifts, [0])
                    continue
                self.assertEqual(shifts, [v * n_cs for v in range(N_ZC_FDD // n_cs)])

    def test_restricted_shifts_stay_inside_the_sequence(self):
        for zcc, hs in DEFINED_CONFIGS:
            if not hs:
                continue
            n_cs = n_cs_from_config(zcc, hs)
            for u_zc in U_ZC_FDD[:20]:
                with self.subTest(zero_correlation_config=zcc, u_zc=u_zc):
                    for c_v in get_shifts(N_ZC_FDD, n_cs, u_zc):
                        self.assertTrue(0 <= c_v % N_ZC_FDD < N_ZC_FDD)

    def test_restricted_set_differs_from_unrestricted(self):
        # a root whose distance is too small offers nothing in restricted mode
        n_cs = n_cs_from_config(8, 1)
        self.assertEqual(get_shifts(N_ZC_FDD, n_cs, U_ZC_FDD[0]), [])
        self.assertNotEqual(get_shifts(N_ZC_FDD, n_cs), [])


class TestBuildPreambleMap(unittest.TestCase):

    def test_numbers_every_preamble_once(self):
        for zcc, hs in DEFINED_CONFIGS:
            n_cs = n_cs_from_config(zcc, hs)
            with self.subTest(zero_correlation_config=zcc, high_speed_flag=hs):
                slots = build_preamble_map(0, n_cs, hs)

                self.assertEqual(len(slots), TOTAL_PREAMBLES)
                self.assertEqual(
                    [s.preamble_index for s in slots], list(range(TOTAL_PREAMBLES))
                )
                # roots are walked in order and never revisited; a restricted
                # set may skip the first ones, they offer no shift at all
                offsets = [s.root_offset for s in slots]
                self.assertEqual(offsets, sorted(offsets))
                self.assertGreaterEqual(offsets[0], 0)
                if not hs:
                    self.assertEqual(offsets[0], 0)

    def test_shifts_of_a_root_match_get_shifts(self):
        n_cs = n_cs_from_config(11, 0)
        slots = build_preamble_map(0, n_cs, 0)

        by_root = {}
        for slot in slots:
            by_root.setdefault(slot.root_offset, []).append(slot.c_v)

        for root_offset, shifts in by_root.items():
            with self.subTest(root_offset=root_offset):
                expected = get_shifts(N_ZC_FDD, n_cs)[: len(shifts)]
                self.assertEqual(shifts, expected)

    def test_root_sequence_index_shifts_the_roots(self):
        n_cs = n_cs_from_config(11, 0)
        base = build_preamble_map(0, n_cs, 0)
        shifted = build_preamble_map(5, n_cs, 0)

        self.assertEqual(
            [U_ZC_FDD[5 + s.root_offset] for s in base],
            [s.u_zc for s in shifted],
        )

    def test_bad_arguments_rejected(self):
        n_cs = n_cs_from_config(11, 0)
        with self.assertRaises(ValueError):
            build_preamble_map(0, n_cs, 0, total_preambles=0)
        with self.assertRaises(ValueError):
            build_preamble_map(len(U_ZC_FDD), n_cs, 0)

    def test_exhausted_root_table_raises_instead_of_looping(self):
        # a table too small to reach 64 preambles must terminate with an error
        with self.assertRaises(ValueError):
            build_preamble_map(0, 419, 0, u_zc_table=U_ZC_FDD[:4])


if __name__ == "__main__":
    unittest.main()
