import unittest

import numpy as np

from prach.blocks.enb import DetectorBlock
from prach.math import zadoff_chu
from prach.pipeline import PRACHConfiguration
from prach.pipeline.spec import (
    DELTA_F_RA,
    F_S,
    N_ZC_FDD,
    T_A_GRANULARITY,
    U_ZC_FDD,
    build_preamble_map,
    n_cs_from_config,
)

# The receiver zero pads the 839 correlated bins before transforming back
N_IFFT = 2048

# power of one correlation peak, used to size the noise of a test
PEAK_POWER = (N_ZC_FDD**2 / N_IFFT) ** 2

# an index of every preamble format, all of them "any system frame, subframe 1"
FORMAT_CONFIG_INDEX = [3, 19, 35, 51]


def expected_timing_advance(delay_bins):
    delay_seconds = delay_bins / (N_IFFT * DELTA_F_RA)
    return round(delay_seconds * F_S / T_A_GRANULARITY)


def make_config(zero_correlation_config, high_speed_flag=0, root_sequence_index=0,
                config_index=FORMAT_CONFIG_INDEX[0]):
    return PRACHConfiguration(
        config_index=config_index,
        zero_correlation_config=zero_correlation_config,
        high_speed_flag=high_speed_flag,
        root_sequence_index=root_sequence_index,
    )


class PrachFixture:
    """Builds the power delay profiles the detector expects to be fed.

    The transforms here go through numpy on purpose: prach.math holds explicit
    O(N^2) loops that are part of what the project demonstrates, and a fixture
    has no business paying for them.
    """

    _preambles = {}
    _roots = {}

    @classmethod
    def preambles(cls, zero_correlation_config, high_speed_flag):
        key = (zero_correlation_config, high_speed_flag)
        if key not in cls._preambles:
            n_cs = n_cs_from_config(zero_correlation_config, high_speed_flag)
            slots = build_preamble_map(0, n_cs, high_speed_flag)
            sequences = []
            for slot in slots:
                base = zadoff_chu(slot.u_zc, N_ZC_FDD)
                sequences.append(base[slot.c_v:] + base[: slot.c_v])
            cls._preambles[key] = sequences
        return cls._preambles[key]

    @classmethod
    def root_spectrum(cls, root_offset):
        if root_offset not in cls._roots:
            u_zc = U_ZC_FDD[root_offset % len(U_ZC_FDD)]
            cls._roots[root_offset] = np.fft.fft(
                np.array(zadoff_chu(u_zc, N_ZC_FDD))
            )
        return cls._roots[root_offset]

    @classmethod
    def pdps(
        cls,
        transmitted,
        config,
        delay_bins=0,
        noise_amplitude=0.0,
        seed=0,
        gains=None,
        frequency_offset=0,
        root_count=None,
    ):
        preambles = cls.preambles(
            config.zero_correlation_config, config.high_speed_flag
        )
        gains = gains or [1.0] * len(transmitted)

        received = np.zeros(N_ZC_FDD, dtype=complex)
        for index, gain in zip(transmitted, gains):
            received += gain * np.fft.fft(np.asarray(preambles[index]))

        if frequency_offset:
            received = np.roll(received, frequency_offset)
        if delay_bins:
            ramp = -2j * np.pi * np.arange(N_ZC_FDD) * delay_bins / N_IFFT
            received = received * np.exp(ramp)

        if root_count is None:
            root_count = DetectorBlock(config).root_count

        rng = np.random.default_rng(seed)
        profiles = []
        for root_offset in range(root_count):
            padded = np.zeros(N_IFFT, dtype=complex)
            padded[:N_ZC_FDD] = received * np.conj(cls.root_spectrum(root_offset))
            profile = np.fft.ifft(padded)
            if noise_amplitude:
                noise = rng.standard_normal(N_IFFT) + 1j * rng.standard_normal(N_IFFT)
                profile = profile + noise_amplitude * noise / np.sqrt(2)
            profiles.append(np.abs(profile) ** 2)
        return profiles

    @classmethod
    def noise_only_pdps(cls, root_count, rng):
        """Noise that went through the same zero padding as a real profile."""
        profiles = []
        for _ in range(root_count):
            spectrum = rng.standard_normal(N_ZC_FDD) + 1j * rng.standard_normal(
                N_ZC_FDD
            )
            padded = np.zeros(N_IFFT, dtype=complex)
            padded[:N_ZC_FDD] = spectrum / np.sqrt(2)
            profiles.append(np.abs(np.fft.ifft(padded)) ** 2)
        return profiles


class TestConstruction(unittest.TestCase):

    def test_tuning_is_optional(self):
        DetectorBlock(make_config(11))

    def test_invalid_pfa_rejected(self):
        for pfa in (0.0, 1.0, -1.0):
            with self.subTest(pfa=pfa):
                with self.assertRaises(ValueError):
                    DetectorBlock(make_config(11), pfa=pfa)

    def test_root_count_matches_the_preamble_map(self):
        for zcc in (0, 1, 11, 15):
            with self.subTest(zero_correlation_config=zcc):
                config = make_config(zcc)
                n_cs = n_cs_from_config(zcc, 0)
                slots = build_preamble_map(0, n_cs, 0)
                self.assertEqual(
                    DetectorBlock(config).root_count, slots[-1].root_offset + 1
                )


class TestInputValidation(unittest.TestCase):

    def test_absent_and_empty_input(self):
        block = DetectorBlock(make_config(11))
        for pdps in (None, [], [np.zeros(0)]):
            with self.subTest(pdps=pdps):
                self.assertEqual(block.detect(pdps), [])

    def test_short_profile_rejected(self):
        with self.assertRaises(ValueError):
            DetectorBlock(make_config(11)).detect([np.zeros(100)])

    def test_ragged_profiles_rejected(self):
        with self.assertRaises(ValueError):
            DetectorBlock(make_config(11)).detect(
                [np.zeros(N_IFFT), np.zeros(N_IFFT + 1)]
            )

    def test_two_dimensional_profile_rejected(self):
        with self.assertRaises(ValueError):
            DetectorBlock(make_config(11)).detect([np.zeros((2, N_IFFT))])

    def test_too_few_roots_rejected(self):
        block = DetectorBlock(make_config(11))
        self.assertGreater(block.root_count, 1)
        with self.assertRaises(ValueError):
            block.detect([np.zeros(N_IFFT)])

    def test_undefined_zero_correlation_config_rejected(self):
        with self.assertRaises(ValueError):
            DetectorBlock(make_config(15, high_speed_flag=1)).detect(
                [np.zeros(N_IFFT)]
            )

    def test_configuration_without_prach_rejected(self):
        # Table 5.7.1-2 marks configuration index 30 N/A
        config = make_config(11, config_index=30)
        with self.assertRaises(ValueError):
            DetectorBlock(config).detect([np.zeros(N_IFFT)])

    def test_complex_profile_is_squared(self):
        config = make_config(11)
        power = PrachFixture.pdps([4], config)
        amplitude = [np.sqrt(p).astype(complex) for p in power]

        block = DetectorBlock(config)
        self.assertEqual(
            [d.preamble_index for d in block.detect(power)],
            [d.preamble_index for d in block.detect(amplitude)],
        )


class TestRoundTrip(unittest.TestCase):
    """Every preamble the cell offers must come back under its own index."""

    def test_unrestricted_sets_noiseless(self):
        for zcc in (0, 1, 4, 11, 15):
            config = make_config(zcc)
            block = DetectorBlock(config)
            for transmitted in range(64):
                with self.subTest(zero_correlation_config=zcc, preamble=transmitted):
                    found = block.detect(PrachFixture.pdps([transmitted], config))
                    self.assertEqual(
                        [d.preamble_index for d in found], [transmitted]
                    )
                    self.assertEqual([d.timing_advance for d in found], [0])

    def test_restricted_sets_noiseless(self):
        for zcc in (2, 4, 9):
            config = make_config(zcc, high_speed_flag=1)
            block = DetectorBlock(config)
            for transmitted in range(0, 64, 7):
                with self.subTest(zero_correlation_config=zcc, preamble=transmitted):
                    found = block.detect(PrachFixture.pdps([transmitted], config))
                    self.assertEqual(
                        [d.preamble_index for d in found], [transmitted]
                    )

    def test_with_noise(self):
        noise = np.sqrt(PEAK_POWER) / 100.0
        for zcc in (1, 4, 11):
            config = make_config(zcc)
            block = DetectorBlock(config)
            misses = extras = 0
            for transmitted in range(64):
                pdps = PrachFixture.pdps(
                    [transmitted], config, noise_amplitude=noise, seed=transmitted
                )
                found = [d.preamble_index for d in block.detect(pdps)]
                misses += transmitted not in found
                extras += len([i for i in found if i != transmitted])
            with self.subTest(zero_correlation_config=zcc):
                self.assertEqual(misses, 0)
                self.assertLessEqual(extras, 3, f"extras={extras}")


class TestTimingAdvance(unittest.TestCase):

    def test_matches_injected_delay(self):
        config = make_config(11)
        block = DetectorBlock(config)
        noise = np.sqrt(PEAK_POWER) / 300.0

        for delay_bins in (0, 5, 20, 60, 120, 200):
            with self.subTest(delay_bins=delay_bins):
                found = block.detect(
                    PrachFixture.pdps(
                        [3],
                        config,
                        delay_bins=delay_bins,
                        noise_amplitude=noise,
                        seed=5,
                    )
                )
                self.assertEqual([d.preamble_index for d in found], [3])
                self.assertAlmostEqual(
                    found[0].timing_advance,
                    expected_timing_advance(delay_bins),
                    delta=1,
                )

    def test_search_stops_at_the_declared_reach(self):
        config = make_config(11)
        pdps = PrachFixture.pdps([3], config, delay_bins=120)

        # the whole zone is searched by default and the peak is found where it
        # was put
        found = DetectorBlock(config).detect(pdps)
        self.assertEqual([(d.preamble_index, d.timing_advance) for d in found],
                         [(3, expected_timing_advance(120))])

        # once the cell is declared smaller than that delay, no report may
        # claim an advance the cell cannot produce
        for reach in (40, 10):
            with self.subTest(max_timing_advance=reach):
                found = DetectorBlock(
                    config, max_timing_advance=reach
                ).detect(pdps)
                self.assertTrue(all(d.timing_advance <= reach for d in found))

    def test_reach_also_limits_a_configuration_without_cyclic_shifts(self):
        # n_cs 0 gives one preamble per root and the whole profile as its
        # window; a declared reach has to cut that down too
        config = make_config(0)
        pdps = PrachFixture.pdps([3], config, delay_bins=400)

        self.assertEqual(
            [d.preamble_index for d in DetectorBlock(config).detect(pdps)], [3]
        )
        near = DetectorBlock(config, max_timing_advance=40)
        self.assertTrue(all(d.timing_advance <= 40 for d in near.detect(pdps)))

    def test_reach_defaults_to_what_the_zone_can_tell_apart(self):
        # a zone of n_cs samples cannot represent a delay longer than itself
        for zcc, expected in ((11, 170), (1, 23), (15, 767)):
            with self.subTest(zero_correlation_config=zcc):
                block = DetectorBlock(make_config(zcc))
                n_cs = n_cs_from_config(zcc, 0)
                self.assertEqual(block.zone_timing_advance(n_cs), expected)

    def test_searchable_reach_is_what_a_report_can_carry(self):
        """What the block advertises has to be what it looks at.

        The zone is one number, the searched part of a window is another: the
        guard band at its tail goes before the search starts. The two used to
        differ silently by three or four units, so a cell dimensioned on the
        zone gave up its outer couple of hundred metres without saying so.
        """
        for zcc in (1, 4, 11, 15):
            with self.subTest(zero_correlation_config=zcc):
                config = make_config(zcc)
                block = DetectorBlock(config)
                n_cs = n_cs_from_config(zcc, 0)

                reach = block.searchable_timing_advance(n_cs, N_IFFT)
                self.assertLessEqual(reach, block.zone_timing_advance(n_cs))

                # walk the far end of the window and see what still comes back
                span = block._search_span(n_cs, N_IFFT)
                furthest = -1
                for delay_bins in range(max(span - 4, 0), span + 4):
                    found = block.detect(
                        PrachFixture.pdps([3], config, delay_bins=delay_bins)
                    )
                    for detection in found:
                        if detection.preamble_index == 3:
                            furthest = max(furthest, detection.timing_advance)
                self.assertEqual(furthest, reach)

    def test_delay_spread_gives_up_the_tail_of_every_window(self):
        """An echo past the zone is the next preamble, so it is not searched."""
        config = make_config(11)
        n_cs = n_cs_from_config(11, 0)
        plain = DetectorBlock(config)
        dispersive = DetectorBlock(config, delay_spread=5e-6)

        # the zone follows from the configuration and does not move
        self.assertEqual(
            plain.zone_timing_advance(n_cs),
            dispersive.zone_timing_advance(n_cs),
        )
        # what is searched does
        self.assertLess(
            dispersive.searchable_timing_advance(n_cs, N_IFFT),
            plain.searchable_timing_advance(n_cs, N_IFFT),
        )

        # a preamble sitting past the shortened reach is still seen - the
        # skirt of its peak reaches the last tap searched - but it may never
        # be reported as having come from further than that tap
        beyond = dispersive._search_span(n_cs, N_IFFT) + 3
        found = dispersive.detect(
            PrachFixture.pdps([3], config, delay_bins=beyond)
        )
        reach = dispersive.searchable_timing_advance(n_cs, N_IFFT)
        self.assertTrue(all(d.timing_advance <= reach for d in found))
        self.assertLess(reach, plain.searchable_timing_advance(n_cs, N_IFFT))

    def test_negative_delay_spread_rejected(self):
        with self.assertRaises(ValueError):
            DetectorBlock(make_config(11), delay_spread=-1e-6)


class TestFalseAlarmRate(unittest.TestCase):
    TRIALS = 400
    TARGET_PFA = 0.05

    def test_stays_near_the_target(self):
        """The rate is per call, the way TS 36.141 states the requirement.

        A loose target is used here so that the run stays short: at the 1e-3
        the specification asks for, seeing a single alarm would take thousands
        of trials.
        """
        for zcc in (1, 11):
            config = make_config(zcc)
            block = DetectorBlock(config, pfa=self.TARGET_PFA)
            rng = np.random.default_rng(7)

            alarms = 0
            for _ in range(self.TRIALS):
                pdps = PrachFixture.noise_only_pdps(block.root_count, rng)
                alarms += bool(block.detect(pdps))

            rate = alarms / self.TRIALS
            with self.subTest(zero_correlation_config=zcc):
                self.assertLessEqual(rate, 3 * self.TARGET_PFA, f"measured {rate}")

    def test_rate_is_shared_over_the_windows_searched(self):
        # a configuration that packs more preambles per root searches more
        # windows for the same target, so each window must clear a higher bar
        dense = DetectorBlock(make_config(1))  # 64 preambles on one root
        wide = DetectorBlock(make_config(15))  # 2 preambles per root

        self.assertGreater(
            dense._threshold_from_pfa(32, 1e-3 / 64),
            dense._threshold_from_pfa(32, 1e-3),
        )
        self.assertEqual(wide.root_count, 32)

    def test_threshold_factor_overrides_pfa(self):
        config = make_config(11)
        pdps = PrachFixture.pdps([9], config)

        self.assertEqual(
            [d.preamble_index for d in DetectorBlock(config).detect(pdps)], [9]
        )
        # a threshold no peak can clear
        blind = DetectorBlock(config, threshold_factor=1e12)
        self.assertEqual(blind.detect(pdps), [])


class TestMultipleUsers(unittest.TestCase):

    def test_simultaneous_preambles(self):
        config = make_config(11)
        block = DetectorBlock(config)

        for group in ([0, 1], [3, 17, 40], [5, 6, 7, 8], list(range(0, 64, 8))):
            with self.subTest(group=group):
                found = block.detect(PrachFixture.pdps(group, config))
                self.assertEqual(sorted(d.preamble_index for d in found), sorted(group))

    def test_weak_user_beside_a_strong_one(self):
        # the cross correlation floor of a foreign root sits about 10*log10(839)
        # below the strong peak, so a 10 dB imbalance must still resolve
        config = make_config(11)
        pdps = PrachFixture.pdps(
            [0, 30],
            config,
            gains=[1.0, 10 ** (-10 / 20)],
            noise_amplitude=np.sqrt(PEAK_POWER) / 300.0,
            seed=2,
        )
        found = DetectorBlock(config).detect(pdps)
        self.assertEqual(sorted(d.preamble_index for d in found), [0, 30])


class TestDoppler(unittest.TestCase):

    def test_frequency_offset_keeps_the_preamble_index(self):
        config = make_config(4, high_speed_flag=1)
        block = DetectorBlock(config)

        for transmitted in (0, 7, 21):
            for offset in (1, -1):
                with self.subTest(preamble=transmitted, frequency_offset=offset):
                    found = block.detect(
                        PrachFixture.pdps(
                            [transmitted], config, frequency_offset=offset
                        )
                    )
                    self.assertEqual(
                        [d.preamble_index for d in found], [transmitted]
                    )
                    self.assertNotEqual(found[0].doppler_offset, 0)

    def test_reported_offset_carries_the_sign_of_the_error(self):
        """The tag must not flip on roots whose inverse got folded.

        root_distance reports min(p, n_zc - p), so a replica of a positive
        error sits below the nominal shift for one root and above it for
        another. Preambles 0 and 21 sit on either side of that fold.
        """
        config = make_config(4, high_speed_flag=1)
        block = DetectorBlock(config)

        for transmitted in (0, 7, 21, 30, 50):
            for offset in (-1, 0, 1):
                with self.subTest(preamble=transmitted, frequency_offset=offset):
                    found = block.detect(
                        PrachFixture.pdps(
                            [transmitted], config, frequency_offset=offset
                        )
                    )
                    reported = [
                        d.doppler_offset
                        for d in found
                        if d.preamble_index == transmitted
                    ]
                    self.assertEqual(reported, [offset])

    def test_replica_windows_report_the_delay_they_were_found_at(self):
        """A peak found in a replica still carries the timing of the access.

        The replica window sits where the frequency error moved the preamble,
        so a delay is measured from there. The report has to come back with
        the advance that was injected and not with the shift the error caused,
        which is a whole d_u away.
        """
        config = make_config(4, high_speed_flag=1)
        block = DetectorBlock(config)

        for transmitted in (0, 7, 21):
            for offset in (-1, 0, 1):
                for delay_bins in (0, 10, 30):
                    with self.subTest(preamble=transmitted,
                                      frequency_offset=offset,
                                      delay_bins=delay_bins):
                        found = block.detect(
                            PrachFixture.pdps(
                                [transmitted],
                                config,
                                delay_bins=delay_bins,
                                frequency_offset=offset,
                            )
                        )
                        reported = [
                            d for d in found
                            if d.preamble_index == transmitted
                        ]
                        self.assertEqual(len(reported), 1)
                        self.assertEqual(
                            reported[0].timing_advance,
                            expected_timing_advance(delay_bins),
                        )

    def test_unrestricted_set_reports_no_replica(self):
        config = make_config(11)
        found = DetectorBlock(config).detect(PrachFixture.pdps([5], config))
        self.assertEqual([d.doppler_offset for d in found], [0])

    def test_replica_search_can_be_switched_off(self):
        config = make_config(4, high_speed_flag=1)
        pdps = PrachFixture.pdps([0], config, frequency_offset=-1)

        enabled = DetectorBlock(config).detect(pdps)
        self.assertEqual([d.preamble_index for d in enabled], [0])
        self.assertTrue(any(d.doppler_offset for d in enabled))

        disabled = DetectorBlock(config, detect_doppler=False).detect(pdps)
        self.assertTrue(all(d.doppler_offset == 0 for d in disabled))
        # without the replica windows the offset preamble is mistaken for a
        # neighbour of the one that was actually sent
        self.assertNotEqual([d.preamble_index for d in disabled], [0])


class TestSurveyedConfigurations(unittest.TestCase):
    """Five configurations that span what changes the detector's behaviour.

    Window width sets the threshold, the roots decide how many profiles the
    receiver builds, and the preambles packed on one root decide how many
    windows share a polluted profile when a channel spreads a preamble. The
    five below sit at the corners of that space; the shape of the check
    follows TS 36.141 section 8.4, which asks for a detection probability and
    a false alarm probability, though the operating points here are ours.
    """

    # (label, zero_correlation_config, high_speed_flag, roots, per root)
    CONFIGURATIONS = [
        ("dense", 1, 0, 1, 64),
        ("macro", 11, 0, 8, 9),
        ("wide", 15, 0, 32, 2),
        ("high speed", 4, 1, 61, 8),
        ("high speed, large zone", 11, 1, 248, 2),
    ]

    def test_shape_of_each_configuration(self):
        for label, zcc, hs, roots, per_root in self.CONFIGURATIONS:
            with self.subTest(configuration=label):
                block = DetectorBlock(make_config(zcc, hs))
                slots = build_preamble_map(0, n_cs_from_config(zcc, hs), hs)

                self.assertEqual(block.root_count, roots)
                busiest = max(
                    sum(1 for s in slots if s.root_offset == r) for r in range(roots)
                )
                self.assertEqual(busiest, per_root)

    def test_every_preamble_is_detected(self):
        # a handful per configuration, the exhaustive sweep is done elsewhere
        for label, zcc, hs, _roots, _per_root in self.CONFIGURATIONS:
            config = make_config(zcc, hs)
            block = DetectorBlock(config)
            noise = np.sqrt(PEAK_POWER) / 100.0

            for transmitted in (0, 17, 63):
                with self.subTest(configuration=label, preamble=transmitted):
                    pdps = PrachFixture.pdps(
                        [transmitted],
                        config,
                        noise_amplitude=noise,
                        seed=transmitted,
                    )
                    found = [d.preamble_index for d in block.detect(pdps)]
                    self.assertIn(transmitted, found)

    def test_reach_and_zone_agree(self):
        expected = {1: 23, 11: 170, 15: 767, 4: 58}
        for label, zcc, hs, _roots, _per_root in self.CONFIGURATIONS:
            if hs or zcc not in expected:
                continue
            with self.subTest(configuration=label):
                block = DetectorBlock(make_config(zcc, hs))
                self.assertEqual(
                    block.zone_timing_advance(n_cs_from_config(zcc, hs)),
                    expected[zcc],
                )


class TestReportedFields(unittest.TestCase):

    def test_detection_carries_its_evidence(self):
        config = make_config(11)
        found = DetectorBlock(config).detect(
            PrachFixture.pdps(
                [12], config, noise_amplitude=np.sqrt(PEAK_POWER) / 300.0, seed=1
            )
        )
        detection = found[0]

        self.assertEqual(detection.preamble_index, 12)
        self.assertEqual(detection.root_offset, 1)
        self.assertGreater(detection.peak_to_noise, 1.0)
        self.assertGreater(detection.noise_floor, 0.0)
        self.assertEqual(
            detection.peak_to_noise, detection.peak_value / detection.noise_floor
        )
        self.assertIn("timing_advance", detection.as_dict())


if __name__ == "__main__":
    unittest.main()
