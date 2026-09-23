import math
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from prach.pipeline.block import Block
from prach.pipeline.config import Deployment, PRACHConfiguration
from prach.pipeline.spec import (
    F_S,
    N_ZC_FDD,
    T_A_GRANULARITY,
    T_A_MAX,
    TOTAL_PREAMBLES,
    DELTA_F_RA,
    PreambleSlot,
    build_preamble_map,
    n_cs_from_config,
)


def _tail(x: float, branches: int) -> float:
    """P(X > x) for X the sum of `branches` exponential taps of unit mean.

    Chi square with 2*branches degrees of freedom; for whole branches its tail
    is a finite series rather than a special function.
    """
    total = 0.0
    term = 1.0
    for step in range(branches):
        if step:
            term *= x / step
        total += term
    return math.exp(-x) * total


@lru_cache(maxsize=None)
def _tail_inverse(probability: float, branches: int) -> float:
    """Where that tail falls to `probability`.

    The series has no closed form inverse, so the answer is bisected. Cached:
    the answers are few, one per branch count and one per configuration.
    """
    low, high = 0.0, float(branches)
    while _tail(high, branches) > probability:
        high *= 2.0

    # a double has 52 bits of mantissa: nothing moves after ~60 halvings
    for _ in range(60):
        middle = 0.5 * (low + high)
        if _tail(middle, branches) > probability:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


@dataclass
class Detection:
    """One preamble reported by the detector."""

    preamble_index: int
    root_offset: int
    cyclic_shift: int
    peak_bin: int
    peak_value: float
    noise_floor: float
    peak_to_noise: float
    delay_bins: int
    delay_seconds: float
    timing_advance: int
    # frequency error the peak implies, in DELTA_F_RA subcarriers: 0 in the
    # nominal window, -1 or +1 in a replica. Replicas: restricted set only.
    doppler_offset: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class _Window:
    """Delay region searched for one preamble."""

    slot: PreambleSlot
    start: int
    span: int
    length: int  # size of the profile the window indexes into
    doppler_offset: int


def _taps(windows: Sequence["_Window"], length: int) -> np.ndarray:
    """Taps every window of one root covers, one row per window."""
    starts = np.array([window.start for window in windows])[:, None]
    offsets = np.arange(windows[0].span)[None, :]
    return (starts + offsets) % length


@dataclass
class _Candidate:
    """Strongest tap found in one window, before it is accepted or rejected."""

    window: _Window
    local_peak: int
    peak_bin: int
    peak_value: float
    noise: float


class DetectorBlock(Block):
    def __init__(
        self,
        config: PRACHConfiguration,
        *,
        deployment: Optional[Deployment] = None,
        pfa: float = 1e-3,
        interference_margin: float = 1.0,
        threshold_factor: float = 0.0,
        guard_bins: int = -1,
        max_timing_advance: Optional[int] = None,
        delay_spread: Optional[float] = None,
        sidelobe_margin: float = 2.0,
        detect_doppler: bool = True,
    ):
        """
        pfa
            Target probability that one call reports a preamble that was not
            sent. Shared out over every window the call searches, so a
            configuration packing more preambles per root gets a higher
            threshold on each rather than a worse rate overall. TS 36.141
            states the requirement per call; see the detector document for
            that reading.
        interference_margin
            Decibels added to that threshold, covering what the model behind
            `pfa` leaves out: it takes the noise floor as known while it is
            estimated, and the bins of a window as independent while zero
            padding correlates them. Both bounds are measured: below 1 dB the
            false alarm rate does not meet 8.4.1.5, above 2.5 dB detections
            start being lost in ETU70. AWGN is untouched throughout.
        threshold_factor
            Fixed threshold over the noise floor, overriding pfa when positive.
        guard_bins
            Bins dropped at the tail of every window, where the main lobe of
            the next preamble reaches in. Negative derives it from the
            oversampling.
        deployment
            Where this base station stands. None reaches as far as the cyclic
            shift zone allows and assumes no echo. The two arguments below
            override what it declares.
        max_timing_advance
            How far the cell reaches, in timing advance units; the search
            stops there. None takes what `deployment` declares, and failing
            that the largest advance the cyclic shift zone can tell apart,
            which is all the detector can infer on its own. What is actually
            searched is smaller - `searchable_timing_advance` is the number to
            dimension a cell against.
        delay_spread
            Longest echo the cell has to survive, in seconds, given up at the
            tail of every window: a path that late behind a preamble at the
            far end of its zone lands in the next cyclic shift's window and
            reads as a different preamble. None takes what `deployment`
            declares. Zero, the default, assumes a non dispersive channel; a
            cell facing a profile like ETU has to say so here and pick an
            `n_cs` with room for it.
        sidelobe_margin
            Factor above the computed side lobe leakage a candidate must clear
            before it is believed. Zero disables the check, safe only when the
            profile was built with a tapering window.
        detect_doppler
            Also search the +-d_u replicas a frequency offset produces. Only
            meaningful for a restricted set.
        """
        super().__init__(config)

        if not 0.0 < pfa < 1.0:
            raise ValueError(f"pfa must be in (0, 1), got {pfa}")

        # where this base station stands is a fact of its deployment, not of
        # the cell it signals; an explicit argument overrides what was declared
        deployment = Deployment() if deployment is None else deployment
        if max_timing_advance is None:
            declared = deployment.max_timing_advance
            max_timing_advance = None if declared < 0 else int(declared)
        if delay_spread is None:
            delay_spread = deployment.delay_spread

        if max_timing_advance is not None and max_timing_advance <= 0:
            raise ValueError(
                f"max_timing_advance must be positive, got {max_timing_advance}"
            )

        if delay_spread < 0.0:
            raise ValueError(
                f"delay_spread must not be negative, got {delay_spread}"
            )

        if interference_margin < 0.0:
            raise ValueError(
                f"interference_margin must not be negative, got "
                f"{interference_margin}"
            )

        # the windows a profile is searched in follow from the configuration
        # and the profile length alone, so they are built once and kept until
        # one of those changes
        self._layout = None

        self.pfa = pfa
        self.interference_margin = interference_margin
        self.threshold_factor = threshold_factor
        self.guard_bins = guard_bins
        self.max_timing_advance = max_timing_advance
        self.delay_spread = delay_spread
        self.sidelobe_margin = sidelobe_margin
        self.detect_doppler = detect_doppler

    def _preamble_map(self) -> List[PreambleSlot]:
        config = self.config

        # raises when the configuration index carries no PRACH opportunity
        config.preamble_format

        n_cs = n_cs_from_config(
            config.zero_correlation_config, config.high_speed_flag
        )
        return build_preamble_map(
            config.root_sequence_index,
            n_cs,
            config.high_speed_flag,
            total_preambles=TOTAL_PREAMBLES,
        )

    @property
    def root_count(self) -> int:
        """Roots the cell spreads its preambles over.

        The receiver correlates against this many, from root_sequence_index
        on, and hands the detector one profile per root.
        """
        return self._preamble_map()[-1].root_offset + 1

    def detect(self, pdps: Sequence[np.ndarray], branches: int = 1,
               hypotheses: int = 1) -> List[Detection]:
        """Preambles present in one power delay profile per root.

        pdps
            One profile per root, already summed over the receive branches.
        branches
            How many were summed. Not inferable from the profiles, and needed:
            a tap of one branch is exponential, a sum of N is chi square with
            2N degrees of freedom, and both the median noise floor and the
            threshold follow that distribution. Reading a combined profile as
            one branch puts the threshold well high.
        hypotheses
            How many profiles this one was picked out of, the same window
            correlated against several candidate frequency errors. Picking the
            strongest is another set of tries for noise to win, as another
            window is, so the rate is shared over these too. Omitting them
            leaves the measured rate well above the target.
        """
        if branches < 1:
            raise ValueError(f"branches must be at least 1, got {branches}")
        if hypotheses < 1:
            raise ValueError(
                f"hypotheses must be at least 1, got {hypotheses}"
            )

        config = self.config

        slots = self._preamble_map()
        n_cs = n_cs_from_config(
            config.zero_correlation_config, config.high_speed_flag
        )
        profiles = self._validate_pdps(pdps, slots[-1].root_offset + 1)
        if not profiles:
            return []

        windows_by_root, taps_by_root, window_pfa, span = self._search_layout(
            slots, n_cs, profiles[0].size, hypotheses
        )

        # taken from the same span the search was cut to, so that what this
        # block advertises and what it looks at cannot drift apart
        reach = self.searchable_timing_advance(n_cs, profiles[0].size)

        # every window searches the same number of taps and carries the same
        # share of the rate, so one number serves the whole call. An explicit
        # threshold_factor is taken as given and gets no margin.
        if self.threshold_factor > 0.0:
            factor = self.threshold_factor
        else:
            factor = self._threshold_from_pfa(span, window_pfa, branches)
            factor *= 10.0 ** (self.interference_margin / 10.0)

        detections: List[Detection] = []
        for root_offset in sorted(windows_by_root):
            pdp = profiles[root_offset]
            windows = windows_by_root[root_offset]

            noise = self._noise_floor(pdp, branches)
            candidates, values, bins = self._peaks(
                pdp, windows, taps_by_root[root_offset], noise
            )

            floor = factor * noise
            for index, candidate in enumerate(candidates):
                # leakage only raises the bar, so a peak below the noise
                # threshold is rejected without computing it - which on an
                # idle frame is every window of every root
                if candidate.peak_value <= floor:
                    continue

                # side lobe power lands on top of the noise in that tap: the
                # two add rather than compete
                leakage = self._leakage_floor(values, bins, index, pdp.size)
                threshold = floor + leakage

                detection = self._to_detection(candidate, threshold)
                if detection is None:
                    continue
                if detection.timing_advance > reach:
                    # both sides round the same way, so this is unreachable;
                    # kept as the invariant the two share
                    continue
                detections.append(detection)

        return self._strongest_per_preamble(detections)

    @staticmethod
    def _strongest_per_preamble(detections: List[Detection]) -> List[Detection]:
        """One report per preamble, ordered by index.

        A preamble owns its nominal window and, in a restricted set, the two
        replica windows a frequency offset moves it into. Several can hold a
        tap above the threshold, but all describe one access attempt.
        """
        best: Dict[int, Detection] = {}
        for detection in detections:
            current = best.get(detection.preamble_index)
            if current is None or detection.peak_value > current.peak_value:
                best[detection.preamble_index] = detection
        return [best[index] for index in sorted(best)]

    @staticmethod
    def _validate_pdps(pdps: Any, root_count: int) -> List[np.ndarray]:
        """Normalise the input to a list of equal length real power profiles."""
        if pdps is None:
            return []

        profiles: List[np.ndarray] = []
        for index, raw in enumerate(pdps):
            arr = np.asarray(raw)
            if arr.ndim != 1:
                raise ValueError(
                    f"pdps[{index}] must be one dimensional, got shape {arr.shape}"
                )
            # complex input is accepted so the block can be driven straight
            # from the IFFT output, without a separate power stage
            power = np.abs(arr) ** 2 if np.iscomplexobj(arr) else arr.astype(float)
            if power.size and power.size < N_ZC_FDD:
                raise ValueError(
                    f"pdps[{index}] holds {power.size} bins, at least "
                    f"{N_ZC_FDD} are needed"
                )
            profiles.append(power)

        sizes = {p.size for p in profiles}
        if len(sizes) > 1:
            raise ValueError(f"all pdps must share one length, got {sorted(sizes)}")

        profiles = [p for p in profiles if p.size]
        if profiles and len(profiles) < root_count:
            raise ValueError(
                f"this configuration spreads its preambles over {root_count} "
                f"roots, got {len(profiles)} profiles"
            )
        return profiles

    def _auto_guard(self, length: int) -> int:
        """Bins covered by the main lobe of one delay tap."""
        if self.guard_bins >= 0:
            return self.guard_bins
        return int(math.ceil(length / N_ZC_FDD)) + 1

    @staticmethod
    def _timing_advance_bins(timing_advance: int, length: int) -> int:
        """Profile bins a timing advance of that many units corresponds to."""
        delay_seconds = timing_advance * T_A_GRANULARITY / F_S
        return int(math.floor(delay_seconds * length * DELTA_F_RA))

    def zone_timing_advance(self, n_cs: int) -> int:
        """Largest advance a cyclic shift zone can tell apart.

        A delay longer than the zone lands in the region of the next cyclic
        shift, where it is a different preamble rather than a late one, so the
        configuration itself caps how far a cell may reach.

        This is the zone alone, a property of the configuration. It is not
        what the block searches, which is shorter - `searchable_timing_advance`
        is that, and it is the one a cell should be dimensioned against.
        """
        if n_cs == 0:
            return T_A_MAX
        delay_seconds = n_cs / (N_ZC_FDD * DELTA_F_RA)
        return min(T_A_MAX, int(delay_seconds * F_S / T_A_GRANULARITY))

    def _max_timing_advance(self, n_cs: int) -> int:
        zone = self.zone_timing_advance(n_cs)
        if self.max_timing_advance is None:
            return zone
        return min(self.max_timing_advance, zone)

    def _window_geometry(self, n_cs: int, length: int):
        """Width of one preamble's zone, the guard at its tail, what is left.

        In one place because the reach this block advertises and the reach it
        searches both follow from these three.
        """
        if n_cs == 0:
            # a single preamble per root, the whole profile belongs to it
            return length, 0, length

        oversampling = length / N_ZC_FDD
        width = max(int(round(n_cs * oversampling)), 1)
        guard = min(self._auto_guard(length), width - 1)
        return width, guard, max(width - guard, 1)

    def _search_span(self, n_cs: int, length: int) -> int:
        """Bins of every window the search covers.

        The zone gives the width; the guard at its tail goes to the main lobe
        of the next cyclic shift, `delay_spread` off what remains, and
        `max_timing_advance` cuts that down again. Searching past any of them
        buys only chances to fire on interference, and a shorter window holds
        the same false alarm rate at a lower threshold.
        """
        _width, _guard, usable = self._window_geometry(n_cs, length)

        # an echo this far behind a preamble at the end of its zone lands in
        # the next cyclic shift's window and reads as a preamble of its own
        spread = int(math.ceil(self.delay_spread * length * DELTA_F_RA))
        usable = max(usable - spread, 1)

        reach = self._timing_advance_bins(self._max_timing_advance(n_cs), length)
        return max(min(usable, reach + 1), 1)

    def searchable_timing_advance(self, n_cs: int, length: int) -> int:
        """Largest advance a report can carry, on a profile this long.

        `zone_timing_advance` follows from the configuration alone. This is
        smaller - the guard band goes before the search starts and
        `delay_spread` on top of it - and it is the one to dimension a cell
        against.
        """
        span = self._search_span(n_cs, length)
        delay_seconds = (span - 1) / (length * DELTA_F_RA)
        return min(T_A_MAX, int(round(delay_seconds * F_S / T_A_GRANULARITY)))

    def _windows(
        self,
        slots: Sequence[PreambleSlot],
        length: int,
        n_cs: int,
        with_doppler: bool,
    ) -> List[_Window]:
        """Delay region owned by every preamble of one root."""
        span = self._search_span(n_cs, length)

        if n_cs == 0:
            # a single preamble per root, the whole profile belongs to it
            return [_Window(slot, 0, span, length, 0) for slot in slots]

        oversampling = length / N_ZC_FDD

        windows: List[_Window] = []
        for slot in slots:
            shifts = [(0, 0)]
            if with_doppler and slot.d_u:
                shifts += self._replica_shifts(slot)

            for shift, doppler_offset in shifts:
                # a preamble shifted by c is seen at -c in the delay domain;
                # propagation delay then moves the peak to higher bins
                start = int(round((slot.c_v + shift) * oversampling)) % length
                start = (length - start) % length
                windows.append(
                    _Window(
                        slot=slot,
                        start=start,
                        span=span,
                        length=length,
                        doppler_offset=doppler_offset,
                    )
                )

        return windows

    @staticmethod
    def _replica_shifts(slot: PreambleSlot) -> List[tuple]:
        """Where a frequency offset moves this preamble, and by how much.

        A frequency error of k subcarriers shifts a Zadoff-Chu sequence by
        -k * p, where p is the inverse of the root. `root_distance` reports
        the folded magnitude, min(p, n_zc - p), so the replica of a positive
        error sits below the nominal shift for a root that was not folded and
        above it for one that was. Reporting the shift without undoing that
        fold would flip the sign of the error the detection is tagged with.
        """
        folded = pow(slot.u_zc, -1, N_ZC_FDD) >= N_ZC_FDD / 2
        above, below = (1, -1) if folded else (-1, 1)
        return [(slot.d_u, above), (-slot.d_u, below)]

    def _search_layout(self, slots, n_cs: int, length: int,
                       hypotheses: int = 1):
        """Windows of every root, the taps they cover, and the rate each holds.

        None of it depends on the profile, only on the configuration and how
        long the profile is, so it survives between calls.
        """
        key = (
            length,
            n_cs,
            self.config.root_sequence_index,
            self.config.high_speed_flag,
            self.detect_doppler,
            self.guard_bins,
            self.max_timing_advance,
            self.delay_spread,
            hypotheses,
        )
        if self._layout is not None and self._layout[0] == key:
            return self._layout[1:]

        by_root: Dict[int, List[PreambleSlot]] = {}
        for slot in slots:
            by_root.setdefault(slot.root_offset, []).append(slot)

        windows_by_root = {
            root_offset: self._windows(
                root_slots,
                length,
                n_cs,
                self.config.high_speed_flag and self.detect_doppler,
            )
            for root_offset, root_slots in by_root.items()
        }

        # every window of one root covers the same number of taps, so the taps
        # of all of them are one array and the whole root is searched at once
        taps_by_root = {
            root_offset: _taps(windows, length)
            for root_offset, windows in windows_by_root.items()
            if windows
        }

        # the target rate is per call, and a call searches every window of
        # every root in every hypothesis, so each carries its share. Hypotheses
        # see the same noise turned by different amounts, so counting them as
        # separate tries is pessimistic and the rate lands under the target.
        searched = sum(len(windows) for windows in windows_by_root.values())
        window_pfa = -math.expm1(
            math.log1p(-self.pfa) / max(1, searched * hypotheses)
        )

        # the span follows from n_cs, the profile length and the reach, none of
        # which vary between roots, so there is one for the whole call
        spans = {windows[0].span for windows in windows_by_root.values()
                 if windows}
        if len(spans) != 1:
            raise ValueError(f"windows of one call must share a span, got {spans}")

        self._layout = (
            key, windows_by_root, taps_by_root, window_pfa, spans.pop()
        )
        return self._layout[1:]

    @staticmethod
    def _peaks(pdp: np.ndarray, windows: Sequence[_Window],
               taps: np.ndarray, noise: float):
        """Strongest tap of every window of one root.

        Returned as the candidates and, beside them, their heights and
        positions as arrays: read off the same lookups, so they cost nothing
        here and save `_leakage_floor` walking the list per candidate.
        """
        power = pdp[taps]
        local = np.argmax(power, axis=1)
        rows = np.arange(len(windows))

        values = power[rows, local]
        bins = taps[rows, local]

        candidates = [
            _Candidate(
                window=window,
                local_peak=int(local[row]),
                peak_bin=int(bins[row]),
                peak_value=float(values[row]),
                noise=noise,
            )
            for row, window in zip(rows, windows)
        ]
        return candidates, values, bins

    def _leakage_floor(
        self, values: np.ndarray, bins: np.ndarray, index: int, length: int
    ) -> float:
        """Power a stronger peak of the same profile leaks into this window.

        Zero padding a block of ``N_ZC_FDD`` bins up to ``length`` convolves
        the profile with a Dirichlet kernel, whose skirt falls off as one over
        the distance. A candidate that does not clear that skirt is a side lobe
        of a stronger preamble rather than a preamble of its own.

        All windows of the root are weighed at once: the arithmetic is the
        same one at a time, but 64 preambles on one root then cost more per
        frame here than the transforms that built the profile.
        """
        if self.sidelobe_margin <= 0.0:
            return 0.0

        # a peak of equal height is not a side lobe of itself, so this drops
        # the candidate along with everything below it
        stronger = values > values[index]
        if not stronger.any():
            return 0.0

        distance = np.abs(bins - bins[index])
        distance = np.minimum(distance, length - distance)

        # at zero distance the two candidates are the same tap, not one
        # leaking into the other, and the skirt is undefined
        stronger &= distance >= 1
        if not stronger.any():
            return 0.0

        envelope = (length / (math.pi * N_ZC_FDD * distance[stronger])) ** 2
        return self.sidelobe_margin * float((values[stronger] * envelope).max())

    @staticmethod
    def _to_detection(
        candidate: _Candidate, threshold: float
    ) -> Optional[Detection]:
        if candidate.peak_value <= threshold or candidate.peak_value <= 0.0:
            return None

        window = candidate.window
        delay_seconds = candidate.local_peak / (window.length * DELTA_F_RA)
        timing_advance = int(round(delay_seconds * F_S / T_A_GRANULARITY))

        return Detection(
            preamble_index=window.slot.preamble_index,
            root_offset=window.slot.root_offset,
            cyclic_shift=window.slot.c_v,
            peak_bin=candidate.peak_bin,
            peak_value=candidate.peak_value,
            noise_floor=candidate.noise,
            peak_to_noise=(
                candidate.peak_value / candidate.noise
                if candidate.noise > 0.0
                else float("inf")
            ),
            delay_bins=candidate.local_peak,
            delay_seconds=delay_seconds,
            timing_advance=timing_advance,
            doppler_offset=window.doppler_offset,
        )

    @staticmethod
    def _noise_floor(pdp: np.ndarray, branches: int = 1) -> float:
        """Mean noise power of a whole profile, estimated from its median.

        One window alone is too noisy: after the guard bins it keeps a handful
        of independent taps, and the spread of that estimate drives the false
        alarm rate far above the one the threshold was derived for. The median
        over the whole profile uses every tap and, unlike the mean, is not
        dragged up by the preambles present, as long as they occupy less than
        half of it.

        A tap of one branch is exponential, whose median is ``ln 2`` times its
        mean. A sum over branches is not: the ratio climbs towards one as they
        are added. Scaling by ``ln 2`` throughout would report a floor too high,
        and so a threshold too high, paid for in detections.
        """
        if pdp.size == 0:
            return 0.0

        # the median of the sum, in units of the mean of one branch
        median = _tail_inverse(0.5, branches)
        return float(np.median(pdp)) * branches / median

    @staticmethod
    def _threshold_from_pfa(span: int, pfa: float, branches: int = 1) -> float:
        """Threshold over the noise floor for one window's share of the rate.

        The strongest of ``n`` independent taps stays below ``t`` times the
        noise floor with probability ``(1 - P(t))**n``, where ``P`` is the tail
        of one tap. Summing branches makes that tail lighter, so the same rate
        is held at a lower multiple of the floor.

        Zero padding correlates neighbouring bins, which would suggest counting
        ``span / oversampling`` independent taps, but the estimated noise floor
        widens the tail of the maximum. The bin count is used directly, which
        lands the rate under the target rather than over it.
        """
        tap_pfa = -math.expm1(math.log1p(-pfa) / max(1, span))
        return _tail_inverse(tap_pfa, branches) / branches
