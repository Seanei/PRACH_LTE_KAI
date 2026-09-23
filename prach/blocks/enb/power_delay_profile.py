from typing import List

import numpy as np

from prach.blocks.ue import DFTBlock
from prach.pipeline.block import Block
from prach.pipeline.config import PRACHConfiguration
from prach.pipeline.spec import N_FFT, N_ZC_FDD, U_ZC_FDD
from prach.math import multi_bef_detect, zadoff_chu

from .subcarrier_demapping import SubcarrierDemappingBlock

# Roots per transform call. One at a time spends more on call overhead than on
# the transform; all 64 at once no longer fit in cache.
ROOT_BLOCK = 16


class PowerDelayProfileBlock(Block):
    """Windows of one PRACH opportunity, as one profile per root sequence."""

    def __init__(
        self,
        config: PRACHConfiguration,
        root_count: int,
        *,
        branches: int = 1,
        n_ifft: int = 2048,
        frequency_hypotheses: int = 2,
    ):
        """
        root_count
            Roots to correlate against, one profile each. Stated rather than
            derived, so that it cannot differ from what `DetectorBlock`
            searches.
        branches
            Receive branches, and so windows per call. Stated rather than
            counted: the number summed into a profile sets the detector's
            threshold.
        n_ifft
            Profile length. The correlation is zero padded from the 839 PRACH
            bins to this before the inverse transform, which sets how finely a
            peak can be placed.
        frequency_hypotheses
            Fractional frequency errors tried, on the grid
            k * DELTA_F_RA / hypotheses. Two covers 0 and half a subcarrier,
            leaving a quarter as the worst unsearched case. One switches the
            search off.
        """
        super().__init__(config)

        if root_count < 1:
            raise ValueError(f"root_count must be at least 1, got {root_count}")

        if branches < 1:
            raise ValueError(f"branches must be at least 1, got {branches}")

        if n_ifft < N_ZC_FDD:
            raise ValueError(
                f"n_ifft must be at least {N_ZC_FDD} to hold the PRACH bins, "
                f"got {n_ifft}"
            )

        if frequency_hypotheses < 1:
            raise ValueError(
                f"frequency_hypotheses must be at least 1, got "
                f"{frequency_hypotheses}"
            )

        self.branches = branches
        self.n_ifft = n_ifft
        self.frequency_hypotheses = frequency_hypotheses

        self.dft = DFTBlock(config)
        self.subcarrier_demapping = SubcarrierDemappingBlock(config)

        self._multiply = multi_bef_detect

        # built once per window length, not per frame
        self._rotations = {}

        # roots in the frequency domain, as one array so that a window is
        # correlated against all of them in one call
        self._references = np.stack([
            self.dft.transform(
                np.array(
                    zadoff_chu(
                        U_ZC_FDD[
                            (config.root_sequence_index + offset) % len(U_ZC_FDD)
                        ],
                        N_ZC_FDD,
                    )
                )
            )
            for offset in range(root_count)
        ])

    @property
    def root_count(self) -> int:
        """Roots a window is correlated against, one profile each."""
        return self._references.shape[0]

    def profiles_of(self, window) -> np.ndarray:
        """Profiles of one window seen by one branch: `profiles([window])`.

        A name rather than a shape the block infers, so that the reading meant
        cannot be arrived at by accident.
        """
        if self.branches != 1:
            raise ValueError(
                f"this block serves {self.branches} receive branches, so a "
                f"lone window is not enough; `profiles` takes one per branch"
            )
        return self.profiles([window])

    def profiles(self, windows) -> np.ndarray:
        """Profiles of one PRACH window, one per root, summed over the branches.

        `windows` is one window per receive branch, all of one length. One
        branch still passes a sequence of one, or calls `profiles_of`.

        A whole number of subcarriers of frequency error leaves the preamble
        on the grid, where a Zadoff-Chu sequence turns it into a cyclic shift
        and the detector finds it in a replica window. A fraction does not: at
        half a subcarrier a format 0 peak drops by (2/pi)**2. Formats 2 and 3
        combine two repetitions coherently, which doubles the phase the same
        error turns over, and there half a subcarrier is the null of that
        sinc: the peak goes altogether. So the window is also correlated after
        being turned back by a fraction of a subcarrier, and the hypothesis
        that answered is kept.

        The error belongs to the transmitter, so it is one error for the whole
        array: the hypothesis is chosen once, on the profiles of all branches
        together. Per branch it would let two branches answer to different
        errors and give noise one try per branch instead of one per call.
        """
        windows = self._windows_per_branch(windows)

        best = None
        for hypothesis in range(self.frequency_hypotheses):
            combined = None
            for window in windows:
                per_branch = self._branch_profiles(window, hypothesis)
                combined = (
                    per_branch if combined is None else combined + per_branch
                )

            peak = float(combined.max())
            if best is None or peak > best[0]:
                best = (peak, combined)

        return best[1]

    def _windows_per_branch(self, windows) -> List[np.ndarray]:
        """The one window per branch this call was given, or why it was not.

        Nothing is inferred from the shape of what arrived: the number of
        branches summed into a profile is what the detector's threshold is
        set from.
        """
        try:
            per_branch = [np.asarray(window, dtype=complex) for window in windows]
        except TypeError:
            raise ValueError(
                f"windows must be a sequence of one window per receive "
                f"branch, got {type(windows).__name__}"
            )

        if not per_branch:
            raise ValueError(
                f"this block serves {self.branches} receive branches, got no "
                f"windows"
            )

        ranks = {window.ndim for window in per_branch}
        if ranks != {1}:
            if ranks == {0}:
                # a single window iterated apart into its own samples
                raise ValueError(
                    "windows must be one window per receive branch, not the "
                    "samples of a single window; `profiles_of` takes one "
                    "window"
                )
            raise ValueError(
                f"a window is one dimensional, got windows of "
                f"{sorted(ranks)} dimensions"
            )

        lengths = {window.size for window in per_branch}
        if len(lengths) != 1:
            raise ValueError(
                f"every branch sees the same window, so all of them are one "
                f"length, got {sorted(lengths)}"
            )

        if lengths == {0}:
            raise ValueError("a window must hold samples, got empty ones")

        if len(per_branch) != self.branches:
            raise ValueError(
                f"this block serves {self.branches} receive branches, so it "
                f"takes that many windows, got {len(per_branch)}"
            )

        return per_branch

    def _branch_profiles(self, preamble: np.ndarray, hypothesis: int) -> np.ndarray:
        """One branch's profiles, turned back by one frequency hypothesis.

        Formats 2 and 3 send the sequence twice. The repetitions are summed
        before the transform rather than their profiles after it: the preamble
        doubles in amplitude while the noise grows as its square root, the
        full 3 dB. The same turning back aligns them, so it costs nothing
        extra.
        """
        preamble = np.asarray(preamble, dtype=complex)
        repetitions = max(len(preamble) // N_FFT, 1)

        rotation = self._rotation(len(preamble), hypothesis)
        window = preamble if rotation is None else preamble * rotation

        if repetitions > 1:
            blocks = np.stack(np.array_split(window, repetitions))
            window = blocks.sum(axis=0) / np.sqrt(repetitions)

        bins = self.subcarrier_demapping.demap(self.dft.transform(window))
        return self.correlate(bins)

    def _rotation(self, length: int, hypothesis: int):
        """What to multiply a window of `length` by to undo one hypothesis.

        None for hypothesis 0, so that the no-error case costs no arithmetic.
        """
        if hypothesis == 0:
            return None

        key = (length, hypothesis)
        if key not in self._rotations:
            # DELTA_F_RA / F_S is 1 / N_FFT, so this is a fraction of a
            # subcarrier
            turn = hypothesis / (self.frequency_hypotheses * N_FFT)
            self._rotations[key] = np.exp(
                -2j * np.pi * turn * np.arange(length)
            )
        return self._rotations[key]

    def correlate(self, prach_bins: np.ndarray) -> np.ndarray:
        """Power delay profile of the PRACH bins against every root, a row each.

        One array rather than a list of profiles, so that several branches are
        added in one operation. Roots are walked in blocks of ROOT_BLOCK.

        The inverse transform is `np.fft.ifft` over the last axis, not the
        explicit loop in `prach.math`, which takes one dimension at a time and
        would cost seconds per root here.
        """
        bins = np.asarray(prach_bins, dtype=complex)
        roots = self._references.shape[0]

        profiles = np.empty((roots, self.n_ifft))
        for start in range(0, roots, ROOT_BLOCK):
            block = self._references[start:start + ROOT_BLOCK]

            # correlation in frequency is multiplication by the conjugate
            # root; zero padding then oversamples the delay axis
            padded = np.zeros((block.shape[0], self.n_ifft), dtype=complex)
            padded[:, :N_ZC_FDD] = self._multiply(
                np.broadcast_to(bins, block.shape), block
            )

            profiles[start:start + block.shape[0]] = (
                np.abs(np.fft.ifft(padded, axis=-1)) ** 2
            )

        return profiles
