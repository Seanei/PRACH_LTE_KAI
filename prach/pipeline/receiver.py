from typing import List

import numpy as np

from .config import PRACHConfiguration
from .spec import NUM_SUBFRAMES, SAMPLES_PER_SUBFRAME

# what one radio frame looks like at an antenna connector
FRAME_SHAPE = (NUM_SUBFRAMES, SAMPLES_PER_SUBFRAME)


class Receiver:
    """Walks a radio frame down to the preambles it carries::

        subframe demapping -> power delay profile -> detector

    with one subframe demapper per receive branch. What is left here is
    cutting the windows out of a frame, checking the branches agree on what
    they found, and telling the detector what it cannot work out on its own.
    """

    def __init__(
        self,
        config: PRACHConfiguration,
        *,
        n_ifft: int = 2048,
        frequency_hypotheses: int = 2,
        branches: int = 1,
        **detector_options,
    ):
        """
        n_ifft, frequency_hypotheses
            Passed to `PowerDelayProfileBlock`, which is what they describe.
        branches
            Receive branches, and so frames per call. Every branch carries the
            same preamble through a channel of its own, so their profiles are
            summed before detection: that needs no phase relation between
            them, only the same air. Each gets its own subframe demapper,
            since a preamble running past the end of a frame leaves state in
            it.
        """
        # imported here to avoid a circular import at package load
        from prach.blocks.enb import (
            DetectorBlock,
            PowerDelayProfileBlock,
            SubframeDemappingBlock,
        )

        if branches < 1:
            raise ValueError(f"branches must be at least 1, got {branches}")

        self.config = config
        self.branches = branches

        # one per branch: a preamble running past the end of a frame leaves
        # its head in the demapper, and branches are separate streams
        self.subframe_demappers = [
            SubframeDemappingBlock(config) for _ in range(branches)
        ]
        self.detector = DetectorBlock(config, **detector_options)

        # the root count is the detector's answer, asked once rather than
        # derived twice and trusted to match
        self.power_delay_profile = PowerDelayProfileBlock(
            config,
            self.detector.root_count,
            branches=branches,
            n_ifft=n_ifft,
            frequency_hypotheses=frequency_hypotheses,
        )

    @property
    def n_ifft(self) -> int:
        """Length of the profiles the detector is handed."""
        return self.power_delay_profile.n_ifft

    @property
    def frequency_hypotheses(self) -> int:
        """Frequency errors every window is correlated against."""
        return self.power_delay_profile.frequency_hypotheses

    def receive(self, frames, sf_n: int = 0):
        """Preambles found in every PRACH window of one radio frame.

        `frames` is one frame per receive branch, or a single frame when the
        base station has one branch. Returns one entry per window, holding the
        subframe it started in and the detections made in it.

        This is the base station between the antenna connectors and the
        preambles it reports, where TS 36.141 draws the device under test.
        What the detector cannot work out on its own - branches summed into a
        profile, hypotheses it was picked out of, both of which set how often
        noise alone clears a threshold - is filled in here.
        """
        branch_frames = self._frames_per_branch(frames)

        # every branch is its own stream, with its own demapper carrying any
        # preamble that ran past the end of the frame
        per_branch = [
            demapper.demap(frame, sf_n)
            for demapper, frame in zip(self.subframe_demappers, branch_frames)
        ]


        counts = {len(windows) for windows in per_branch}
        if len(counts) != 1:
            raise ValueError(
                f"branches disagree on how many PRACH windows this frame "
                f"carries, got {sorted(counts)}"
            )

        results = []
        for index in range(counts.pop()):
            starts = {windows[index][0] for windows in per_branch}
            if len(starts) != 1:
                raise ValueError(
                    f"branches disagree on where window {index} starts, got "
                    f"{sorted(starts)}"
                )

            windows = [branch[index][1] for branch in per_branch]
            results.append((
                starts.pop(),
                self.detector.detect(
                    self.power_delay_profile.profiles(windows),
                    # what this base station has, not what this call holds:
                    # the number summed over and the number the threshold is
                    # derived from are one fact
                    self.branches,
                    self.frequency_hypotheses,
                ),
            ))
        return results

    def _frames_per_branch(self, frames) -> List[np.ndarray]:
        """One frame per branch, however the caller spelled it.

        The public edge, and the one place allowed to read what it was handed
        from the shape of it: a single branch should not have to wrap its
        frame in a list. A shape that is neither reading is rejected here
        rather than carried inwards.

        Nothing is converted - copying every branch of every frame on the way
        in costs more than the demapping does.
        """
        try:
            shape = np.shape(frames)
        except ValueError:
            # ragged input, which numpy refuses to describe
            raise ValueError(
                "every branch must be handed the same frame, got frames of "
                "differing shapes"
            )

        # a frame is subframes by samples, so a run of them is three
        # dimensional; a lone frame is the only branch
        if shape == FRAME_SHAPE:
            branch_frames = [frames]
        elif len(shape) == 3 and shape[1:] == FRAME_SHAPE:
            branch_frames = list(frames)
        else:
            raise ValueError(
                f"a frame is {FRAME_SHAPE[0]} subframes of "
                f"{FRAME_SHAPE[1]} samples, and a run of them one per "
                f"branch; got shape {shape}"
            )

        if len(branch_frames) != self.branches:
            raise ValueError(
                f"this receiver has {self.branches} branches, got "
                f"{len(branch_frames)} frames"
            )
        return branch_frames
