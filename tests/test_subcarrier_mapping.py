import unittest
import pathlib

import numpy as np

from dataclasses import dataclass, field
from typing import Dict, Any

from prach.pipeline import CommonData
from prach.blocks.ue import SubcarrierMappingBlock


@dataclass(kw_only=True)
class CommonDataEx(CommonData):
    meta: Dict[str, Any] = field(default_factory=dict)


def read_complex_file(path):
    data = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            line = line.replace("i", "j")
            data.append(complex(line))

    return np.array(data, dtype=complex)


class TestSubcarrierMapping(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.input_data = read_complex_file(
            pathlib.Path("tests/input_data.txt")
        )

        cls.reference_output = read_complex_file(
            pathlib.Path("tests/output_data.txt")
        )

    def perform_test(self):

        config = {
            "n_ul_rb": 100,
            "n_ra_prb_offset": 0,
            "phi": 7,
            "delta_f_ra": 1250,
            "delta_f": 15000
        }

        block = SubcarrierMappingBlock(config)

        data = CommonDataEx()

        data.meta = {
            "dft": self.input_data,
            "n_ul_rb": config["n_ul_rb"],
            "n_ra_prb_offset": config["n_ra_prb_offset"]
        }

        block.process(data)

        python_output = np.array(
            data.meta.get("Subcarrier_Mapping", []),
            dtype=complex
        )

        tolerance = 1e-7

        error_vector = np.abs(
            python_output - self.reference_output
        )

        max_error = np.max(error_vector)

        mismatch_count = np.sum(
            error_vector > tolerance
        )

        print(f"MAX ERROR: {max_error}")
        print(f"MISMATCH COUNT: {mismatch_count}")

        self.assertTrue(
            np.allclose(
                python_output,
                self.reference_output,
                atol=tolerance
            ),
            "Subcarrier Mapping differs from MATLAB reference"
        )

    def test_mapping(self):

        self.perform_test()


if __name__ == "__main__":
    unittest.main()
