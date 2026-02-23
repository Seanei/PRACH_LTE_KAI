import unittest
import pathlib
import struct
import mmap

from dataclasses import dataclass, field
from typing import Dict, Any
from prach.pipeline import CommonData
from prach.blocks.ue import PreambleGeneratorBlock


@dataclass(kw_only=True)
class CommonDataEx(CommonData):
    meta: Dict[str, Any] = field(default_factory=dict)


class TestPreambleGenerator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_preambles = []
        cls.test_configs = []

        for file in pathlib.Path("tests/preambles").glob("*.bin"):
            params = list(map(int, file.name.split("_")[:5]))

            config = {
                "preamble_format": params[0],
                "root_sequence_index": params[1],
                "preamble_index": params[2],
                "high_speed_flag": params[3],
                "zero_correlation_config": params[4],
            }

            with open(file, "rb") as f:
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                mv = memoryview(mm)

                count = len(mm) // 8
                floats = struct.unpack_from(f"{count * 2}f", mv)
                test_preamble = [
                    complex(floats[i], floats[i + 1]) for i in range(0, len(floats), 2)
                ]

                cls.test_preambles.append(test_preamble)
                cls.test_configs.append(config)

            mv.release()
            mm.close()

        cls.amount = len(cls.test_preambles)

    def perform_test(self, config, test_preamble):
        block = PreambleGeneratorBlock(config)
        data = CommonDataEx()
        block.process(data)
        preamble = data.meta.get("generated_preamble", [])

        tolerance = 1e-7

        if not all(abs(x - y) <= tolerance for x, y in zip(preamble, test_preamble)):
            self.fail("lists differ beyond tolerance", tolerance)

    def test_all_preambles(self):
        passed = 0

        for i in range(self.amount):
            print(f"{passed}/{self.amount}, current: {self.test_configs[i]}. ", end="")

            self.perform_test(self.test_configs[i], self.test_preambles[i])

            print("--- passed")

            passed += 1


if __name__ == "__main__":
    unittest.main()
