import unittest
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional

from prach.pipeline import *


@dataclass(kw_only=True)
class CommonDataEx(CommonData):
    meta: Dict[str, Any] = field(default_factory=dict)


class MyMode(Enum):
    A = "mode_a"
    B = "mode_b"


class TestBlockForValidate(Block):
    int_field: int
    float_field: float
    str_field: str
    bool_field: bool
    opt_field: Optional[int]

    class Color(Enum):
        RED = "red"
        GREEN = "green"

    enum_field: Color

    def process(self, data):
        return data  # mocked


class MyBlock(Block):
    """Block to check values and enum parsing"""

    value: int
    mode: MyMode

    _internal: str = None  # should not be required

    def process(self, data: CommonData) -> CommonData:
        data.meta["my_value"] = self.value
        data.meta["my_mode"] = self.mode.value
        self._internal = f"calc-{self.value}"
        data.meta["_internal"] = self._internal
        return data


class AppendBlock(Block):
    """Block adds strings to meta["items"]"""

    text: str

    def process(self, data: CommonData) -> CommonData:
        arr = data.meta.get("items", [])
        arr.append(self.text)
        data.meta["items"] = arr
        return data


class BadBlock(Block):
    x: int

    def process(self, data):
        return "wrong"


# register them globally for pipeline to find
BlockRegistry.register(MyBlock)
BlockRegistry.register(AppendBlock)
BlockRegistry.register(BadBlock)


class TestBlockValidation(unittest.TestCase):

    def setUp(self):
        self.block = TestBlockForValidate(
            {
                "int_field": 1,
                "float_field": 1.0,
                "str_field": "x",
                "bool_field": True,
                "opt_field": None,
                "enum_field": TestBlockForValidate.Color.RED,
            }
        )

    def test_int_conversion(self):
        result = self.block._validate_and_convert("int_field", "42", int)
        self.assertEqual(result, 42)

    def test_float_conversion(self):
        result = self.block._validate_and_convert("float_field", "3.14", float)
        self.assertEqual(result, 3.14)

    def test_str_conversion(self):
        result = self.block._validate_and_convert("str_field", 123, str)
        self.assertEqual(result, "123")

    def test_bool_conversion(self):
        self.assertTrue(self.block._validate_and_convert("bool_field", "true", bool))
        self.assertFalse(self.block._validate_and_convert("bool_field", "false", bool))
        self.assertTrue(self.block._validate_and_convert("bool_field", 1, bool))
        self.assertFalse(self.block._validate_and_convert("bool_field", 0, bool))

    def test_optional_none(self):
        result = self.block._validate_and_convert("opt_field", None, Optional[int])
        self.assertIsNone(result)

    def test_optional_value(self):
        result = self.block._validate_and_convert("opt_field", "5", Optional[int])
        self.assertEqual(result, 5)

    def test_optional_error(self):
        with self.assertRaises(ValueError):
            self.block._validate_and_convert("int_field", None, int)

    def test_enum_by_member(self):
        result = self.block._validate_and_convert(
            "enum_field", TestBlockForValidate.Color.RED, TestBlockForValidate.Color
        )
        self.assertEqual(result, TestBlockForValidate.Color.RED)

    def test_enum_by_name(self):
        result = self.block._validate_and_convert(
            "enum_field", "RED", TestBlockForValidate.Color
        )
        self.assertEqual(result, TestBlockForValidate.Color.RED)

    def test_enum_by_value(self):
        result = self.block._validate_and_convert(
            "enum_field", "green", TestBlockForValidate.Color
        )
        self.assertEqual(result, TestBlockForValidate.Color.GREEN)

    def test_enum_invalid(self):
        with self.assertRaises(ValueError):
            self.block._validate_and_convert(
                "enum_field", "BLUE", TestBlockForValidate.Color
            )

    def test_type_mismatch_raises(self):
        with self.assertRaises(ValueError):
            self.block._validate_and_convert("int_field", "not-an-int", int)

        with self.assertRaises(ValueError):
            self.block._validate_and_convert("float_field", "abc", float)

    def test_missing_required_field(self):
        with self.assertRaises(ValueError):
            MyBlock({"value": 1})  # no mode

    def test_enum_parsing(self):
        b = MyBlock({"value": 5, "mode": "mode_a"})
        self.assertEqual(b.mode, MyMode.A)

        b2 = MyBlock({"value": 7, "mode": MyMode.B})
        self.assertEqual(b2.mode, MyMode.B)

        with self.assertRaises(ValueError):
            MyBlock({"value": 1, "mode": "unknown"})

    def test_type_casting(self):
        b = MyBlock({"value": "123", "mode": "mode_b"})
        self.assertEqual(b.value, 123)

    def test_internal_fields_not_required(self):
        b = MyBlock({"value": 2, "mode": "mode_a"})
        self.assertTrue(hasattr(b, "_internal"))
        self.assertIsNone(b._internal)


class TestBlockProcess(unittest.TestCase):

    def test_myblock_process(self):
        data = CommonDataEx()
        b = MyBlock({"value": 10, "mode": "mode_b"})
        out = b.process(data)

        self.assertEqual(out.meta["my_value"], 10)
        self.assertEqual(out.meta["my_mode"], "mode_b")
        self.assertEqual(out.meta["_internal"], "calc-10")

    def test_append_block(self):
        data = CommonDataEx()
        b = AppendBlock({"text": "hello"})
        out = b.process(data)
        self.assertEqual(out.meta["items"], ["hello"])

        out2 = b.process(out)
        self.assertEqual(out2.meta["items"], ["hello", "hello"])


class TestPipeline(unittest.TestCase):

    def test_pipeline_explicit_chain(self):
        cfg = {
            "config": {
                "MyBlock": {"value": 3, "mode": "mode_a"},
                "AppendBlock": {"text": "x"},
            },
            "chain": ["MyBlock", "AppendBlock"],
        }

        pipeline = Pipeline(cfg)
        result = pipeline.run(CommonDataEx())

        self.assertEqual(result.meta["my_value"], 3)
        self.assertEqual(result.meta["items"], ["x"])

    def test_pipeline_default_chain(self):
        cfg = {
            "config": {
                "MyBlock": {"value": 1, "mode": "mode_b"},
                "AppendBlock": {"text": "z"},
            }
        }

        pipeline = Pipeline(cfg)
        result = pipeline.run(CommonDataEx())

        self.assertEqual(result.meta["my_value"], 1)
        self.assertEqual(result.meta["items"], ["z"])

    def test_invalid_block_output(self):
        cfg = {"config": {"BadBlock": {"x": 1}}, "chain": ["BadBlock"]}

        p = Pipeline(cfg)
        with self.assertRaises(RuntimeError):
            p.run(CommonDataEx())


if __name__ == "__main__":
    unittest.main()
