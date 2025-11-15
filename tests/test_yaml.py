import copy
import unittest
from collections import OrderedDict

from prach.yaml import load, YAMLError


class TestSimpleYAML(unittest.TestCase):

    def test_simple_scalars(self):
        yaml_text = """
        null_val: null
        true_val: true
        false_val: false
        int_val: 123
        float_val: 3.14
        string_val: hello
        quoted_val1: "world"
        quoted_val2: 'foo'
        """
        result = load(yaml_text)
        expected = OrderedDict(
            [
                ("null_val", None),
                ("true_val", True),
                ("false_val", False),
                ("int_val", 123),
                ("float_val", 3.14),
                ("string_val", "hello"),
                ("quoted_val1", "world"),
                ("quoted_val2", "foo"),
            ]
        )
        self.assertEqual(result, expected)

    def test_sequences(self):
        yaml_text = """
        list1:
          - 1
          - 2
          - 3
        list2: [a, b, c]
        nested_list:
          - [1, 2]
          - [3, 4]
        """
        result = load(yaml_text)
        expected = OrderedDict(
            [
                ("list1", [1, 2, 3]),
                ("list2", ["a", "b", "c"]),
                ("nested_list", [[1, 2], [3, 4]]),
            ]
        )
        self.assertEqual(result, expected)

    def test_mappings(self):
        yaml_text = """
        mapping1:
          key1: val1
          key2: val2
        flow_map: {a: 1, b: 2, c: 3}
        nested_map:
          outer:
            inner: value
        """
        result = load(yaml_text)
        expected = OrderedDict(
            [
                ("mapping1", OrderedDict([("key1", "val1"), ("key2", "val2")])),
                ("flow_map", OrderedDict([("a", 1), ("b", 2), ("c", 3)])),
                (
                    "nested_map",
                    OrderedDict([("outer", OrderedDict([("inner", "value")]))]),
                ),
            ]
        )
        self.assertEqual(result, expected)

    def test_multiline_strings(self):
        yaml_text = """
        literal_block: |
          line1
          line2
        folded_block: >
          line1
          line2
        """
        result = load(yaml_text)
        expected = OrderedDict(
            [
                ("literal_block", "line1\nline2"),
                ("folded_block", "line1 line2"),
            ]
        )
        self.assertEqual(result, expected)

    def test_anchors_and_aliases_reference(self):
        yaml_text = """
        a: &X
          k: 1
        b: *X
        """
        result = load(yaml_text, alias_mode="reference")
        # изменение b должно влиять на a при reference
        result["b"]["k"] = 99
        self.assertEqual(result["a"]["k"], 99)

    def test_anchors_and_aliases_deepcopy(self):
        yaml_text = """
        a: &X
          k: 1
        b: *X
        """
        result = load(yaml_text, alias_mode="deepcopy")
        result["b"]["k"] = 99
        # изменение b не должно влиять на a при deepcopy
        self.assertEqual(result["a"]["k"], 1)

    def test_merge_key(self):
        yaml_text = """
        defaults: &defaults
          a: 1
          b: 2
        merged:
          <<: *defaults
          c: 3
        """
        result = load(yaml_text)
        expected = OrderedDict(
            [
                ("defaults", OrderedDict([("a", 1), ("b", 2)])),
                ("merged", OrderedDict([("a", 1), ("b", 2), ("c", 3)])),
            ]
        )
        self.assertEqual(result, expected)

    def test_multidoc(self):
        yaml_text = """
        ---
        a: 1
        ---
        b: 2
        """
        result = load(yaml_text)
        expected = [
            OrderedDict([("a", 1)]),
            OrderedDict([("b", 2)]),
        ]
        self.assertEqual(result, expected)

    def test_flow_nested_structures(self):
        yaml_text = """
        nested: {a: [1, 2, {x: 'y'}], b: {k: 10}}
        """
        result = load(yaml_text)
        expected = OrderedDict(
            [
                (
                    "nested",
                    OrderedDict(
                        [
                            ("a", [1, 2, OrderedDict([("x", "y")])]),
                            ("b", OrderedDict([("k", 10)])),
                        ]
                    ),
                )
            ]
        )
        self.assertEqual(result, expected)

    def test_tags(self):
        yaml_text = """
        tagged: !mytag value
        """
        result = load(yaml_text)
        expected = OrderedDict([("tagged", ("!mytag", "value"))])
        self.assertEqual(result, expected)

    def test_alias_error(self):
        yaml_text = """
        a: *unknown
        """
        with self.assertRaises(YAMLError):
            load(yaml_text)


if __name__ == "__main__":
    unittest.main()
