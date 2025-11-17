from typing import Any, List, Dict, Optional, Tuple
from collections import OrderedDict
import copy

from .utils import is_number_like
from .error import YAMLError


class FlowParser:
    """
    Parse flow values. Anchors dict and alias_mode must be provided
    """

    def __init__(
        self, text: str, anchors: Dict[str, Any], alias_mode: str = "reference"
    ):
        self.text = text
        self.i = 0
        self.n = len(text)
        self.anchors = anchors
        if alias_mode not in ("reference", "deepcopy"):
            raise ValueError("alias_mode must be 'reference' or 'deepcopy'")
        self.alias_mode = alias_mode

    def parse(self) -> Any:
        self._skip_whitespace()
        if self.i >= self.n:
            return None
        val = self._parse_value()
        self._skip_whitespace()
        if self.i < self.n:
            rem = self.text[self.i :].strip()
            if rem not in ("", ","):
                raise YAMLError("Trailing characters in flow value: %r" % rem)
        return val

    def _peek(self) -> Optional[str]:
        return self.text[self.i] if self.i < self.n else None

    def _next(self) -> Optional[str]:
        if self.i < self.n:
            ch = self.text[self.i]
            self.i += 1
            return ch
        return None

    def _skip_whitespace(self):
        while self.i < self.n and self.text[self.i] in " \t\r\n":
            self.i += 1

    def _parse_value(self) -> Any:
        self._skip_whitespace()
        ch = self._peek()
        if ch is None:
            return None
        if ch == "&":
            self._next()
            name = self._parse_name()
            self._skip_whitespace()
            val = self._parse_value()
            self.anchors[name] = val
            return val
        if ch == "*":
            self._next()
            name = self._parse_name()
            if name not in self.anchors:
                raise YAMLError("Unknown alias *%s" % name)
            if self.alias_mode == "reference":
                return self.anchors[name]
            else:
                return copy.deepcopy(self.anchors[name])
        if ch == "!":
            tag = self._parse_tag()
            self._skip_whitespace()
            val = self._parse_value()
            return (tag, val)
        if ch == "{":
            return self._parse_mapping()
        if ch == "[":
            return self._parse_sequence()
        if ch in ('"', "'"):
            return self._parse_quoted()
        return self._parse_plain_scalar()

    def _parse_name(self) -> str:
        start = self.i
        while self.i < self.n and self.text[self.i] not in " \t\r\n,]}":
            self.i += 1
        return self.text[start : self.i]

    def _parse_tag(self) -> str:
        self._next()  # skip '!'
        start = self.i
        while self.i < self.n and self.text[self.i] not in " \t\r\n,]}":
            self.i += 1
        return "!" + self.text[start : self.i]

    def _parse_quoted(self) -> str:
        quote = self._next()
        out = []
        while True:
            ch = self._next()
            if ch is None:
                raise YAMLError("Unterminated quoted string")
            if ch == quote:
                if quote == "'" and self._peek() == "'":
                    self._next()
                    out.append("'")
                    continue
                break
            if ch == "\\" and quote == '"':
                nxt = self._next()
                if nxt is None:
                    raise YAMLError("Invalid escape")
                if nxt == "n":
                    out.append("\n")
                elif nxt == "t":
                    out.append("\t")
                elif nxt == "r":
                    out.append("\r")
                else:
                    out.append(nxt)
            else:
                out.append(ch)
        return "".join(out)

    def _parse_plain_scalar(self) -> Any:
        start = self.i
        while self.i < self.n and self.text[self.i] not in ",]} \t\r\n":
            self.i += 1
        token = self.text[start : self.i].strip()
        if token in ("null", "Null", "NULL", "~"):
            return None
        if token in ("true", "True", "TRUE"):
            return True
        if token in ("false", "False", "FALSE"):
            return False
        if is_number_like(token):
            try:
                if "." in token or "e" in token or "E" in token:
                    return float(token)
                return int(token)
            except Exception:
                pass
        return token

    def _parse_sequence(self) -> List[Any]:
        self._next()  # skip '['
        arr: List[Any] = []
        while True:
            self._skip_whitespace()
            if self._peek() == "]":
                self._next()
                break
            val = self._parse_value()
            arr.append(val)
            self._skip_whitespace()
            ch = self._peek()
            if ch == ",":
                self._next()
                continue
            if ch == "]":
                self._next()
                break
            continue
        return arr

    def _parse_mapping(self) -> "OrderedDict[str, Any]":
        self._next()  # skip '{'
        mapping: "OrderedDict[str, Any]" = OrderedDict()
        while True:
            self._skip_whitespace()
            if self._peek() == "}":
                self._next()
                break
            ch = self._peek()
            if ch in ("'", '"'):
                key = self._parse_quoted()
            elif ch in ("{", "["):
                val = self._parse_value()
                key = str(val)
            else:
                start = self.i
                while self.i < self.n and self.text[self.i] != ":":
                    if self.text[self.i] in ",]}":
                        raise YAMLError("Flow mapping key without ':'")
                    self.i += 1
                key = self.text[start : self.i].strip()
            self._skip_whitespace()
            if self._peek() != ":":
                raise YAMLError("Missing ':' in flow mapping")
            self._next()
            self._skip_whitespace()
            val = self._parse_value()
            mapping[key] = val
            self._skip_whitespace()
            ch = self._peek()
            if ch == ",":
                self._next()
                continue
            if ch == "}":
                self._next()
                break
            continue
        return mapping
