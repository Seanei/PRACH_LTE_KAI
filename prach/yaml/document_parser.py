import copy
from typing import Any, List, Dict, Optional, Tuple
from collections import OrderedDict

from .flow_parser import FlowParser
from .utils import lstrip_spaces, strip_comment, unquote, is_number_like
from .error import YAMLError


class YAMLDocumentParser:
    """Parses small subset of YAML into Dict"""

    def __init__(self, alias_mode: str = "reference"):
        # (indent, stripped line)
        self.tokens: List[Tuple[Optional[int], str]] = []
        # current line id
        self.idx: int = 0
        self.anchors: Dict[str, Any] = {}
        if alias_mode not in ("reference", "deepcopy"):
            raise ValueError("alias_mode must be 'reference' or 'deepcopy'")
        self.alias_mode: str = alias_mode

    def load(self, text: str) -> Any:
        self._reset()
        self._preprocess(text)
        return self._parse_block(0)

    def _reset(self):
        self.tokens = []
        self.idx = 0
        self.anchors = {}

    def _preprocess(self, text: str):
        lines = text.splitlines()
        toks: List[Tuple[Optional[int], str]] = []
        for raw in lines:
            if raw.lstrip().startswith("%"):
                continue
            ln = strip_comment(raw).replace("\t", "    ")
            if ln.strip() == "":
                toks.append((None, ""))
            else:
                indent, content = lstrip_spaces(ln)
                toks.append((indent, content))
        self.tokens = toks
        self.idx = 0

    def _peek(self):
        if self.idx < len(self.tokens):
            return self.tokens[self.idx]
        return None

    def _next(self):
        token = self._peek()
        if token is not None:
            self.idx += 1
        return token

    def _parse_block(self, base_indent: int) -> Any:
        token: Optional[Tuple[Optional[int], str]] = None
        while True:
            token = self._peek()
            if token is None:
                return None
            # indent is None => empty line
            if token[0] is None:
                self._next()
                continue
            # indent differs, seems to be part of another block
            if token[0] < base_indent:
                return None
            break
        # returns early if we got None, so, it's safe
        indent, content = token
        if content.startswith("-"):
            return self._parse_sequence(base_indent)
        if ":" in content:
            return self._parse_mapping(base_indent)
        return self._parse_scalar_block(base_indent)

    def _parse_sequence(self, base_indent: int) -> List[Any]:
        arr: List[Any] = []
        while True:
            token = self._peek()
            if token is None:
                break
            indent, content = token
            if indent is None:
                self._next()
                continue
            if indent < base_indent:
                break
            if not content.startswith("-"):
                break
            self._next()
            after = content[1:].lstrip()
            if after == "":
                val = self._parse_block(indent + 1)
                arr.append(val)
                continue
            if after[0] in ("|", ">"):
                arr.append(self._parse_block_scalar_inline(indent, after))
                continue
            val = self._parse_value_inline(after)
            arr.append(val)
        return arr

    def _parse_mapping(self, base_indent: int) -> OrderedDict[str, Any]:
        mapping: OrderedDict[str, Any] = OrderedDict()
        while True:
            p = self._peek()
            if p is None:
                break
            indent, content = p
            if indent is None:
                self._next()
                continue
            if indent < base_indent:
                break
            if ":" not in content:
                break
            self._next()
            key, rest = self._split_key_value(content)
            if rest and rest[0] in ("|", ">"):
                val = self._parse_block_scalar_inline(indent, rest)
                self._assign_key(mapping, key, val)
                continue
            if rest != "":
                val = self._parse_value_inline(rest)
                self._assign_key(mapping, key, val)
                continue
            nxt = self._peek()
            if nxt is None or nxt[0] is None or nxt[0] <= indent:
                self._assign_key(mapping, key, None)
            else:
                child = self._parse_block(nxt[0])
                self._assign_key(mapping, key, child)
        return mapping

    def _split_key_value(self, content: str) -> Tuple[str, str]:
        pos = content.find(":")
        key = content[:pos].strip()
        val = content[pos + 1 :].lstrip()
        return unquote(key), val

    def _assign_key(self, mapping: OrderedDict[str, Any], key: str, val: Any):
        if key == "<<":
            if val is None:
                return
            if isinstance(val, dict):
                for k, v in val.items():
                    if k not in mapping:
                        mapping[k] = v
                return
            if isinstance(val, list):
                for item in val:
                    if isinstance(item, dict):
                        for k, v in item.items():
                            if k not in mapping:
                                mapping[k] = v
                return
            return
        mapping[key] = val

    def _parse_block_scalar_inline(self, header_indent: int, header: str) -> str:
        style = header[0]
        explicit_indent = None
        chomp = None
        j = 1
        if j < len(header) and header[j].isdigit():
            explicit_indent = int(header[j])
            j += 1
        if j < len(header) and header[j] in ("+", "-"):
            chomp = header[j]
        lines_raw: List[Tuple[Optional[int], str]] = []
        min_indent: Optional[int] = None
        while True:
            p = self._peek()
            if p is None:
                break
            indent, content = p
            if indent is None:
                self._next()
                lines_raw.append((None, ""))
                continue
            if indent <= header_indent:
                break
            self._next()
            lines_raw.append((indent, content))
            if indent is not None:
                if min_indent is None or indent < min_indent:
                    min_indent = indent
        if explicit_indent is not None:
            block_indent = header_indent + explicit_indent
        else:
            block_indent = min_indent if min_indent is not None else header_indent + 1
        stripped: List[str] = []
        for it in lines_raw:
            if it[0] is None:
                stripped.append("")
            else:
                indent, content = it
                if indent >= block_indent:
                    stripped.append(" " * (indent - block_indent) + content)
                else:
                    stripped.append(content)
        if style == "|":
            text = "\n".join(stripped)
        else:
            out = []
            buf: List[str] = []
            for ln in stripped:
                if ln == "":
                    if buf:
                        out.append(" ".join(buf))
                        buf = []
                    out.append("")
                else:
                    buf.append(ln)
            if buf:
                out.append(" ".join(buf))
            text = "\n".join(out)
        if chomp == "+":
            if not text.endswith("\n"):
                text += "\n"
        elif chomp == "-":
            text = text.rstrip("\n")
        else:
            if text.endswith("\n"):
                text = text[:-1]
        return text

    def _parse_scalar_block(self, base_indent: int) -> Any:
        parts: List[str] = []
        first_indent: Optional[int] = None
        while True:
            p = self._peek()
            if p is None:
                break
            indent, content = p
            if indent is None or indent < base_indent:
                break
            if first_indent is None:
                first_indent = indent
            if indent != first_indent:
                break
            self._next()
            parts.append(content)
        if not parts:
            return ""
        combined = "\n".join(parts)
        return self._parse_value_inline(combined)

    def _parse_value_inline(self, s: str) -> Any:
        s = s.strip()
        if not s:
            return ""
        if s[0] == "&":
            parts = s.split(" ", 1)
            name = parts[0][1:]
            if len(parts) == 1 or parts[1].strip() == "":
                # value after anchor in next block
                val = self._parse_block(self.tokens[self.idx - 1][0] + 1)
            else:
                val = self._parse_value_inline(parts[1])
            self.anchors[name] = val
            return val
        if s[0] == "*":
            name = s[1:].strip()
            if name not in self.anchors:
                raise YAMLError("Unknown alias *%s" % name)
            if self.alias_mode == "reference":
                return self.anchors[name]
            else:
                return copy.deepcopy(self.anchors[name])
        if s[0] == "!":
            parts = s.split(" ", 1)
            tag = parts[0]
            val = None
            if len(parts) == 2:
                val = self._parse_value_inline(parts[1])
            return (tag, val)
        if s[0] in ('"', "'"):
            return unquote(s)
        if s[0] in ("{", "["):
            parser = FlowParser(s, self.anchors, alias_mode=self.alias_mode)
            return parser.parse()
        if s in ("null", "Null", "NULL", "~"):
            return None
        if s in ("true", "True", "TRUE"):
            return True
        if s in ("false", "False", "FALSE"):
            return False
        number_like = is_number_like(s)
        if number_like is not None:
            if number_like:
                return float(s)
            else:
                return int(s)
        return s
