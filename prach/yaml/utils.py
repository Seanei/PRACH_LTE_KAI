from typing import List, Tuple, Optional


def lstrip_spaces(s: str) -> Tuple[int, str]:
    i = 0
    n = len(s)
    while i < n and s[i] == " ":
        i += 1
    return i, s[i:]


def strip_comment(line: str) -> str:
    in_sq = False
    in_dq = False
    escaped = False
    for i, ch in enumerate(line):
        if ch == "\\" and not escaped:
            escaped = True
            continue
        if ch == "'" and not in_dq and not escaped:
            in_sq = not in_sq
        elif ch == '"' and not in_sq and not escaped:
            in_dq = not in_dq
        elif ch == "#" and not in_sq and not in_dq:
            return line[:i].rstrip()
        escaped = False
    return line.rstrip()


def unquote(s: str) -> str:
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        if s[0] == "'":
            return s[1:-1].replace("''", "'")
        else:
            out = []
            body = s[1:-1]
            i = 0
            while i < len(body):
                if body[i] == "\\" and i + 1 < len(body):
                    nxt = body[i + 1]
                    if nxt == "n":
                        out.append("\n")
                    elif nxt == "t":
                        out.append("\t")
                    elif nxt == "r":
                        out.append("\r")
                    else:
                        out.append(nxt)
                    i += 2
                    continue
                out.append(body[i])
                i += 1
            return "".join(out)
    return s


def is_number_like(s: str) -> Optional[bool]:
    """Checks if token is number like

    Considers number as if it is scientific or
    simple notation of decimals and signed or unsigned integers

    Returns True on float, False on int or None on neither"""
    if not s:
        return None
    i = 0
    n = len(s)
    if s[0] in "+-":
        i += 1
        if i >= n:
            return None
    digits = 0
    dot = False
    exp = False
    while i < n:
        c = s[i]
        if c.isdigit():
            digits += 1
        elif c == ".":
            if dot or exp:
                return None
            dot = True
        elif c in "eE":
            if exp or digits == 0:
                return None
            exp = True
            digits = 0
            if i + 1 < n and s[i + 1] in "+-":
                i += 1
        else:
            return None
        i += 1
    return (dot or exp)
