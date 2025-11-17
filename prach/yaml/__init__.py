from typing import Any, List

from .error import YAMLError


def load(text: str, alias_mode: str = "reference") -> Any:
    """
    Parse YAML text
    
    `alias_mode`: 'reference' or 'deepcopy'.

    If multiple documents present returns list;
    else returns single document.
    """

    from .document_parser import YAMLDocumentParser

    def split_documents(text: str) -> List[str]:
        docs: List[str] = []
        cur_lines: List[str] = []
        lines = text.splitlines()
        saw_doc_start = False
        for raw in lines:
            stripped = raw.strip()
            if stripped.startswith("---") and (
                stripped == "---" or stripped.startswith("--- ")
            ):
                if saw_doc_start or cur_lines:
                    docs.append("\n".join(cur_lines).rstrip("\n"))
                cur_lines = []
                saw_doc_start = True
                if len(stripped) > 3 and stripped[3] == " ":
                    rest = raw[raw.find("---") + 4 :]
                    cur_lines.append(rest)
                continue
            if stripped == "...":
                if cur_lines or saw_doc_start:
                    docs.append("\n".join(cur_lines).rstrip("\n"))
                cur_lines = []
                saw_doc_start = False
                continue
            cur_lines.append(raw)
        if cur_lines or not docs:
            docs.append("\n".join(cur_lines).rstrip("\n"))
        docs = [d for d in docs if d is not None and d.strip() != ""]
        return docs

    parts = split_documents(text)
    results: List[Any] = []
    for part in parts:
        parser = YAMLDocumentParser(alias_mode=alias_mode)
        doc = parser.load(part)
        results.append(doc)
    if len(results) == 1:
        return results[0]
    return results
