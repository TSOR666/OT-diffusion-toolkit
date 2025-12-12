from pathlib import Path
import re


CONFLICT_PATTERNS = (
    re.compile(r"^<{7}(?: .*)?$"),
    re.compile(r"^={7}(?: .*)?$"),
    re.compile(r"^>{7}(?: .*)?$"),
)


def test_no_merge_conflict_markers() -> None:
    """Ensure repository sources are free of Git merge conflict markers."""

    root = Path(__file__).resolve().parent.parent
    for py_file in root.rglob("*.py"):
        text = py_file.read_text(encoding="utf-8", errors="ignore")
        for line in text.splitlines():
            for pattern in CONFLICT_PATTERNS:
                assert not pattern.match(line), (
                    f"Merge conflict marker '{pattern.pattern}' found in {py_file}"
                )
