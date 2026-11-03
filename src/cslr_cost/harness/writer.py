"""writer.py — append-only JSONL sink for measurement rows.

The results contract: "One JSONL, one row per measurement. Flat, verbose, append-only."

Append-only is a methodological choice, not a convenience. A results file you
can rewrite is a results file you can quietly correct after seeing the plot.
Rows are validated on the way in and never mutated afterwards; a superseded
measurement is superseded by appending a newer row, not by editing the old one.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Optional, Union

from cslr_cost.harness.schema import ResultRow, validate_row

RowLike = Union[ResultRow, Mapping[str, Any]]


class JsonlWriter:
    """Validating, append-only JSONL writer.

    Usage::

        with JsonlWriter("results/results.jsonl") as w:
            w.write(row)

    Every row is validated before it touches the file, so an invalid
    measurement fails at the call site — where the context to fix it still
    exists — rather than in Week 12 when the table is assembled.
    """

    def __init__(self, path: Union[str, Path], *, fsync: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fsync = fsync
        self._fh = None
        self._count = 0

    # -- lifecycle ---------------------------------------------------------

    def open(self) -> "JsonlWriter":
        if self._fh is None:
            self._fh = self.path.open("a", encoding="utf-8")
        return self

    def close(self) -> None:
        if self._fh is not None:
            self._fh.flush()
            if self._fsync:
                os.fsync(self._fh.fileno())
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "JsonlWriter":
        return self.open()

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- writing -----------------------------------------------------------

    def write(self, row: RowLike) -> dict[str, Any]:
        """Validate and append one row. Returns the dict that was written."""
        payload = row.to_dict() if isinstance(row, ResultRow) else dict(row)
        validate_row(payload)

        if self._fh is None:
            self.open()
        self._fh.write(json.dumps(payload, separators=(",", ":"), default=str) + "\n")
        self._fh.flush()
        if self._fsync:
            os.fsync(self._fh.fileno())
        self._count += 1
        return payload

    def write_all(self, rows: Iterable[RowLike]) -> int:
        """Write many rows. Validation is per-row, so a bad row stops the batch
        at that row — earlier rows are already durably appended."""
        n = 0
        for row in rows:
            self.write(row)
            n += 1
        return n

    @property
    def count(self) -> int:
        """Rows written by this writer instance (not the file's total)."""
        return self._count


def read_jsonl(path: Union[str, Path], *, validate: bool = False) -> Iterator[dict[str, Any]]:
    """Stream rows back out of a results file.

    `validate=True` re-checks every row against the current contract — useful
    after a schema change, to find rows written under the old rules.
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{p}:{lineno}: malformed JSON — {e}") from e
            if validate:
                validate_row(row)
            yield row


def load_rows(path: Union[str, Path], *, validate: bool = False) -> list[dict[str, Any]]:
    """Eager `read_jsonl`."""
    return list(read_jsonl(path, validate=validate))
