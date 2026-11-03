"""
wer_guard.py — safety assertion: no WER on resampled input.

THE most important piece of the resolution ladder. The ladder is valid for
COST measurement ONLY. Upscaling PHOENIX-2014 (native 210x260) to 360/480/720/
1080p adds zero information, so any WER measured on a resampled clip is
scientifically indefensible.

This module refuses to let a WER value coexist with resampled==true in a result
record. Import `emit_result` (or `assert_no_wer_on_resampled`) everywhere the
harness writes a result, so it is structurally impossible to publish an
upscaled-WER number by accident.
"""

from __future__ import annotations
from typing import Any, Mapping


class ResampledWerError(AssertionError):
    """Raised when a result tries to attach a WER to a resampled input."""
    pass


# Keys that count as "a WER field" (be generous — catch aliases too).
_WER_KEYS = {"wer", "word_error_rate", "wer_total", "del", "sub", "ins",
             "wer_del", "wer_sub", "wer_ins"}


def assert_no_wer_on_resampled(record: Mapping[str, Any]) -> None:
    """Raise ResampledWerError if `record` is resampled AND carries any WER field.

    A record is 'resampled' when record['resampled'] is truthy.
    """
    if not record.get("resampled", False):
        return  # native resolution: WER is allowed
    present = [k for k in record.keys() if k.lower() in _WER_KEYS]
    if present:
        raise ResampledWerError(
            "REFUSED to emit result: WER on a resampled input is indefensible "
            "(upscaling adds no information). Offending field(s): "
            f"{present}. resampled=True, source_resolution="
            f"{record.get('source_resolution', 'unknown')}. "
            "Accuracy must only ever be measured at native resolution."
        )


def emit_result(record: Mapping[str, Any], writer=None) -> Mapping[str, Any]:
    """Validate a result record, then emit it.

    - Always runs the resampled-WER guard first.
    - If `writer` is given (a callable, e.g. json line writer), it is called
      with the validated record. Otherwise the record is just returned.
    """
    assert_no_wer_on_resampled(record)
    if writer is not None:
        writer(record)
    return record


if __name__ == "__main__":
    # Self-test: the guard must pass native+WER and block resampled+WER.
    ok_native = {"clip": "01April_2010", "resampled": False, "wer": 21.2}
    ok_cost = {"clip": "01April_2010", "resampled": True,
               "source_resolution": "210x260", "resolution": "720p",
               "latency_ms": 33.4, "joules": 1.9}  # cost only, no WER
    bad = {"clip": "01April_2010", "resampled": True,
           "source_resolution": "210x260", "wer": 18.7}

    assert_no_wer_on_resampled(ok_native); print("PASS: native + WER allowed")
    assert_no_wer_on_resampled(ok_cost);   print("PASS: resampled cost-only allowed")
    try:
        assert_no_wer_on_resampled(bad)
        print("FAIL: guard did not block resampled+WER")
    except ResampledWerError as e:
        print("PASS: guard blocked resampled+WER ->", str(e)[:60], "...")