"""Tests for the results contract.

These are not incidental unit tests. Each one pins a rule that some claim in
the report depends on, and the file is the executable form of the results
contract.
"""

from __future__ import annotations

import pytest

from cslr_cost.harness.schema import (
    EnergyError,
    EnumError,
    MissingFieldError,
    RegimeConsistencyError,
    ResampledWerError,
    ResultRow,
    SchemaError,
    config_hash,
    validate_row,
)
from cslr_cost.harness.writer import JsonlWriter, load_rows


# ---------------------------------------------------------------------------
# Fixtures: the two archetypal rows
# ---------------------------------------------------------------------------

def cost_row(**over) -> ResultRow:
    """A resolution-ladder cost row: resampled, therefore WER-free."""
    kw = dict(
        stage="pose",
        regime="streaming_simulated",
        measurement_mode="isolated",
        batch_size=1,
        paced_fps=25,
        input_resolution="720p",
        resampled=True,
        source_resolution="210x260",
        device="rtx3060",
        joules_per_frame=0.41,
        energy_backend="nvml",
    )
    kw.update(over)
    return ResultRow(**kw)


def accuracy_row(**over) -> ResultRow:
    """A native-resolution accuracy row: WER is legitimate here."""
    kw = dict(
        stage="end_to_end",
        regime="batch",
        measurement_mode="composed",
        batch_size=16,
        input_resolution="210x260",
        resampled=False,
        corpus="phoenix-2014",
        split="test",
        wer=21.4,
        del_=8.1,
        sub=9.7,
        ins=3.6,
        joules_per_frame=None,
        energy_unavailable_reason="accuracy-only run; no power rail attached",
    )
    kw.update(over)
    return ResultRow(**kw)


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------

def test_round_trip_preserves_every_field():
    row = cost_row()
    back = ResultRow.from_dict(row.to_dict())
    assert back.to_dict() == row.to_dict()


def test_del_keyword_maps_to_json_del():
    """`del` is a Python reserved word; it must still be `del` in the JSON."""
    row = accuracy_row()
    d = row.to_dict()
    assert "del" in d and "del_" not in d
    assert d["del"] == 8.1
    assert ResultRow.from_dict(d).del_ == 8.1


def test_unknown_field_is_rejected():
    d = cost_row().to_dict()
    d["joules_per_furlong"] = 1.0
    with pytest.raises(SchemaError, match="unknown field"):
        ResultRow.from_dict(d)


# ---------------------------------------------------------------------------
# The resampled-WER interlock
# ---------------------------------------------------------------------------

def test_resampled_row_may_not_carry_wer():
    with pytest.raises(ResampledWerError, match="indefensible"):
        cost_row(wer=18.7).validate()


@pytest.mark.parametrize("field_name", ["wer", "del_", "sub", "ins"])
def test_every_wer_component_is_blocked_on_resampled(field_name):
    with pytest.raises(ResampledWerError):
        cost_row(**{field_name: 3.0}).validate()


def test_resampled_row_with_null_wer_is_fine():
    cost_row().validate()  # must not raise


def test_native_row_may_carry_wer():
    accuracy_row().validate()  # must not raise


def test_resampled_row_self_documents_its_null():
    """The null should explain itself in the results file, not look like an omission."""
    assert "resampled" in cost_row()._wer_note


# ---------------------------------------------------------------------------
# Regime consistency  (P7 rule 1 — a common reporting error)
# ---------------------------------------------------------------------------

def test_streaming_forbids_batching():
    with pytest.raises(RegimeConsistencyError, match="batch_size=1"):
        cost_row(batch_size=32).validate()


def test_streaming_requires_declared_pacing():
    with pytest.raises(RegimeConsistencyError, match="paced_fps"):
        cost_row(paced_fps=None).validate()


def test_batch_regime_must_not_be_paced():
    with pytest.raises(RegimeConsistencyError, match="unpaced"):
        accuracy_row(paced_fps=25).validate()


# ---------------------------------------------------------------------------
# Energy honesty  (P2)
# ---------------------------------------------------------------------------

def test_null_energy_requires_a_stated_reason():
    with pytest.raises(EnergyError, match="energy_unavailable_reason"):
        cost_row(joules_per_frame=None, energy_backend=None).validate()


def test_null_energy_with_reason_is_accepted():
    cost_row(
        joules_per_frame=None,
        energy_backend="unavailable",
        energy_unavailable_reason="RAPL not readable as non-root on dev box",
    ).validate()


def test_zero_energy_is_never_a_measurement():
    with pytest.raises(EnergyError, match="never a real measurement"):
        cost_row(joules_per_frame=0.0).validate()


def test_energy_number_must_name_its_rail():
    with pytest.raises(EnergyError, match="name the rail"):
        cost_row(joules_per_frame=0.4, energy_backend=None).validate()


# ---------------------------------------------------------------------------
# Vocabularies and required fields
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "field_name,bad",
    [("stage", "inference"), ("regime", "realtime"), ("measurement_mode", "estimated")],
)
def test_controlled_vocabularies(field_name, bad):
    with pytest.raises(EnumError):
        cost_row(**{field_name: bad}).validate()


def test_missing_non_negotiable_field_is_refused():
    d = cost_row().to_dict()
    del d["regime"]
    with pytest.raises(MissingFieldError, match="non-negotiable"):
        validate_row(d)


def test_non_negotiable_fields_present_but_null_is_the_required_shape():
    """`del: null` must be present, not absent — absence and null differ."""
    d = cost_row().to_dict()
    assert d["del"] is None and "del" in d


def test_p95_cannot_be_faster_than_median():
    with pytest.raises(SchemaError, match="tail cannot"):
        cost_row(latency_ms_median=20.0, latency_ms_p95=12.0).validate()


# ---------------------------------------------------------------------------
# config_hash  (feeds the extraction cache)
# ---------------------------------------------------------------------------

def test_same_configuration_hashes_identically():
    assert config_hash(cost_row().to_dict()) == config_hash(cost_row().to_dict())


def test_configuration_change_changes_the_hash():
    a = config_hash(cost_row().to_dict())
    b = config_hash(cost_row(input_resolution="1080p").to_dict())
    assert a != b


@pytest.mark.parametrize("field_name", ["frontend_runtime", "head_runtime"])
def test_runtime_is_a_controlled_vocabulary(field_name):
    cost_row(**{field_name: "tensorrt"}).validate()
    with pytest.raises(EnumError):
        cost_row(**{field_name: "trt"}).validate()


def test_runtime_and_trt_version_are_configuration():
    """The same weights under PyTorch and under an engine are two measurements."""
    base = config_hash(cost_row(frontend_runtime="pytorch").to_dict())
    assert base != config_hash(cost_row(frontend_runtime="tensorrt").to_dict())
    assert config_hash(cost_row(trt_version="10.3.0").to_dict()) != config_hash(
        cost_row(trt_version="8.6.2").to_dict()
    )


def test_engine_hash_is_provenance_not_configuration():
    """Two builds of one config may pick different tactics; they are still one config."""
    a = config_hash(cost_row(frontend_engine="aaaa").to_dict())
    b = config_hash(cost_row(frontend_engine="bbbb").to_dict())
    assert a == b


def test_measurement_outcome_does_not_change_the_hash():
    """Two runs of one configuration must share a hash even with different numbers."""
    a = config_hash(cost_row(joules_per_frame=0.41).to_dict())
    b = config_hash(cost_row(joules_per_frame=0.55).to_dict())
    assert a == b


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

def test_writer_appends_and_reads_back(tmp_path):
    path = tmp_path / "results.jsonl"
    with JsonlWriter(path) as w:
        w.write(cost_row())
        w.write(accuracy_row())
    with JsonlWriter(path) as w:
        w.write(cost_row(input_resolution="1080p"))

    rows = load_rows(path, validate=True)
    assert len(rows) == 3
    assert [r["input_resolution"] for r in rows] == ["720p", "210x260", "1080p"]


def test_writer_refuses_an_invalid_row(tmp_path):
    path = tmp_path / "results.jsonl"
    with JsonlWriter(path) as w:
        with pytest.raises(ResampledWerError):
            w.write(cost_row(wer=12.0))
    assert not path.exists() or path.read_text() == ""


# ---------------------------------------------------------------------------
# Which weights, and which cost regime
# ---------------------------------------------------------------------------

def test_frontend_checkpoint_and_input_are_carried():
    """A model NAME is not an identity. `rtmpose-m` ships as a COCO-WholeBody
    checkpoint and as a COCO+UBody one — 58.2 published against 0.631 measured
    — and a row naming only the family cannot say which was measured."""
    row = cost_row(
        frontend_model="rtmpose-m",
        frontend_checkpoint="rtmpose-m_8xb64-270e_coco-ubody-wholebody-256x192",
        frontend_input="256x192",
    ).to_dict()

    assert row["frontend_checkpoint"].endswith("coco-ubody-wholebody-256x192")
    assert row["frontend_input"] == "256x192"


def test_the_new_fields_sit_with_the_other_model_configuration_fields():
    """Field order is the schema's readability contract, and these belong next
    to frontend_model rather than appended at the end."""
    from cslr_cost.harness.schema import CONFIG_FIELDS

    order = list(CONFIG_FIELDS)
    assert order.index("frontend_model") < order.index("frontend_checkpoint")
    assert order.index("frontend_checkpoint") < order.index("frontend_precision")
    assert order.index("frontend_input") < order.index("frontend_precision")
