"""fixtures.py — materialised stage-boundary inputs.

P1 requires each stage to be measured **alone**, in a tight loop, for >=60s at
steady state. But a stage cannot run alone without its input, and that is where
per-stage energy accounting quietly goes wrong: an "isolated pose" measurement
that decodes its own frames is really measuring decode + detect + crop + pose.
The isolated numbers then exceed the composed total, the reconciliation is
forced, and the residual — the one number P1 insists you report honestly —
becomes meaningless.

So isolation is done in two phases:

    1. CAPTURE   run the composed pipeline once over a clip, recording every
                 boundary artifact (frames, boxes, crops, keypoints, logits).
    2. REPLAY    run each stage alone against its captured input, from memory.

The captured artifact is the *real* thing the stage would have received, at the
real shape and dtype, so the isolated measurement is faithful without paying
upstream cost.

Fixtures can be persisted so a long sweep does not re-run capture per
configuration — but note the cache key must include everything upstream of the
boundary. A fixture captured at 360p is not a valid input for a 1080p pose
measurement, and silently reusing it would produce a cost curve that is flat
because it never actually varied the input.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Optional, Union

from cslr_cost.harness.stages.base import BOUNDARIES, Pipeline, Stage


class MissingFixtureError(KeyError):
    """A stage was asked to run in isolation without its input captured."""


class FixtureStore:
    """Boundary artifacts captured from one composed pass.

    Keyed by boundary name (`frames`, `crops`, `keypoints`, ...), so a stage
    asks for `store[stage.input_key]` and gets exactly what the composed run
    handed it.
    """

    def __init__(
        self,
        artifacts: Optional[Mapping[str, Any]] = None,
        *,
        provenance: Optional[dict[str, Any]] = None,
    ) -> None:
        self._artifacts: dict[str, Any] = dict(artifacts or {})
        #: What produced these fixtures — clip id, resolution, stage configs.
        #: Carried so a persisted fixture cannot be silently reused against a
        #: configuration it does not correspond to.
        self.provenance: dict[str, Any] = dict(provenance or {})

    # -- capture -----------------------------------------------------------

    @classmethod
    def capture(
        cls,
        pipeline: Pipeline,
        source: Any,
        n_frames: int,
        *,
        provenance: Optional[dict[str, Any]] = None,
    ) -> "FixtureStore":
        """Run `pipeline` once over `source`, recording every boundary.

        Per-frame boundaries accumulate one entry per frame; clip-level
        boundaries store the single assembled object.

        This pass is *not* a measurement — it exists to produce inputs. Time it
        if you like, but do not report it: it runs cold, before warm-up.
        """
        record: dict[str, Any] = {}
        pipeline.run(source, n_frames, record=record)
        prov = {"n_frames": n_frames, **(provenance or {})}
        return cls(record, provenance=prov)

    # -- access ------------------------------------------------------------

    def __getitem__(self, key: str) -> Any:
        try:
            return self._artifacts[key]
        except KeyError:
            raise MissingFixtureError(
                f"no fixture captured for boundary '{key}'. Available: "
                f"{sorted(self._artifacts)}. Run FixtureStore.capture() over a "
                "pipeline that includes the stage producing it — measuring a "
                "stage in isolation without its real input is how per-stage "
                "energy accounting silently becomes wrong."
            ) from None

    def __contains__(self, key: str) -> bool:
        return key in self._artifacts

    def get(self, key: str, default: Any = None) -> Any:
        return self._artifacts.get(key, default)

    def keys(self) -> Iterable[str]:
        return self._artifacts.keys()

    def units_for(self, stage: Stage) -> list[Any]:
        """Every captured unit this stage could consume.

        A per-frame stage sees one entry per frame; a per-clip stage sees a
        single-element list holding the assembled sequence.
        """
        artifact = self[stage.input_key]
        if stage.granularity == "frame":
            if not isinstance(artifact, list):
                raise TypeError(
                    f"boundary '{stage.input_key}' feeds the per-frame stage "
                    f"'{stage.name}' but was captured as {type(artifact).__name__}, "
                    "not a list of per-frame units"
                )
            return artifact
        return [artifact]

    def input_for(self, stage: Stage, i: int = 0) -> Any:
        """One unit for this stage — the `i`-th, cycling.

        Isolation loops call this with an incrementing index so a >=60s run
        cycles through the clip's real frames rather than reprocessing one
        buffer thousands of times. Hammering a single buffer measures an
        artificially cache-hot path and understates the stage, which then shows
        up as unexplained residual against the composed run.
        """
        units = self.units_for(stage)
        if not units:
            raise MissingFixtureError(
                f"boundary '{stage.input_key}' was captured empty — no units to "
                f"replay for stage '{stage.name}'"
            )
        return units[i % len(units)]

    def covers(self, stages: Iterable[Stage]) -> bool:
        """True if every stage's input is present."""
        return all(s.input_key in self._artifacts for s in stages)

    def missing_for(self, stages: Iterable[Stage]) -> list[str]:
        return [s.input_key for s in stages if s.input_key not in self._artifacts]

    # -- persistence --------------------------------------------------------

    def save(self, path: Union[str, Path]) -> Path:
        """Persist to disk. Pickle, because boundary artifacts are a mix of
        numpy arrays, torch tensors and plain Python — and this is a local
        cache, never an interchange format."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as fh:
            pickle.dump(
                {"artifacts": self._artifacts, "provenance": self.provenance},
                fh,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        return p

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        *,
        expect: Optional[Mapping[str, Any]] = None,
    ) -> "FixtureStore":
        """Load fixtures, optionally asserting they match the expected config.

        `expect` is checked against the stored provenance. Pass the fields that
        determine the artifact — resolution, clip, front-end model — so a
        fixture captured under one configuration cannot be reused under
        another. A cost curve built from mismatched fixtures looks smooth and
        means nothing.
        """
        with Path(path).open("rb") as fh:
            blob = pickle.load(fh)
        store = cls(blob["artifacts"], provenance=blob.get("provenance", {}))
        if expect:
            mismatched = {
                k: (v, store.provenance.get(k))
                for k, v in expect.items()
                if store.provenance.get(k) != v
            }
            if mismatched:
                raise ValueError(
                    f"fixture at {path} does not match the requested "
                    f"configuration: {mismatched} (expected, stored). Re-capture "
                    "rather than reusing it."
                )
        return store

    def __repr__(self) -> str:
        return f"<FixtureStore {sorted(self._artifacts)}>"


def ordered_boundaries(present: Iterable[str]) -> list[str]:
    """Boundary names sorted into pipeline order, for readable reporting."""
    index = {name: i for i, name in enumerate(BOUNDARIES)}
    return sorted(present, key=lambda k: index.get(k, len(BOUNDARIES)))
