# Measurement protocol

The rules every number from this harness follows. Code comments refer to them as
P1–P7. Each rule exists because breaking it produces a number that looks fine
and is wrong.

## P1 — Per-stage energy comes from isolation, not from one pass

Board power monitors refresh every 10–400 ms (the Jetson Orin's INA3221: every
160–400 ms), while pipeline stages take microseconds to a few hundred
milliseconds per call. Energy cannot be attributed to stages within one pass.

1. **Record.** One instrumented pass of the pipeline captures every stage's real
   input at its boundary (`harness/fixtures.py`).
2. **Isolate.** Each stage replays its recorded inputs alone, in a loop, for at
   least 60 s after a warm-up outside the window. The replay cycles through the
   clip's frames rather than repeating one buffer, so caches are not artificially
   warm. Energy over the window divided by the units processed is the stage's
   energy per unit.
3. **Compose.** The whole pipeline is measured the same way.
4. **Reconcile.** The sum of the isolated stages is compared with the composed
   total, and the difference is reported as a residual, never spread over the
   stages.

**Time is reconciled in situ.** Isolation serialises work that the pipeline
overlaps (on the Orin, video decode alone costs about twice its in-pipeline
time, because its worker threads wait on each other). So each stage is also
timed *inside* the composed loop, and those per-stage times must sum to the
composed clip time within **5%** (`harness/reconcile.py`). A run that misses the
gate keeps its end-to-end figures, but its stage shares are marked indicative.

Per-clip stages (head, CTC) are normalised to energy per frame by dividing by the
clip's frame count. Without that, the head's share is inflated by the number of
frames.

Every row says `measurement_mode: isolated | composed`, so the two are never
mixed silently.

## P2 — Idle is subtracted, and stated

Energy is reported twice: **total** (what a battery pays) and **over idle** (what
the workload adds). Over-idle energy is a difference, and differences amplify
errors, so idle is measured carefully:

- the device settles for 20 s after load, then idle is measured for 10 s,
  before *and* after the run, at the same clocks;
- if the two idle readings differ by more than 20%, the run's over-idle figures
  are withheld;
- comparisons between models use idle from the same session, because idle is not
  a constant (attaching a USB disk raised one Orin's floor by 1 W).

## P3 — Sustained, not peak

After warming to a temperature plateau, the pipeline runs unpaced for 10 minutes
and reports the throughput it sustains and the end temperature. If sustained
throughput falls far below peak, the device is throttling, and that is reported
as a finding.

## P4 — Fix the clocks

On a Jetson, set a power mode and pin the clocks (`sudo jetson_clocks`). The
harness refuses to run on unpinned Jetson clocks unless `--allow-unpinned` is
passed, and then marks every row. Desktop clocks usually cannot be pinned; their
energy figures are labelled indicative.

## P5 — Median and p95, never the mean

A mean hides the tail, and a live user experiences the tail. Latency is reported
as median and 95th percentile over repetitions.

## P6 — Cost is a curve over input resolution

Decode and resize cost depend on the camera's resolution, so cost can be measured
on a resolution ladder (360p to 1080p) built from the same clip with one fixed
codec and bitrate (`scripts/resample_ladder.sh`). The pose stage sees a
fixed-size crop and does not depend on it. **Accuracy is never measured on
resampled input**: rows from the ladder are marked `resampled: true`, and the
schema refuses a WER on them.

## P7 — Two regimes, and the rules that make "live" honest

**Batch.** Clips processed back to back at batch 1 (plus an optional batch-N
row). Time per frame is the clip time divided by its frame count.

**Live (streaming simulation).** Three rules:

1. **Batch size 1.** No "effective batch", no padding several clips together.
   Quoting a batched throughput figure as streaming fps is a common reporting
   error, and the schema rejects it.
2. **Pace the frames.** One frame every 40 ms (25 fps). The gap is *slept*, not
   spun, because a spinning wait draws power in the interval that should be idle.
   No frame is dropped: a frame that misses its slot is an overrun, and lateness
   accumulates.
3. **No undeclared lookahead.** A bidirectional head reads the whole clip, so
   its first-gloss latency is bounded below by the clip's duration. It is
   reported that way, never as "time to decode after the clip ended".

What the simulation cannot capture, and should be stated: camera and ISP power,
and the thermal load of a live camera.

## Contention gate

Shared machines are common, and no power rail can separate another user's job
from yours. Every 2 s the harness lists other processes holding GPU memory (NVML
on desktops; the `nvmap` driver's per-client accounting on Jetson) and computes
other processes' CPU time from `/proc/stat`. A run is judged **per measurement
window**, not on average: another process on the GPU, or other processes' CPU
above 10% in any window longer than 20 s, marks the run `SHARED MACHINE`, and
its energy is not to be reported. If the GPU check cannot be made, the verdict is
`PARTIALLY CHECKED`, never `clean`.

## Memory

Peak device memory of the process at batch 1, sampled once per second: the
`nvmap` footprint on Jetson, NVML's per-process figure on desktops. Framework
allocator statistics are not used, because they cannot see TensorRT's memory.
On a Jetson, CPU and GPU share one pool, so device memory and RAM are not added.

## Accuracy

WER on PHOENIX-2014 dev and test, always with its deletion, substitution and
insertion rates, from the output stream a deployment would run (chosen before
evaluation, not the best of several streams on the evaluation split). Two
conventions, never mixed in one column:

- **markers removed:** `__ON__`/`__OFF__` dropped from the references
  (`eval/wer.py`, using MSKA's scorer);
- **official:** the dataset's `evaluatePhoenix2014.sh` with NIST `sclite`
  (`eval/official_phoenix.py`), which keeps them.

## Model size

Parameters and multiply-accumulates are reported because papers do, but they are
descriptive, not cost. MACs are counted from the ONNX graphs; one MAC is two
FLOPs.
