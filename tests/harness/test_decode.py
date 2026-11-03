"""decode stage: source handling, the rewind counter, and the accel contract."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from cslr_cost.harness.stages.decode import (  # noqa: E402
    ImageSequenceSource,
    VideoDecodeStage,
    VideoSource,
    open_video,
)

W, H, N = 64, 48, 6


def _write_frames(d: Path) -> list[Path]:
    """N stills, each a distinct constant colour, so order is checkable."""
    paths = []
    for i in range(N):
        img = np.full((H, W, 3), 0, dtype=np.uint8)
        img[:, :, 0] = i * 20          # vary the BLUE channel: BGR, not RGB
        p = d / f"frame_{i:03d}.png"
        cv2.imwrite(str(p), img)
        paths.append(p)
    return paths


def _write_video(p: Path) -> None:
    writer = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (W, H))
    for i in range(N):
        img = np.full((H, W, 3), 0, dtype=np.uint8)
        img[:, :, 0] = i * 20
        writer.write(img)
    writer.release()


# -- image sequences --------------------------------------------------------


def test_directory_opens_as_a_still_sequence_not_video(tmp_path):
    _write_frames(tmp_path)
    src = open_video(tmp_path)
    assert isinstance(src, ImageSequenceSource)
    # The codec label is what stops a PNG directory being reported in the same
    # row as h264 decode; they are different workloads at different cost.
    assert src.codec == "png-sequence"
    assert (src.width, src.height) == (W, H)
    assert len(src) == N


def test_frames_come_back_in_sorted_order(tmp_path):
    _write_frames(tmp_path)
    src = open_video(tmp_path)
    blues = [int(src.read()[0, 0, 0]) for _ in range(N)]
    assert blues == [i * 20 for i in range(N)]


def test_reaching_the_end_rewinds_and_counts_it(tmp_path):
    """A >=60s isolation window wraps the clip hundreds of times. Each wrap
    costs a seek; the counter is what makes that overhead auditable instead of
    assumed negligible."""
    _write_frames(tmp_path)
    src = open_video(tmp_path)
    for _ in range(N):
        src.read()
    assert src.rewinds == 0
    first_after_wrap = src.read()
    assert src.rewinds == 1
    assert int(first_after_wrap[0, 0, 0]) == 0


def test_empty_directory_is_an_error_not_an_empty_clip(tmp_path):
    with pytest.raises(ValueError, match="no images"):
        open_video(tmp_path)


# -- video ------------------------------------------------------------------


def test_video_opens_with_metadata(tmp_path):
    p = tmp_path / "clip.mp4"
    _write_video(p)
    src = open_video(p, accel="software")
    assert isinstance(src, VideoSource) and not isinstance(src, ImageSequenceSource)
    assert (src.width, src.height) == (W, H)
    assert src.codec == "h264"
    frame = src.read()
    assert frame.shape == (H, W, 3) and frame.dtype == np.uint8
    src.close()


def test_video_rewinds_at_end_of_stream(tmp_path):
    p = tmp_path / "clip.mp4"
    _write_video(p)
    src = open_video(p, accel="software")
    for _ in range(N + 2):
        src.read()
    assert src.rewinds >= 1
    src.close()


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        open_video(tmp_path / "nope.mp4")


def test_unknown_accel_is_rejected(tmp_path):
    p = tmp_path / "clip.mp4"
    _write_video(p)
    with pytest.raises(ValueError, match="accel must be one of"):
        open_video(p, accel="nvdec")


def test_hardware_decode_is_never_silently_downgraded(tmp_path):
    """The whole point of decode being a measured axis is comparing hardware
    against software. pip's OpenCV wheels have no CUVID, so asking for hardware
    and getting libavcodec must fail loudly rather than produce a software
    number under a hardware label."""
    p = tmp_path / "clip.mp4"
    _write_video(p)
    try:
        src = open_video(p, accel="hardware")
    except RuntimeError as exc:
        assert "granted" in str(exc)
    else:
        assert src.accel != "none"
        src.close()


def test_reported_accel_is_what_was_granted_not_what_was_asked(tmp_path):
    p = tmp_path / "clip.mp4"
    _write_video(p)
    src = open_video(p, accel="software")
    assert src.accel_requested == "software"
    assert src.describe()["decode_accel"] == src.accel
    src.close()


# -- the stage --------------------------------------------------------------


def test_stage_pulls_successive_frames_from_the_source(tmp_path):
    """`Pipeline.run` hands the same source object to the first stage once per
    frame and expects a different frame back each time, so the decoder state
    belongs to the source, not the stage."""
    _write_frames(tmp_path)
    src = open_video(tmp_path)
    stage = VideoDecodeStage()
    stage.setup()
    blues = [int(stage.run(src)[0, 0, 0]) for _ in range(3)]
    assert blues == [0, 20, 40]


def test_stage_declares_its_boundary_and_granularity():
    stage = VideoDecodeStage()
    assert (stage.name, stage.input_key, stage.output_key) == ("decode", "video", "frames")
    assert stage.granularity == "frame"
    assert stage.uses_cuda is False
    assert stage.config()["decode_path"] == "cpu"


def test_rewind_actually_seeks_the_decoder(tmp_path):
    """Assigning `position = 0` sets a counter; the decoder's real position
    lives in the VideoCapture. Setting the counter alone leaves the two
    disagreeing — the next read returns whatever came next, not frame 0."""
    p = tmp_path / "clip.mp4"
    _write_video(p)
    src = open_video(p, accel="software")

    first = int(src.read()[0, 0, 0])
    src.read()
    src.read()

    src.position = 0                      # the counter only
    not_rewound = int(src.read()[0, 0, 0])
    assert not_rewound != first, "this assignment is not expected to seek"

    src.rewind()                          # the decoder
    assert int(src.read()[0, 0, 0]) == first
    src.close()


def test_rewind_works_on_a_still_sequence_too(tmp_path):
    """Callers must not have to know which kind of source they hold."""
    _write_frames(tmp_path)
    src = open_video(tmp_path)
    src.read()
    src.read()
    src.rewind()
    assert src.position == 0
    assert int(src.read()[0, 0, 0]) == 0
