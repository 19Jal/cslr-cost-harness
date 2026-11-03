"""decode.py — video decode (stage 1).

The first stage is **video decode, not camera capture**, and it is a measured
axis rather than a footnote. Hardware decode (NVDEC / V4L2 / MediaCodec) and a
software libavcodec fallback can differ by ~2x in both latency and energy, and
neither appears anywhere in a model-level profile. A system that reports
"the model costs X ms/frame" while silently paying software decode is
mis-stating its own budget, which is the class of error a whole-pipeline account
exists to correct.

--------------------------------------------------------------------------
A SILENT FALLBACK IS THE FAILURE MODE
--------------------------------------------------------------------------
OpenCV accepts `CAP_PROP_HW_ACCELERATION` and then quietly ignores it when the
build has no CUVID/VAAPI support — which is the normal case for a pip wheel.
The capture still opens, the frames are still correct, and the measurement is
now software decode wearing a hardware label. So `open_video` **reads the
property back** and refuses to proceed if `hardware` was asked for and not
granted. `VideoSource.accel` records what was actually used, and it goes into
the results row; nothing infers it from what was requested.

--------------------------------------------------------------------------
TWO SOURCE KINDS, NEVER CONFLATED
--------------------------------------------------------------------------
PHOENIX-2014 ships as **PNG frame directories**, not video. The resolution
ladder ships as **h264 mp4**. Both are legitimate inputs and they
cost completely different amounts — PNG is zlib on an intra-only still, h264 is
motion-compensated inter-frame decode. Averaging them, or reporting one as the
other, would put a number in the decode row that corresponds to no deployable
system.

They are therefore separate source classes, and each stamps `codec` into the
row. The ladder mp4s are the decode benchmark; the PNG directories are the
accuracy path, where decode cost is an artifact of how the corpus was
distributed rather than a property of anything we would ship.

--------------------------------------------------------------------------
DECODE EMITS BGR, AND IT STAYS BGR ALL THE WAY INTO THE MODEL
--------------------------------------------------------------------------
OpenCV decodes to BGR. So does RTMPose expect it: its `PoseDataPreprocessor`
carries `bgr_to_rgb: True` and does the swap **on the GPU**, batched, together
with mean/std normalisation.

That settles a boundary question that would otherwise silently move
milliseconds between rows. A `cvtColor` on the CPU is ~0.1-0.3 ms/frame at
210x260 and several times that at 1080p, and whichever stage performs it pays
for it. Here no stage does: the conversion happens inside `pose`, on device,
where it is nearly free and where a real deployment would put it. Decode's
contract is therefore **raw BGR uint8, HxWx3, exactly as the codec produced
it**, and it survives unchanged through `crop`.

--------------------------------------------------------------------------
STATE LIVES IN THE SOURCE, NOT THE STAGE
--------------------------------------------------------------------------
`Pipeline.run` hands the same `source` object to the first stage once per
frame, and expects a different frame back each time. Decode is therefore
inherently stateful, and that state is the open decoder — so it lives in
`VideoSource`, not in the stage. This is what lets an isolation replay work at
all: P1 wants >=60s in a tight loop, and cycling one buffer would measure a
cache-hot memcpy instead of a codec. Reaching the end rewinds and counts the
rewind, so the seek overhead is auditable rather than hidden.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from cslr_cost.harness.stages.base import Stage

#: Extensions treated as still images when a source path is a directory.
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp")

#: OpenCV's accel name -> the schema's `decode_path` vocabulary
#: (`schema.DECODE_PATHS`). The schema constrains this field to a closed set on
#: purpose: "decode path" is a comparison axis, and a free-form string would
#: make `nvdec` and `opencv-ffmpeg/hardware` two incomparable rows describing
#: the same thing.
#:
#: `none` maps to `cpu` because that is what a software decode *is*; the schema
#: reserves its own `none` for a pipeline with no decode stage at all -- a
#: keypoint-only run reading cached tensors, which costs nothing here.
SCHEMA_DECODE_PATH = {
    "none": "cpu",
    "vaapi": "vaapi",
    "d3d11": "nvdec",
    "mfx": "nvdec",
    "any": "nvdec",
}

#: What `accel` may report. Requested and granted are tracked separately.
ACCEL_NAMES = {
    0: "none",       # cv2.VIDEO_ACCELERATION_NONE
    1: "any",        # cv2.VIDEO_ACCELERATION_ANY
    2: "d3d11",
    3: "vaapi",
    4: "mfx",
}


class VideoSource:
    """An open, positioned video decoder — the `video` boundary artifact.

    Not a stage: it holds the decoder state so the stage can stay stateless and
    so an isolation replay advances through real frames instead of re-reading
    one.
    """

    codec = "h264"

    def __init__(self, path: Path, cap: Any, accel: str, accel_requested: str) -> None:
        import cv2

        self.path = Path(path)
        self._cap = cap
        self.accel = accel
        self.accel_requested = accel_requested
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
        self.n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        #: How many times the replay loop wrapped around. Each wrap costs a
        #: seek plus a keyframe decode, so a long isolation window's overhead
        #: can be bounded rather than assumed negligible.
        self.rewinds = 0
        self.position = 0

    @property
    def wh(self) -> tuple[int, int]:
        return (self.width, self.height)

    def rewind(self) -> None:
        """Seek back to frame 0.

        Assigning `position = 0` does **not** do this: for a video source
        `position` is a counter, and the decoder's real position lives in the
        `VideoCapture`. Setting the counter alone leaves the two disagreeing —
        the next `read()` returns whatever frame came next, not frame 0, while
        `position` claims otherwise and the rewind count is silently wrong.

        It matters for reproducibility rather than for cost: a run that thinks
        it is measuring frames 0..127 and is really measuring 340..467 costs
        the same but is not the same experiment, and the fixtures captured
        alongside it are of different frames again.
        """
        import cv2

        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.position = 0

    def read(self) -> np.ndarray:
        """Next frame as BGR uint8 (H, W, 3), rewinding at end of stream."""
        import cv2

        ok, frame = self._cap.read()
        if not ok:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.rewinds += 1
            self.position = 0
            ok, frame = self._cap.read()
            if not ok:
                raise RuntimeError(
                    f"{self.path} decoded no frames even after rewinding — the "
                    "file is unreadable by this OpenCV build, not merely at EOF"
                )
        self.position += 1
        return frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def describe(self) -> dict[str, Any]:
        return {
            "source": str(self.path),
            "codec": self.codec,
            "source_resolution": f"{self.width}x{self.height}",
            "fps": self.fps,
            "n_frames": self.n_frames,
            "decode_accel": self.accel,
        }

    def __len__(self) -> int:
        return self.n_frames

    def __repr__(self) -> str:
        return (
            f"<VideoSource {self.path.name} {self.width}x{self.height} "
            f"{self.n_frames}f {self.codec}/{self.accel}>"
        )


class ImageSequenceSource(VideoSource):
    """A directory of stills — how PHOENIX-2014 is distributed.

    Kept a separate class rather than a flag so `codec` cannot accidentally
    read `h264` for a PNG directory. Intra-only still decoding is a different
    workload with a different cost, and mixing the two in one decode row would
    describe no real system.
    """

    codec = "png-sequence"

    def __init__(self, path: Path, files: Sequence[Path]) -> None:
        import cv2

        if not files:
            raise ValueError(f"no images matching {IMAGE_SUFFIXES} under {path}")
        self.path = Path(path)
        self.files = list(files)
        self.accel = "none"
        self.accel_requested = "software"
        probe = cv2.imread(str(self.files[0]), cv2.IMREAD_COLOR)
        if probe is None:
            raise RuntimeError(f"could not decode {self.files[0]}")
        self.height, self.width = probe.shape[:2]
        self.fps = 25.0
        self.n_frames = len(self.files)
        self.rewinds = 0
        self.position = 0

    def read(self) -> np.ndarray:
        import cv2

        if self.position >= self.n_frames:
            self.position = 0
            self.rewinds += 1
        frame = cv2.imread(str(self.files[self.position]), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"could not decode {self.files[self.position]}")
        self.position += 1
        return frame

    def rewind(self) -> None:
        self.position = 0

    def close(self) -> None:
        return


def open_video(path: Path | str, accel: str = "software") -> VideoSource:
    """Open a clip. `accel` is 'software', 'hardware' or 'any'.

    A directory is opened as a still sequence; anything else as video.

    Raises if `hardware` is requested and the build does not grant it — a
    software decode reported as hardware is worse than no number at all,
    because it is the one comparison the decode axis exists to make.
    """
    import cv2

    p = Path(path)
    if p.is_dir():
        files = sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES)
        return ImageSequenceSource(p, files)

    if not p.exists():
        raise FileNotFoundError(p)

    requested = {
        "software": cv2.VIDEO_ACCELERATION_NONE,
        "hardware": cv2.VIDEO_ACCELERATION_ANY,
        "any": cv2.VIDEO_ACCELERATION_ANY,
    }
    if accel not in requested:
        raise ValueError(f"accel must be one of {sorted(requested)}, got {accel!r}")

    cap = cv2.VideoCapture(
        str(p), cv2.CAP_FFMPEG, [int(cv2.CAP_PROP_HW_ACCELERATION), requested[accel]]
    )
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open {p}")

    granted_code = int(cap.get(cv2.CAP_PROP_HW_ACCELERATION))
    granted = ACCEL_NAMES.get(granted_code, f"code-{granted_code}")

    if accel == "hardware" and granted == "none":
        cap.release()
        raise RuntimeError(
            f"hardware decode requested for {p.name} but the OpenCV build "
            f"granted '{granted}'. pip's opencv-python wheels ship without "
            "CUVID/VAAPI, so this silently falls back to libavcodec. Either "
            "build OpenCV with hardware decode, or record accel='software' "
            "and say so in the row — do not report a software number under a "
            "hardware label."
        )
    return VideoSource(p, cap, accel=granted, accel_requested=accel)


class VideoDecodeStage(Stage):
    """Decode one frame from an open source."""

    name = "decode"
    input_key = "video"
    output_key = "frames"
    granularity = "frame"
    uses_cuda = False

    def __init__(self, accel: str = "software", source: Optional[Any] = None) -> None:
        self.accel_requested = accel
        #: The source is the only thing that knows what the build actually
        #: granted, and the row must report that rather than what was asked
        #: for. Optional so a stage can be constructed before its clip is
        #: opened; when absent, the request is reported and said to be so.
        self.source = source

    def bind(self, source: Any) -> "VideoDecodeStage":
        """Attach the opened source, so `config()` reports what really ran."""
        self.source = source
        return self

    def config(self) -> dict[str, Any]:
        if self.source is None:
            path = "cpu" if self.accel_requested == "software" else "nvdec"
            return {
                "decode_path": path,
                "notes": f"{self.accel_requested} requested; source not bound, "
                         "so this is the request rather than what was granted",
            }

        granted = getattr(self.source, "accel", "none")
        codec = getattr(self.source, "codec", "unknown")
        return {
            "decode_path": SCHEMA_DECODE_PATH.get(granted, "cpu"),
            "notes": f"opencv-ffmpeg, codec {codec}, accel granted '{granted}'",
        }

    def run(self, video: Any) -> np.ndarray:
        return video.read()
