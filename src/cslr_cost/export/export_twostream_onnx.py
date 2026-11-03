"""Export TwoStream-SLR's deployed network to ONNX.

Runs inside the container with TwoStream's own code on the path (its
`modelling/` and its dependencies importable):

    scripts/dev.sh bash data/twostream/export_onnx.sh

What is exported is what `ensemble_last` needs and nothing else:

    rgb       (1, 3, T, 224, 224)  BGR, (x/255 - 0.5) / 0.5      TwoStream's eval preprocessing
    heatmaps  (1, 79, T, 112, 112) (h - 0.5) / 0.5               rendered from keypoints
        -> visual_backbone_twostream (two S3D streams, bilateral lateral connections)
        -> rgb head, keypoint head, fuse head (keypoint ++ rgb features)
        -> the three heads' logits (1, T/4, 1235) each; ensemble_last = log(sum of their
           softmax), computed outside the graph in float32 (see `ensemble`)

TwoStream's own evaluation also runs the pyramid's auxiliary heads
(`visual_head_remain`, `visual_head_keypoint_remain`), which only training losses
use; they do not touch `ensemble_last`, so a deployment does not run them and
neither does this graph. Heatmap rendering (79 Gaussians, sigma 8, at 260x210, then
resized to 112x112) stays outside the graph, in the harness's postproc stage.

Safety: TwoStream's tokenizer reads `gloss2ids.pkl` with `pickle.load`; it is
replaced here by an unpickler that refuses every global (the file is a plain
dict), and the checkpoint loads with `weights_only=True`.

Parity: the exported graph against TwoStream's own modules on a real dev clip,
with onnxruntime on CPU, before anything is kept.

`--variant keypoint` exports TwoStream's **keypoint stream alone** instead (its
released PHOENIX-2014 `phoenix-2014_keypoint` checkpoint, the stage-1 model the joint
network starts from): the keypoint S3D on the same 79 heatmaps, no lateral
connections, no pyramid, and one head,

    heatmaps  (1, 79, T, 112, 112) -> visual_backbone_keypoint -> visual_head_keypoint
        -> logits_keypoint (1, T/4, 1235); decoded as TwoStream decodes a single
           stream: log_softmax of the head's logits (float32, outside the graph)
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

TS_DIR = Path("data/TwoStream-SLRT/SLRT/TwoStreamNetwork")
VARIANTS = {
    "twostream": (Path("data/twostream/dump/config.yaml"), Path("data/twostream/dump/ckpts/best.ckpt"),
                  "twostream-slr"),
    # TwoStream's experiments/configs/TwoStream/phoenix-2014_keypoint.yaml with local paths and
    # `pretrained_ckpt: scratch` (every weight comes from the checkpoint)
    "keypoint": (Path("data/twostream/keypoint/config.yaml"),
                 Path("data/TwoStream-SLRT/keypoint_stream/best.ckpt"), "twostream-kp"),
}
CONFIG, CKPT, STEM = VARIANTS["twostream"]
KEYPOINTS = Path("data/twostream/phoenix-2014-keypoints.devtest.pkl")

#: sorted(use_keypoints) = face_others_1_3, hand, mouth_half, pose (Dataset.py, Hrnet_Part2index)
JOINTS_79 = (list(range(23, 71))[::3] + list(range(91, 133)) + list(range(71, 91))[::2] + list(range(11)))
assert len(JOINTS_79) == 79


class _PlainUnpickler(pickle.Unpickler):
    """Plain containers only: any global is refused."""

    def find_class(self, module, name):
        if (module, name) in {("numpy.core.multiarray", "_reconstruct"), ("numpy", "ndarray"),
                              ("numpy", "dtype"), ("collections", "OrderedDict")}:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"refused global {module}.{name}")


def _safe_pickle_load(f, *a, **k):
    return _PlainUnpickler(f).load()


def build(config: Path = None, ckpt: Path = None):
    config, ckpt = config or CONFIG, ckpt or CKPT
    pickle.load = _safe_pickle_load          # TwoStream's tokenizer reads gloss2ids.pkl with it
    sys.path.insert(0, str(TS_DIR))
    from modelling.model import build_model
    from utils.misc import load_config, make_logger, neq_load_customized

    cfg = load_config(str(config))
    Path("data/twostream/export").mkdir(parents=True, exist_ok=True)
    make_logger(model_dir="data/twostream/export", log_file="export.log")   # their modules expect it
    cfg["device"] = torch.device("cpu")
    model = build_model(cfg)
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)["model_state"]
    missing = [k for k in model.state_dict() if k not in sd]
    unused = [k for k in sd if k not in model.state_dict()]
    assert not missing and not unused, (missing[:5], unused[:5])
    neq_load_customized(model, sd, verbose=False)
    return model.eval()


class Deployed(torch.nn.Module):
    """ensemble_last from preprocessed inputs, batch 1, full-length clip."""

    def __init__(self, rn):
        super().__init__()
        self.rn = rn

    def forward(self, rgb, heatmaps):
        rn = self.rn
        lengths = torch.ones(1, dtype=torch.long, device=rgb.device) * rgb.shape[2]
        s3d = rn.visual_backbone_twostream(x_rgb=rgb, x_pose=heatmaps, sgn_lengths=lengths)
        mask, vlen = s3d["sgn_mask"][-1], s3d["valid_len_out"][-1]
        r_in, k_in = s3d["rgb_fea_lst"][-1], s3d["pose_fea_lst"][-1]
        r = rn.visual_head(x=r_in, mask=mask, valid_len_in=vlen)["gloss_logits"]
        k = rn.visual_head_keypoint(x=k_in, mask=mask, valid_len_in=vlen)["gloss_logits"]
        f = rn.visual_head_fuse(x=torch.cat([k_in, r_in], dim=-1), mask=mask, valid_len_in=vlen)["gloss_logits"]
        return r, k, f


class DeployedKeypoint(torch.nn.Module):
    """The keypoint stream alone: its S3D and its one head, batch 1, full-length clip."""

    def __init__(self, rn):
        super().__init__()
        self.rn = rn

    def forward(self, heatmaps):
        rn = self.rn
        lengths = torch.ones(1, dtype=torch.long, device=heatmaps.device) * heatmaps.shape[2]
        s3d = rn.visual_backbone_keypoint(sgn_videos=heatmaps, sgn_lengths=lengths)
        return rn.visual_head_keypoint(x=s3d["sgn"], mask=s3d["sgn_mask"][-1],
                                       valid_len_in=s3d["valid_len_out"][-1])["gloss_logits"]


def single(k):
    """A single stream decodes from its head's log_softmax (float32, outside the graph)."""
    return k.float().log_softmax(-1)


def ensemble(r, k, f):
    """ensemble_last: log of the sum of the three heads' softmax, in float32 (outside the
    graph: in an FP16 engine most of a 1,235-class softmax rounds to 0, as MSKA's did)."""
    return sum(x.float().softmax(-1) for x in (r, k, f)).log()


def render_heatmaps(kp: torch.Tensor, raw_hw=(260, 210), out=112, sigma=8.0) -> torch.Tensor:
    """(T, 79, 3) keypoints in 210x260 pixel coordinates -> (1, 79, T, 112, 112), normalised.
    TwoStream's gen_gaussian_hmap_op at raw size, then torchvision Resize (bilinear, no
    antialias on tensors in this torchvision), then (h - 0.5) / 0.5."""
    H, W = raw_hw
    y = torch.arange(H, dtype=torch.float32, device=kp.device).view(1, 1, H, 1)
    x = torch.arange(W, dtype=torch.float32, device=kp.device).view(1, 1, 1, W)
    cy, cx = kp[..., 1].unsqueeze(-1).unsqueeze(-1), kp[..., 0].unsqueeze(-1).unsqueeze(-1)
    hm = torch.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2 * sigma ** 2))          # (T, 79, H, W)
    hm = torch.nn.functional.interpolate(hm, size=(out, out), mode="bilinear", align_corners=False)
    return ((hm - 0.5) / 0.5).permute(1, 0, 2, 3)[None].contiguous()


def preprocess_rgb(frames_bgr: np.ndarray) -> torch.Tensor:
    """(T, H, W, 3) uint8 BGR -> (1, 3, T, 224, 224): resize, /255, (x - 0.5) / 0.5.
    TwoStream loads RGB with PIL and flips to BGR before the network; BGR in = the same."""
    x = torch.from_numpy(frames_bgr).permute(0, 3, 1, 2).float() / 255
    x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    return ((x - 0.5) / 0.5).permute(1, 0, 2, 3)[None].contiguous()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("/workspace/models/onnx/twostream"))
    ap.add_argument("--frames", type=int, default=64)
    ap.add_argument("--params", action="store_true", help="count the deployed modules' parameters and exit")
    ap.add_argument("--variant", choices=tuple(VARIANTS), default="twostream",
                    help="twostream: the joint network; keypoint: the keypoint stream alone")
    a = ap.parse_args()
    import cv2

    config, ckpt, stem = VARIANTS[a.variant]
    kp_only = a.variant == "keypoint"
    model = build(config, ckpt)
    rn = model.recognition_network
    if a.params and kp_only:
        tot = 0
        for k, m in {"keypoint S3D": rn.visual_backbone_keypoint, "keypoint head": rn.visual_head_keypoint}.items():
            n = sum(p.numel() for p in m.parameters()); tot += n
            print(f"PARAMS {k}: {n / 1e6:.3f} M")
        allp = sum(p.numel() for p in model.parameters())
        print(f"PARAMS deployed total: {tot / 1e6:.3f} M; whole checkpointed model: {allp / 1e6:.3f} M")
        return
    if a.params:
        parts = {"backbone (two S3D streams + lateral connections)": rn.visual_backbone_twostream,
                 "rgb head": rn.visual_head, "keypoint head": rn.visual_head_keypoint, "fuse head": rn.visual_head_fuse}
        tot = 0
        for k, m in parts.items():
            n = sum(p.numel() for p in m.parameters()); tot += n
            b = sum(b.numel() for b in m.buffers())
            print(f"PARAMS {k}: {n / 1e6:.3f} M (buffers {b / 1e6:.3f} M)")
        allp = sum(p.numel() for p in model.parameters())
        print(f"PARAMS deployed total: {tot / 1e6:.3f} M; whole checkpointed model (aux heads etc.): {allp / 1e6:.3f} M")
        return
    # padding="same" (the heads' temporal convs) does not export on this torch; for an
    # odd kernel it is exactly symmetric padding of dilation * (k - 1) / 2
    n_same = 0
    for m in model.modules():
        if isinstance(m, torch.nn.modules.conv._ConvNd) and m.padding == "same":
            pads = []
            for k, d in zip(m.kernel_size, m.dilation):
                assert (d * (k - 1)) % 2 == 0, (m, "even total padding needed")
                pads.append(d * (k - 1) // 2)
            m.padding = tuple(pads)
            m._reversed_padding_repeated_twice = [p for p in reversed(pads) for _ in range(2)]
            n_same += 1
    print(f"padding='same' -> numeric in {n_same} convs", flush=True)
    dep = (DeployedKeypoint(rn) if kp_only else Deployed(rn)).eval()

    # a real dev clip: its frames and its keypoints
    kps = _PlainUnpickler(open(KEYPOINTS, "rb")).load()
    name = next(k for k in kps if "/dev/" in k)
    fdir = Path("data/phoenix2014/phoenix2014-release/phoenix-2014-multisigner/features") / name.rsplit("/", 1)[0]
    paths = sorted(fdir.glob("*.png"))[: a.frames]
    frames = np.stack([cv2.imread(str(p)) for p in paths])
    kp = torch.from_numpy(np.asarray(kps[name]["keypoints"], dtype=np.float32)[: len(paths)][:, JOINTS_79])
    rgb, hm = preprocess_rgb(frames), render_heatmaps(kp)

    # the same clip through TwoStream's own preprocessing, for the heatmap path
    with torch.no_grad():
        theirs = rn.generate_batch_heatmap(kp[None])                       # (1, T, 79, 260, 210)
        _, theirs = rn.augment_preprocess_inputs(is_train=False, sgn_videos=None, sgn_heatmaps=theirs)
        if not kp_only:
            rgb_in = torch.from_numpy(frames[..., ::-1].copy()).permute(0, 3, 1, 2)[None].float() / 255  # PIL RGB
            theirs_rgb, _ = rn.augment_preprocess_inputs(is_train=False, sgn_videos=rgb_in, sgn_heatmaps=None)
    d_hm = float((theirs - hm).abs().max())
    d_rgb = None if kp_only else float((theirs_rgb - rgb).abs().max())
    print(f"preprocessing vs TwoStream's own: heatmaps max|d| {d_hm:.2e}, rgb max|d| {d_rgb}", flush=True)

    a.out_dir.mkdir(parents=True, exist_ok=True)
    out = a.out_dir / f"{stem}.onnx"
    if kp_only:
        inputs, in_names, names, combine = (hm,), ["heatmaps"], ["logits_keypoint"], single
    else:
        inputs, in_names, names, combine = (rgb, hm), ["rgb", "heatmaps"], ["logits_rgb", "logits_keypoint", "logits_fuse"], ensemble
    with torch.no_grad():
        outs = dep(*inputs)
        ref = combine(*(outs if isinstance(outs, tuple) else (outs,)))
    torch.onnx.export(dep, inputs, str(out), opset_version=17, input_names=in_names,
                      output_names=names,
                      dynamic_axes={**{n: {2: "frames"} for n in in_names}, **{n: {1: "steps"} for n in names}},
                      do_constant_folding=True)
    import onnxruntime as ort

    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])

    def run_onnx(xs):
        return combine(*(torch.from_numpy(x) for x in sess.run(None, {n: x.numpy() for n, x in zip(in_names, xs)}))).numpy()

    got = run_onnx(inputs)
    d = float(np.abs(ref.numpy() - got).max())
    agree = float((ref.numpy().argmax(-1) == got.argmax(-1)).mean())
    # a second length, so a shape baked into the trace shows up here and not on the Orin
    inputs2 = tuple(x[:, :, :40] for x in inputs)
    with torch.no_grad():
        outs2 = dep(*inputs2)
        ref2 = combine(*(outs2 if isinstance(outs2, tuple) else (outs2,))).numpy()
    d2 = float(np.abs(ref2 - run_onnx(inputs2)).max())
    import hashlib

    side = {"schema": "twostream-onnx-export/1", "variant": a.variant,
            "checkpoint_sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest(),
            "onnx_sha256": hashlib.sha256(out.read_bytes()).hexdigest(), "opset": 17,
            "parity_clip": name, "parity_frames": len(paths), "parity_max_abs": d, "parity_argmax_agree": agree,
            "parity_40_frames_max_abs": d2, "heatmap_vs_twostream_max_abs": d_hm, "rgb_vs_twostream_max_abs": d_rgb,
            "vocab": int(got.shape[-1]), "joints": JOINTS_79}
    (a.out_dir / f"{stem}.json").write_text(json.dumps(side, indent=2) + "\n")
    print(json.dumps({k: v for k, v in side.items() if k != "joints"}, indent=1))
    if d > 1e-3 or agree < 1.0 or d2 > 1e-3:
        out.unlink()
        raise SystemExit("parity failed; ONNX removed")


if __name__ == "__main__":
    main()
