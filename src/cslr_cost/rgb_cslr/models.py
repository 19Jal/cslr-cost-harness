"""VAC and VAC + SMKD, ported for inference and export.

A ResNet18 per frame, a 1D temporal conv (K5 P2 K5 P2, so time / 4), a two-layer
BiLSTM and a linear classifier, CTC over 1296 classes (1295 glosses + blank).

Ported from VIPL-SLP/VAC_CSLR @ 71f3e03 (Apache License 2.0; see NOTICE), not
imported: the original repository downloads ImageNet weights at construction
time and pulls in training code. The math here is theirs line for line; only
the training paths (losses, hooks, decoding inside forward) are left out. Weights load through
the restricted unpickler and strictly: a missing or unexpected key refuses.

VAC's paper checkpoint (`resnet18_slr_pretrained_distill25.pt`) and VAC + SMKD are the
same network but for the classifiers: VAC's are plain `nn.Linear` (with bias), SMKD's
weight-normalised (`weight_norm: True`). `load` tells them apart by the checkpoint's keys.

Deployed stream: the BiLSTM classifier's
output (`sequence_logits`), as both repositories report.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

NUM_CLASSES = 1296
HIDDEN = 1024


class NormLinear(nn.Module):
    """SMKD's weight-normalised classifier: x @ normalize(W, dim=0)."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_dim, out_dim))

    def forward(self, x):
        return torch.matmul(x, F.normalize(self.weight, dim=0))


class TemporalConv(nn.Module):
    """conv_type 2: K5 P2 K5 P2, BatchNorm + ReLU after each conv, no padding."""

    kernel_size = ("K5", "P2", "K5", "P2")

    def __init__(self, input_size: int = 512, hidden_size: int = HIDDEN):
        super().__init__()
        mods = []
        for i, ks in enumerate(self.kernel_size):
            if ks[0] == "P":
                mods.append(nn.MaxPool1d(kernel_size=int(ks[1]), ceil_mode=False))
            else:
                mods += [nn.Conv1d(input_size if i == 0 else hidden_size, hidden_size,
                                   kernel_size=int(ks[1]), stride=1, padding=0),
                         nn.BatchNorm1d(hidden_size), nn.ReLU(inplace=True)]
        self.temporal_conv = nn.Sequential(*mods)
        self.fc = NormLinear(hidden_size, NUM_CLASSES)   # the conv branch's head (VAC)

    @classmethod
    def out_len(cls, n: int) -> int:
        for ks in cls.kernel_size:
            n = n // 2 if ks[0] == "P" else n - (int(ks[1]) - 1)
        return n

    def forward(self, x):            # (B, C, T) -> (B, H, T')
        return self.temporal_conv(x)


class BiLSTM(nn.Module):
    """Two layers, bidirectional, 512 per direction (1024 out). Batch 1, no packing."""

    def __init__(self, input_size: int = HIDDEN, hidden_size: int = HIDDEN):
        super().__init__()
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size // 2,
                           num_layers=2, dropout=0.3, bidirectional=True)

    def forward(self, x):            # (T, B, C) -> (T, B, 1024)
        return self.rnn(x)[0]


class RgbCslr(nn.Module):
    """frames (T, 3, 224, 224), already padded -> sequence logits (T', 1296).

    `kind`: "vac" (torchvision ResNet18, frame-independent, so it may be run in
    chunks of any size). CorrNet's code carries no licence, so it is not ported
    here; see docs/adding-a-pipeline.md for running it from your own export.
    """

    def __init__(self, kind: str, linear_classifier: bool = False):
        super().__init__()
        if kind != "vac":
            raise ValueError(kind)
        self.kind = kind
        self.conv2d = torchvision.models.resnet18(weights=None)
        self.conv2d.fc = nn.Identity()
        self.conv1d = TemporalConv()
        self.temporal_model = BiLSTM()
        self.classifier = NormLinear(HIDDEN, NUM_CLASSES)
        if linear_classifier:                      # VAC's paper checkpoint
            self.conv1d.fc = nn.Linear(HIDDEN, NUM_CLASSES)
            self.classifier = nn.Linear(HIDDEN, NUM_CLASSES)

    def frame_features(self, frames: torch.Tensor, chunk: int = 64) -> torch.Tensor:
        """(T, 3, 224, 224) -> (T, 512)."""
        return torch.cat([self.conv2d(frames[i:i + chunk]) for i in range(0, len(frames), chunk)])

    def head(self, feats: torch.Tensor) -> torch.Tensor:
        """(T, 512) -> (T', 1296) sequence logits."""
        x = self.conv1d(feats.t()[None])                 # (1, 1024, T')
        x = self.temporal_model(x.permute(2, 0, 1))      # (T', 1, 1024)
        return self.classifier(x)[:, 0]

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.head(self.frame_features(frames))


class FrameFrontend(nn.Module):
    """The per-frame part, for export: (N, 3, 224, 224) normalised RGB ->
    (N, 512) pooled features."""

    def __init__(self, m: RgbCslr):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m.conv2d(x)


class ClipHead(nn.Module):
    """The per-clip part, for export: padded per-frame features (1, T, ...) ->
    sequence logits (T', 1296)."""

    def __init__(self, m: RgbCslr):
        super().__init__()
        self.m = m

    def forward(self, feats):
        return self.m.head(feats[0])


def load(kind: str, path: str) -> RgbCslr:
    """Build and strictly load a released checkpoint (eval mode)."""
    from cslr_cost.safe_load import restricted_pickle_module

    ck = torch.load(path, map_location="cpu", pickle_module=restricted_pickle_module())
    sd = ck.get("model_state_dict", ck)
    sd = {k.replace("conv2d.module.", "conv2d."): v for k, v in sd.items()}
    m = RgbCslr(kind, linear_classifier="classifier.bias" in sd)
    ref = m.state_dict()
    # BatchNorm counters are bookkeeping, not weights; tolerate their absence only
    missing = [k for k in ref if k not in sd and not k.endswith("num_batches_tracked")]
    unexpected = [k for k in sd if k not in ref]
    wrong = [k for k in ref if k in sd and sd[k].shape != ref[k].shape]
    if missing or unexpected or wrong:
        raise ValueError(f"{path}: missing {missing[:5]}, unexpected {unexpected[:5]}, wrong shape {wrong[:5]}")
    m.load_state_dict(sd, strict=False)
    return m.eval()
