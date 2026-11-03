"""Resolution ladder.

A controlled resolution sweep built by resampling PHOENIX-2014, valid because
compute is a function of tensor dimensions, not image content: a pose estimator
fed a 720x1280 frame does identical work whether that frame was captured
natively or upscaled from 210x260.

The corollary is enforced, not merely documented: upscaling adds no
information, so WER at a resampled resolution is meaningless. `wer_guard`
refuses it, and the same rule is enforced again in
`cslr_cost.harness.schema`.

Generated media lives in `data/res_ladder/` (untracked, ~6.4G); the
catalogue and clip lists live in `manifests/res_ladder/` (tracked).
"""
