"""cslr-cost-harness: whole-pipeline cost measurement for continuous sign language recognition.

Subpackages:

    harness/    the measurement tool: stages, runners, energy backends, schema, CLI
    eval/       WER (del/sub/ins) under both PHOENIX-2014 conventions; pose AP
    export/     ONNX exports with parity checks; INT8 post-training quantisation
    rgb_cslr/   the VAC port and WER for RGB and two-stream models
    ladder/     resolution-ladder cataloguing and the no-WER-on-resampled guard
"""

__version__ = "0.1.0"
