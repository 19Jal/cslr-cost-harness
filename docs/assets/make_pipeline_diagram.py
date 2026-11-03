"""Draw docs/assets/pipeline-{light,dark}.svg, the README's pipeline diagram.

    python docs/assets/make_pipeline_diagram.py

Plain SVG, no dependencies. GitHub picks the variant that matches the reader's
theme through the <picture> element in README.md.
"""

from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).resolve().parent

THEMES = {
    "light": dict(
        ink="#1f2328", muted="#59636e", line="#8c959f",
        box="#f6f8fa", box_edge="#d0d7de",
        pose="#fff1e5", pose_edge="#eb6834",
        time="#ddf4ff", time_edge="#54aeff",
        energy="#dafbe1", energy_edge="#4ac26b",
        guard="#fbefff", guard_edge="#c297ff",
        gain="#fff8c5", gain_edge="#d4a72c",
        chip="#ffffff", chip_edge="#d0d7de",
    ),
    "dark": dict(
        ink="#e6edf3", muted="#9198a1", line="#6e7681",
        box="#161b22", box_edge="#3d444d",
        pose="#3d2310", pose_edge="#f0883e",
        time="#0d2346", time_edge="#388bfd",
        energy="#0f2e1b", energy_edge="#3fb950",
        guard="#271052", guard_edge="#a371f7",
        gain="#2e2a0f", gain_edge="#d29922",
        chip="#0d1117", chip_edge="#3d444d",
    ),
}

FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
W, H = 900, 520
BOX_W, BOX_H, GAP, ASSEMBLE = 100, 62, 18, 62
LEFT = (W - (7 * BOX_W + 5 * GAP + ASSEMBLE)) / 2

STAGES = [  # name, where it runs, is it the backbone slot
    ("decode", "CPU", False),
    ("detect", "optional", False),
    ("crop", "CPU", False),
    ("pose / CNN", "GPU", True),
    ("postproc", "CPU", False),
    ("head", "GPU", False),
    ("CTC", "CPU", False),
]


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, s, c, size=13, weight=400, anchor="middle"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" fill="{c}" font-size="{size}" '
            f'font-weight="{weight}" text-anchor="{anchor}">{esc(s)}</text>')


def rect(x, y, w, h, fill, edge, r=8, width=1.2):
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{r}" '
            f'fill="{fill}" stroke="{edge}" stroke-width="{width}"/>')


def band(y, h, fill, edge, title, detail, t):
    x0, x1 = LEFT, W - LEFT
    return "\n".join([
        rect(x0, y, x1 - x0, h, fill, edge, r=8),
        text(x0 + 16, y + 21, title, t["ink"], 14, 600, "start"),
        text(x0 + 16, y + 40, detail, t["muted"], 12.5, 400, "start"),
    ])


def draw(t: dict) -> str:
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
           f'viewBox="0 0 {W} {H}" font-family="{FONT}">',
           f'<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
           f'markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" '
           f'fill="{t["line"]}"/></marker></defs>']

    # stage boxes
    xs = []
    x = LEFT
    for i, _ in enumerate(STAGES):
        xs.append(x)
        x += BOX_W + (ASSEMBLE if i == 4 else GAP)
    top = 58
    for (name, where, backbone), x in zip(STAGES, xs):
        out.append(rect(x, top, BOX_W, BOX_H,
                        t["pose"] if backbone else t["box"],
                        t["pose_edge"] if backbone else t["box_edge"],
                        width=1.8 if backbone else 1.2))
        out.append(text(x + BOX_W / 2, top + 28, name, t["ink"], 14, 600))
        out.append(text(x + BOX_W / 2, top + 47, where, t["muted"], 12))

    # arrows between stages
    cy = top + BOX_H / 2
    for i in range(6):
        x0, x1 = xs[i] + BOX_W, xs[i + 1]
        out.append(f'<line x1="{x0 + 2:.1f}" y1="{cy}" x2="{x1 - 3:.1f}" y2="{cy}" '
                   f'stroke="{t["line"]}" stroke-width="1.4" marker-end="url(#a)"/>')
    mid = (xs[4] + BOX_W + xs[5]) / 2
    out.append(text(mid, cy - 9, "assemble", t["muted"], 11))

    # group brackets above the boxes
    def bracket(x0, x1, label):
        y = 40
        return "\n".join([
            f'<path d="M{x0:.1f},{y + 8} L{x0:.1f},{y} L{x1:.1f},{y} L{x1:.1f},{y + 8}" '
            f'fill="none" stroke="{t["line"]}" stroke-width="1.2"/>',
            text((x0 + x1) / 2, y - 9, label, t["ink"], 13, 600),
        ])
    out.append(bracket(xs[0], xs[4] + BOX_W, "per frame: every frame of the video"))
    out.append(bracket(xs[5], xs[6] + BOX_W, "per clip"))

    # costs, per stage
    out.append(text(LEFT, 142, "COSTS, per stage", t["muted"], 11.5, 600, "start"))
    out.append(band(148, 54, t["time"], t["time_edge"],
                    "Time: every stage timed inside the running pipeline",
                    "CUDA events for GPU stages · median and p95, never the mean · "
                    "stages must sum to end-to-end within 5%", t))
    out.append(band(210, 54, t["energy"], t["energy_edge"],
                    "Energy and memory: every stage replayed alone, ≥ 60 s, on its recorded inputs",
                    "power rails sample too slowly to split one pass · energy in total and "
                    "above idle · peak device memory", t))

    # the two guards on the cost figures
    half = (W - 2 * LEFT - 10) / 2
    for i, (title, detail) in enumerate([
        ("Idle bracket", "20 s settle + 10 s idle, before and after the run"),
        ("Contention gate", "other GPU processes or foreign CPU, every 2 s"),
    ]):
        x = LEFT + i * (half + 10)
        out.append(rect(x, 272, half, 46, t["guard"], t["guard_edge"]))
        out.append(text(x + 16, 291, title, t["ink"], 14, 600, "start"))
        out.append(text(x + 16, 309, detail, t["muted"], 12.5, 400, "start"))

    # regimes
    out.append(text(LEFT, 344, "Regimes", t["ink"], 13, 600, "start"))
    chips = ["batch 1 (and batch N)", "live: paced 25 fps, overruns counted",
             "sustained: 10 min", "first-gloss latency"]
    cx = LEFT + 72
    for c in chips:
        w = 13 + 6.9 * len(c)
        out.append(rect(cx, 328, w, 26, t["chip"], t["chip_edge"], r=13))
        out.append(text(cx + w / 2, 345, c, t["ink"], 12))
        cx += w + 8

    # gains, per pipeline
    out.append(text(LEFT, 382, "GAINS, per pipeline", t["muted"], 11.5, 600, "start"))
    out.append(band(388, 54, t["gain"], t["gain_edge"],
                    "Accuracy: WER with deletions, substitutions and insertions",
                    "the deployed output, through the same engines · both PHOENIX-2014 "
                    "conventions · never on upscaled input", t))

    # output
    out.append(f'<line x1="{LEFT}" y1="462" x2="{W - LEFT}" y2="462" '
               f'stroke="{t["box_edge"]}" stroke-width="1"/>')
    out.append(text(W / 2, 486, "→ costs and gains side by side: one JSONL row per measurement", t["ink"], 14, 600))
    out.append(text(W / 2, 506, "device · power mode · clocks · engines · idle · "
                    "contention verdict · git commit", t["muted"], 12.5))
    out.append("</svg>")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    for name, theme in THEMES.items():
        (OUT / f"pipeline-{name}.svg").write_text(draw(theme))
        print("wrote", OUT / f"pipeline-{name}.svg")
