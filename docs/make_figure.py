"""Render Figure 1, the SafeSiteGen framework diagram.

    python docs/make_figure.py --out docs/framework.png
"""

from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK = "#1d2733"
MUTED = "#5f6b78"
LINE = "#9aa6b2"
ACCENT = "#b4322f"
ACCENT_FILL = "#fbeceb"
SPINE_FILL = "#eef2f6"
INPUT_FILL = "#f7f4ec"
LOOP_FILL = "#eef4ef"

FONT = "DejaVu Sans"


def box(ax, x, y, w, h, title, body="", fill=SPINE_FILL, edge=LINE, title_size=8.4,
        body_size=6.9, lw=0.9, title_colour=INK):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.0,rounding_size=1.1",
        linewidth=lw, edgecolor=edge, facecolor=fill, zorder=2))
    if body:
        ax.text(x + w / 2, y + h - 2.6, title, ha="center", va="top", fontsize=title_size,
                fontweight="bold", color=title_colour, family=FONT, zorder=3)
        ax.text(x + w / 2, y + h - 6.0, body, ha="center", va="top", fontsize=body_size,
                color=MUTED, family=FONT, linespacing=1.5, zorder=3)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=title_size,
                fontweight="bold", color=title_colour, family=FONT, zorder=3)


def arrow(ax, p0, p1, colour=LINE, style="-|>", lw=1.1, rad=0.0, dashed=False, z=1):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle=style, mutation_scale=9, linewidth=lw, color=colour,
        connectionstyle=f"arc3,rad={rad}",
        linestyle=(0, (3.2, 2.2)) if dashed else "solid", zorder=z))


def label(ax, x, y, text, size=6.5, colour=MUTED, ha="center", style="normal", weight="normal"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=size, color=colour,
            family=FONT, style=style, fontweight=weight, zorder=4)


def build(out_path: str) -> str:
    fig, ax = plt.subplots(figsize=(10.0, 6.2), dpi=320)
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 132)
    ax.axis("off")

    # ------------------------------------------------------------ input band
    label(ax, 4, 129, "KNOWLEDGE SOURCES", size=7.2, colour=MUTED, ha="left", weight="bold")
    inputs = [
        (4, "Task specification",
         "trade, activity, target hazard\nclasses, difficulty band"),
        (52, "Regulatory corpus",
         "29 CFR 1926 clauses compiled\ninto checkable predicates"),
        (100, "Incident evidence",
         "OSHA IMIS narratives, NIOSH\nFACE reports, near-miss logs"),
        (148, "Site substrate",
         "IFC / BIM, photogrammetry,\nLiDAR, 3D Gaussian splats"),
    ]
    for x, title, body in inputs:
        box(ax, x, 104, 44, 20, title, body, fill=INPUT_FILL, edge="#d8cfbb")

    # ----------------------------------------------------------- main spine
    sy, sh = 70, 24
    box(ax, 4, sy, 40, sh, "Specification",
        "closed-vocabulary\nrequest object", fill=SPINE_FILL)
    box(ax, 54, sy, 44, sh, "Constrained generation",
        "retrieval-augmented, schema-\nconstrained decoding emits the\nHazard Scenario Graph",
        fill=SPINE_FILL)
    box(ax, 108, sy, 40, sh, "Geometric realisation",
        "constraint solver places the\ngraph into real captured\nsite geometry", fill=SPINE_FILL)
    box(ax, 158, sy, 38, sh, "Runtime scene",
        "Unity / XR build with\nper-hazard answer key", fill=SPINE_FILL)

    for x0, x1 in ((44, 54), (98, 108)):
        arrow(ax, (x0, sy + sh / 2), (x1, sy + sh / 2))

    # knowledge feeds
    arrow(ax, (26, 104), (26, sy + sh), dashed=True)
    arrow(ax, (70, 104), (70, sy + sh), dashed=True)
    arrow(ax, (122, 104), (86, sy + sh), dashed=True, rad=-0.16)
    arrow(ax, (170, 104), (134, sy + sh), dashed=True, rad=-0.16)
    # the same clause set drives both generation and checking, routed down the
    # free corridor between the generation and realisation boxes
    arrow(ax, (88, 104), (104, 62), dashed=True, rad=0.20)
    label(ax, 100, 98, "same clauses", size=5.8, colour=MUTED, style="italic")

    # ------------------------------------------------------- validation gate
    gx, gy, gw, gh = 100, 22, 58, 40
    ax.add_patch(FancyBboxPatch(
        (gx, gy), gw, gh, boxstyle="round,pad=0.0,rounding_size=1.1",
        linewidth=1.6, edgecolor=ACCENT, facecolor=ACCENT_FILL, zorder=2))
    ax.text(gx + gw / 2, gy + gh - 2.6, "VALIDATION GATE", ha="center",
            va="top", fontsize=8.6, fontweight="bold", color=ACCENT, family=FONT, zorder=3)
    checks = [
        ("Regulatory", "every declared hazard is realised,\nno undeclared violation exists"),
        ("Physical", "support, clearance, reachability,\nhazard affordance"),
        ("Pedagogical", "cue perceivable, difficulty in band,\ndistractors present"),
    ]
    cy = gy + gh - 10.5
    for name, detail in checks:
        ax.text(gx + 3.0, cy, name, ha="left", va="center", fontsize=7.0,
                fontweight="bold", color=INK, family=FONT, zorder=3)
        ax.text(gx + 3.0, cy - 4.2, detail, ha="left", va="center", fontsize=6.0,
                color=MUTED, family=FONT, linespacing=1.45, zorder=3)
        cy -= 10.6

    arrow(ax, (129, sy), (129, gy + gh), colour=ACCENT, lw=1.4)
    label(ax, 145, 66, "realised scene", size=6.1, colour=ACCENT)

    arrow(ax, (gx + gw, gy + gh / 2), (177, sy), colour=ACCENT, lw=1.4, rad=-0.3)
    label(ax, 174, 46, "pass", size=6.6, colour=ACCENT, weight="bold")

    # repair / reject loop back into generation
    box(ax, 30, 26, 58, 20, "Repair and reject log",
        "targeted repair, or rejection into a\nlabelled negative corpus used for\npreference tuning",
        fill="#ffffff", edge=ACCENT, title_size=7.6, body_size=6.0, lw=1.1, title_colour=ACCENT)
    arrow(ax, (gx, gy + gh / 2), (88, 40), colour=ACCENT, lw=1.4, rad=0.22)
    label(ax, 95, 48, "fail", size=6.6, colour=ACCENT, weight="bold")
    arrow(ax, (66, 46), (72, sy), colour=ACCENT, lw=1.3, rad=-0.18)

    # ---------------------------------------------------------- adaptation loop
    GREEN = "#3f7a58"
    box(ax, 148, 2, 48, 16, "Trainee telemetry",
        "gaze, response, latency", fill=LOOP_FILL, edge="#b6cbbc",
        title_size=7.4, body_size=6.2)
    box(ax, 4, 2, 40, 16, "Learner model",
        "per-class knowledge tracing,\nlayout novelty constraint",
        fill=LOOP_FILL, edge="#b6cbbc", title_size=7.4, body_size=6.0)

    arrow(ax, (186, sy), (186, 18), colour=GREEN, lw=1.2)
    arrow(ax, (148, 10), (44, 10), colour=GREEN, lw=1.2)
    arrow(ax, (14, 18), (14, sy), colour=GREEN, lw=1.2)
    label(ax, 22, 60, "next specification", size=6.1, colour=GREEN)
    label(ax, 96, 13, "what this trainee actually missed", size=6.1, colour=GREEN)

    ax.text(4, -6, "Dashed arrows carry knowledge, solid arrows carry content. The gate is the "
                   "contribution: content reaches a trainee only after every declared hazard is "
                   "verified present\nand every undeclared one is verified absent, because both "
                   "failure modes teach an incorrect mental model.",
            ha="left", va="bottom", fontsize=6.4, color=MUTED, family=FONT,
            style="italic", linespacing=1.5)

    fig.savefig(out_path, bbox_inches="tight", facecolor="white", pad_inches=0.14)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="docs/framework.png")
    a = p.parse_args()
    print(build(a.out))
