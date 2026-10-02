"""Widescreen banner for the README hero image / GitHub social preview.

Draws the same architecture-lineage git-graph that opens MAP.md (Transformer
trunk -> state-space branch -> Jamba hybrid merge), condensed to the key
milestones so it reads at a glance, in the same paper/ink/category palette
docs/visual-map.html already uses -- one visual identity, not a new one.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PAPER = "#eef1f5"
INK = "#151a21"
INK_SOFT = "#4b5768"
LINE = "#ccd4de"

C_TRANSFORMER = "#c05f24"   # --c-attention
C_STATESPACE = "#2e6b8f"    # --c-statespace
C_HYBRID = "#6b6558"        # --c-hybrid

Y_TRANSFORMER = 1.6
Y_STATESPACE = 0.2
Y_HYBRID = 0.9

# (x, label, y-offset direction for the text: 1 above, -1 below)
ROOT = (0.0, "sequence\nmodeling", -1)

TRANSFORMER_COMMITS = [
    (1.0, "Transformer\n(2017)", 1),
    (3.0, "GPT · RoPE ·\nFlashAttention", 1),
    (5.5, "MQA/GQA · MLA ·\nMoE routing", 1),
    (8.5, "LLaMA · DeepSeek-V2 ·\nMixtral", 1),
]

STATESPACE_COMMITS = [
    (3.0, "S4\n(2021)", -1),
    (5.5, "Mamba\n(2023)", -1),
    (7.6, "Mamba-2 / SSD\n(2024)", -1),
    (9.5, "Gated DeltaNet\n/ KDA", -1),
]

HYBRID_COMMIT = (6.6, "Jamba (2024)", 0)


def main() -> None:
    # 12.8 x 6.4in @ 200dpi = 2560x1280px, 2x GitHub's recommended 1280x640
    # social preview size (retina-sharp, GitHub downsamples).
    fig, ax = plt.subplots(figsize=(12.8, 6.4))
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    fig.subplots_adjust(left=0.03, right=0.97, top=0.8, bottom=0.12)

    ax.set_xlim(-0.6, 10.6)
    ax.set_ylim(-0.45, 2.1)
    ax.set_axis_off()


    # Root -> first transformer commit (single trunk before the branch).
    ax.plot([ROOT[0], TRANSFORMER_COMMITS[0][0]], [Y_TRANSFORMER, Y_TRANSFORMER],
             color=C_TRANSFORMER, linewidth=2.4, zorder=1)

    # Transformer lane.
    tx = [TRANSFORMER_COMMITS[0][0]] + [c[0] for c in TRANSFORMER_COMMITS[1:]]
    ax.plot(tx, [Y_TRANSFORMER] * len(tx), color=C_TRANSFORMER, linewidth=2.4, zorder=1)

    # State-space lane.
    sx = [c[0] for c in STATESPACE_COMMITS]
    ax.plot(sx, [Y_STATESPACE] * len(sx), color=C_STATESPACE, linewidth=2.4, zorder=1)

    # Branch connector: transformer trunk -> first state-space commit.
    ax.plot([TRANSFORMER_COMMITS[0][0], STATESPACE_COMMITS[0][0]],
             [Y_TRANSFORMER, Y_STATESPACE],
             color=C_STATESPACE, linewidth=2.0, alpha=0.8, zorder=1)

    # Hybrid branch: Jamba (March 2024) interleaves Transformer and Mamba
    # layers, so it branches off Mamba (2023) -- Mamba-2 only appeared in May
    # 2024 -- with a merge arrow in from the transformer lane.
    ax.plot([STATESPACE_COMMITS[1][0], HYBRID_COMMIT[0]], [Y_STATESPACE, Y_HYBRID],
             color=C_HYBRID, linewidth=2.0, alpha=0.8, zorder=1)
    ax.plot([TRANSFORMER_COMMITS[2][0], HYBRID_COMMIT[0]], [Y_TRANSFORMER, Y_HYBRID],
             color=C_HYBRID, linewidth=2.0, alpha=0.8, zorder=1)

    def draw_commit(x, label, updown, y, color, size=180):
        ax.scatter([x], [y], s=size, color=color, edgecolors=PAPER, linewidths=1.6, zorder=2)
        if updown == 0:  # label to the right (the hybrid lane sits between the others)
            ax.annotate(label, (x, y), xytext=(14, 0), textcoords="offset points",
                        ha="left", va="center", fontsize=8.6, color=INK_SOFT,
                        fontweight="600")
            return
        ax.annotate(
            label, (x, y), xytext=(0, 15 * updown), textcoords="offset points",
            ha="center", va="bottom" if updown > 0 else "top",
            fontsize=8.6, color=INK_SOFT, fontweight="600", linespacing=1.3,
        )

    draw_commit(*ROOT, y=Y_TRANSFORMER, color=INK_SOFT, size=70)
    for x, label, ud in TRANSFORMER_COMMITS:
        draw_commit(x, label, ud, Y_TRANSFORMER, C_TRANSFORMER)
    for x, label, ud in STATESPACE_COMMITS:
        draw_commit(x, label, ud, Y_STATESPACE, C_STATESPACE)
    draw_commit(*HYBRID_COMMIT, y=Y_HYBRID, color=C_HYBRID, size=220)

    fig.text(
        0.03, 0.965, "transformer-atlas",
        fontsize=24, fontweight="700", color=INK, ha="left", va="top",
        family="monospace",
    )
    fig.text(
        0.03, 0.895,
        "A structured, hands-on map of the transformer architecture — "
        "from 2017 to today's frontier LLMs",
        fontsize=11.5, color=INK_SOFT, ha="left", va="top",
    )

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=C_TRANSFORMER,
               markeredgecolor=PAPER, markersize=9, label="Transformer lineage"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=C_STATESPACE,
               markeredgecolor=PAPER, markersize=9, label="State-space lineage"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=C_HYBRID,
               markeredgecolor=PAPER, markersize=9, label="Hybrid architectures"),
    ]
    fig.legend(
        handles=legend_handles, loc="lower center", ncol=3,
        fontsize=9.5, frameon=False, labelcolor=INK_SOFT,
        bbox_to_anchor=(0.5, 0.01), bbox_transform=fig.transFigure,
        columnspacing=1.6, handletextpad=0.6,
    )

    out = Path(__file__).resolve().parents[1] / "docs" / "social-preview.png"
    fig.savefig(str(out), dpi=200, facecolor=PAPER, bbox_inches=None)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
