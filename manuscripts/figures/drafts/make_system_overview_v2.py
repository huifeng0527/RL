from pathlib import Path as FilePath
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch
from matplotlib.lines import Line2D

OUT_DIR = FilePath("manuscripts/figures/paper_ready")
DRAFT_DIR = FilePath("manuscripts/figures/drafts")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DRAFT_DIR.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

C = {
    "ink": "#232323",
    "muted": "#60656F",
    "line": "#AEB5BE",
    "paper": "#FFFFFF",
    "photo": "#F3F4F6",
    "photo_edge": "#B8BEC7",
    "blue": "#376DA3",
    "blue_l": "#EAF2FB",
    "orange": "#BF6B2E",
    "orange_l": "#FFF1E5",
    "green": "#4D8B4A",
    "green_l": "#EAF6EA",
    "purple": "#7057A5",
    "purple_l": "#F2EEF9",
    "gray_l": "#F8F9FB",
}

fig = plt.figure(figsize=(7.45, 4.25), dpi=450)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")


def text(x, y, s, size=8, weight="normal", color=None, ha="center", va="center", z=20, **kw):
    ax.text(x, y, s, fontsize=size, fontweight=weight, color=color or C["ink"], ha=ha, va=va, zorder=z, **kw)


def box(x, y, w, h, s="", fc="white", ec=None, lw=1.0, r=0.018, size=8, weight="normal", color=None, ls="solid", z=5):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.008,rounding_size={r}",
        facecolor=fc,
        edgecolor=ec or C["line"],
        linewidth=lw,
        linestyle=ls,
        zorder=z,
    )
    ax.add_patch(p)
    if s:
        text(x + w / 2, y + h / 2, s, size=size, weight=weight, color=color, z=z + 2)
    return p


def arrow(x1, y1, x2, y2, color=None, lw=1.15, rad=0.0, ms=9, ls="solid", z=10):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>",
        mutation_scale=ms,
        linewidth=lw,
        color=color or C["muted"],
        linestyle=ls,
        connectionstyle=f"arc3,rad={rad}",
        zorder=z,
    ))

# One integrated canvas with three soft functional zones.
box(0.035, 0.105, 0.288, 0.790, fc="#FAFAFB", ec="#D9DDE3", lw=0.9, r=0.023, z=0)
box(0.365, 0.650, 0.565, 0.220, fc="#FBF8FF", ec="#E2DAF0", lw=0.9, r=0.025, z=0)
box(0.365, 0.115, 0.565, 0.220, fc="#F7FBF7", ec="#D8E8D6", lw=0.9, r=0.025, z=0)
text(0.180, 0.855, "Physical platform", size=9.2, weight="bold")
text(0.648, 0.838, "Simulation training", size=9.2, weight="bold", color=C["purple"])
text(0.648, 0.303, "Physical deployment", size=9.2, weight="bold", color=C["green"])

# Empty photograph placeholder.
px, py, pw, ph = 0.067, 0.338, 0.224, 0.405
ax.add_patch(Rectangle((px, py), pw, ph, facecolor=C["photo"], edgecolor=C["photo_edge"], linewidth=1.1, linestyle=(0, (4, 3)), zorder=4))
ax.add_patch(Rectangle((px + 0.018, py + 0.025), pw - 0.036, ph - 0.050, facecolor="white", edgecolor="#D9DDE3", linewidth=0.8, zorder=5))
# Minimal silhouette so the blank region still reads as a photo slot.
ax.add_patch(Rectangle((px + 0.055, py + 0.095), pw - 0.110, 0.108, facecolor="#EEF1F4", edgecolor="#D0D5DB", linewidth=0.7, zorder=6))
ax.add_line(Line2D([px + 0.045, px + 0.195], [py + 0.225, py + 0.250], color="#C4CAD2", lw=4.0, solid_capstyle="round", zorder=6))
ax.add_patch(Circle((px + 0.114, py + 0.239), 0.016, facecolor="#D4D9E0", edgecolor="#B9C0C8", lw=0.7, zorder=7))
text(px + pw / 2, py + ph / 2 + 0.033, "platform photograph", size=8.0, weight="bold", color="#7A8088")
text(px + pw / 2, py + ph / 2 + 0.005, "to be inserted", size=7.2, color="#7A8088")
text(px + pw / 2, py - 0.028, "UR10 + magnetic microrobot + patient hand", size=6.9, color=C["muted"])

# Task abstraction card on the same physical side.
box(0.073, 0.165, 0.212, 0.105, fc="white", ec="#D2D6DC", lw=0.9, r=0.016, z=4)
text(0.179, 0.242, "ZPD task objective", size=7.3, weight="bold")
ax.add_patch(Circle((0.131, 0.197), 0.0125, facecolor=C["blue"], edgecolor="white", lw=0.5, zorder=9))
ax.add_patch(Circle((0.225, 0.197), 0.0150, facecolor=C["orange"], edgecolor="white", lw=0.5, zorder=9))
ax.plot([0.131, 0.225], [0.197, 0.197], color=C["ink"], lw=0.8, ls="--", zorder=8)
text(0.178, 0.215, r"$d_t$", size=8.0)
text(0.179, 0.178, "maintain target challenge", size=6.5, color=C["muted"])

# Central shared state and policy form the visual spine.
box(0.405, 0.440, 0.230, 0.135, fc="white", ec="#C9D0D9", lw=1.1, r=0.022, z=5)
text(0.520, 0.548, "shared interaction state", size=8.4, weight="bold")
text(0.520, 0.522, r"$s_t$: position, boundary, motion history", size=6.8, color=C["muted"])
ax.add_patch(Rectangle((0.435, 0.462), 0.170, 0.035, facecolor="#F9FAFB", edgecolor="#D6DAE0", lw=0.7, zorder=6))
ax.add_patch(Circle((0.470, 0.480), 0.0088, facecolor=C["blue"], edgecolor="white", lw=0.4, zorder=8))
ax.add_patch(Circle((0.570, 0.480), 0.0105, facecolor=C["orange"], edgecolor="white", lw=0.4, zorder=8))
ax.plot([0.470, 0.570], [0.480, 0.480], color=C["ink"], lw=0.7, ls="--", zorder=7)
text(0.520, 0.493, r"$d_t$", size=6.8)

box(0.720, 0.440, 0.165, 0.135, fc=C["blue_l"], ec=C["blue"], lw=1.2, r=0.022, z=6)
text(0.802, 0.535, "adaptive robot policy", size=8.2, weight="bold", color=C["blue"])
text(0.802, 0.506, "trained in simulation", size=6.8, color=C["muted"])
text(0.802, 0.480, "deployed at control rate", size=6.8, color=C["muted"])

# State-policy bidirectional interaction.
arrow(0.635, 0.520, 0.720, 0.520, color=C["blue"], lw=1.25)
text(0.675, 0.542, "observation", size=6.3, color=C["blue"])
arrow(0.720, 0.475, 0.635, 0.475, color=C["blue"], lw=1.05, rad=-0.12)
text(0.674, 0.454, "action", size=6.3, color=C["blue"])
arrow(0.292, 0.540, 0.405, 0.520, color=C["muted"], lw=1.05)
text(0.347, 0.555, "same task abstraction", size=6.2, color=C["muted"], rotation=-8)

# Training loop above the spine.
box(0.405, 0.720, 0.150, 0.070, "virtual patient\npopulation", fc=C["orange_l"], ec=C["orange"], lw=1.0, r=0.016, size=7.0, z=5)
box(0.585, 0.720, 0.125, 0.070, "interaction\nsimulator", fc="white", ec="#B8BEC7", lw=1.0, r=0.016, size=7.0, z=5)
box(0.755, 0.720, 0.115, 0.070, "ZPD\nfeedback", fc=C["purple_l"], ec=C["purple"], lw=1.0, r=0.016, size=7.0, z=5)
arrow(0.555, 0.755, 0.585, 0.755, color=C["orange"], lw=1.15)
arrow(0.710, 0.755, 0.755, 0.755, color=C["purple"], lw=1.15)
arrow(0.650, 0.720, 0.565, 0.575, color=C["muted"], lw=1.0, rad=0.05)
arrow(0.812, 0.720, 0.807, 0.575, color=C["purple"], lw=1.05, rad=-0.04)
text(0.760, 0.642, "policy update", size=6.3, color=C["purple"], rotation=-28)

# Physical deployment loop below the spine.
box(0.405, 0.185, 0.118, 0.065, "camera\nstream", fc="white", ec="#B8BEC7", lw=1.0, r=0.015, size=6.9, z=5)
box(0.555, 0.185, 0.130, 0.065, "state\nestimation", fc="white", ec="#B8BEC7", lw=1.0, r=0.015, size=6.9, z=5)
box(0.730, 0.180, 0.145, 0.075, "UR10\ncontrol", fc=C["green_l"], ec=C["green"], lw=1.1, r=0.015, size=7.1, z=5)
arrow(0.523, 0.218, 0.555, 0.218, color=C["green"], lw=1.15)
arrow(0.685, 0.218, 0.730, 0.218, color=C["green"], lw=1.15)
arrow(0.802, 0.440, 0.802, 0.255, color=C["blue"], lw=1.10)
arrow(0.405, 0.218, 0.286, 0.372, color=C["green"], lw=1.0, rad=0.12)
arrow(0.730, 0.205, 0.292, 0.410, color=C["green"], lw=1.0, rad=0.08)
text(0.355, 0.300, "vision feedback", size=6.1, color=C["green"], rotation=30)
text(0.520, 0.335, "robot actuation", size=6.1, color=C["green"], rotation=18)

# Compact legend.
ax.add_patch(Circle((0.050, 0.925), 0.0060, facecolor=C["blue"], edgecolor="none", zorder=20))
text(0.064, 0.925, "robot / target", size=6.3, color=C["muted"], ha="left")
ax.add_patch(Circle((0.168, 0.925), 0.0060, facecolor=C["orange"], edgecolor="none", zorder=20))
text(0.182, 0.925, "patient hand", size=6.3, color=C["muted"], ha="left")

for path in [OUT_DIR / "fig_overview_system_framework.png", DRAFT_DIR / "fig_overview_system_framework_v2_draft.png"]:
    fig.savefig(path, dpi=450, bbox_inches="tight", pad_inches=0.035, facecolor="white")
fig.savefig(OUT_DIR / "fig_overview_system_framework.pdf", bbox_inches="tight", pad_inches=0.035, facecolor="white")
fig.savefig(OUT_DIR / "fig_overview_system_framework.svg", bbox_inches="tight", pad_inches=0.035, facecolor="white")
plt.close(fig)

print((OUT_DIR / "fig_overview_system_framework.png").as_posix())
print((OUT_DIR / "fig_overview_system_framework.pdf").as_posix())
print((OUT_DIR / "fig_overview_system_framework.svg").as_posix())
print((DRAFT_DIR / "fig_overview_system_framework_v2_draft.png").as_posix())
