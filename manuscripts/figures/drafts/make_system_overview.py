from pathlib import Path as FilePath
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch, PathPatch
from matplotlib.path import Path as MplPath
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
    "ink": "#222222",
    "muted": "#666666",
    "line": "#A7AEB7",
    "photo": "#F3F4F6",
    "blue": "#356AA0",
    "blue_l": "#E8F1FB",
    "orange": "#C56A2D",
    "orange_l": "#FFF0E3",
    "green": "#4C8B4A",
    "green_l": "#EAF5E9",
    "purple": "#7557A8",
    "purple_l": "#F0ECF8",
}

fig = plt.figure(figsize=(7.45, 4.65), dpi=450)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")


def txt(x, y, s, size=8, weight="normal", color=None, ha="center", va="center", z=20, **kw):
    ax.text(x, y, s, fontsize=size, fontweight=weight, color=color or C["ink"], ha=ha, va=va, zorder=z, **kw)


def box(x, y, w, h, text="", fc="white", ec=None, lw=1.0, r=0.018, size=8, weight="normal", color=None, ls="solid", z=5):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.008,rounding_size={r}",
        facecolor=fc,
        edgecolor=ec or C["line"],
        linewidth=lw,
        linestyle=ls,
        zorder=z,
    )
    ax.add_patch(patch)
    if text:
        txt(x + w / 2, y + h / 2, text, size=size, weight=weight, color=color, z=z + 2)
    return patch


def arrow(x1, y1, x2, y2, color=None, lw=1.2, rad=0, ms=9, ls="solid", z=10):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>",
        mutation_scale=ms,
        color=color or C["muted"],
        linewidth=lw,
        linestyle=ls,
        connectionstyle=f"arc3,rad={rad}",
        zorder=z,
    ))


def curve(p0, p1, p2, p3, color=None, lw=1.2, z=8, ls="solid"):
    path = MplPath([p0, p1, p2, p3], [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
    ax.add_patch(PathPatch(path, facecolor="none", edgecolor=color or C["muted"], lw=lw, ls=ls, zorder=z))
    arrow(p2[0], p2[1], p3[0], p3[1], color=color, lw=lw, ms=8, z=z + 1)

# Soft integrated regions, not separated 1x3 panels.
box(0.035, 0.080, 0.300, 0.820, fc="#FAFAFB", ec="#D5D9DE", lw=0.9, r=0.020, z=0)
box(0.300, 0.565, 0.625, 0.300, fc="#FBF8FF", ec="#E0D8F0", lw=0.8, r=0.025, z=0)
box(0.300, 0.115, 0.625, 0.300, fc="#F7FBF7", ec="#D6E6D5", lw=0.8, r=0.025, z=0)

txt(0.500, 0.945, "Closed-loop framework for ZPD-regulated robot-assisted rehabilitation", size=10.2, weight="bold")
txt(0.185, 0.862, "Physical rehabilitation platform", size=9.0, weight="bold")
txt(0.612, 0.832, "Simulation-based policy learning", size=9.0, weight="bold", color=C["purple"])
txt(0.613, 0.382, "Real-time physical deployment", size=9.0, weight="bold", color=C["green"])

# Empty photo placeholder.
px, py, pw, ph = 0.065, 0.320, 0.240, 0.435
ax.add_patch(Rectangle((px, py), pw, ph, facecolor=C["photo"], edgecolor="#B8BEC7", linewidth=1.1, linestyle=(0, (4, 3)), zorder=4))
ax.add_patch(Rectangle((px + 0.018, py + 0.022), pw - 0.036, ph - 0.044, facecolor="white", edgecolor="#D7DADF", lw=0.8, zorder=5))
ax.add_patch(Rectangle((px + 0.042, py + 0.095), pw - 0.084, 0.115, facecolor="#EEF1F4", edgecolor="#D0D4DA", lw=0.7, zorder=6))
ax.add_line(Line2D([px + 0.055, px + 0.225], [py + 0.220, py + 0.245], color="#C4C9D0", lw=4.5, solid_capstyle="round", zorder=6))
ax.add_patch(Circle((px + 0.122, py + 0.236), 0.018, facecolor="#D5D9DE", edgecolor="#B8BEC7", lw=0.7, zorder=7))
txt(px + pw / 2, py + ph / 2 + 0.050, "platform photograph", size=8, weight="bold", color="#7A8088")
txt(px + pw / 2, py + ph / 2 + 0.018, "to be inserted", size=7.2, color="#7A8088")
txt(px + pw / 2, py - 0.030, "UR10 + magnetic microrobot + patient hand", size=7.0, color=C["muted"])

# Shared task variable.
box(0.062, 0.135, 0.245, 0.120, fc="white", ec="#D0D4DA", lw=0.9, r=0.015, z=4)
txt(0.184, 0.225, "shared task variable", size=7.4, weight="bold")
ax.add_patch(Circle((0.132, 0.177), 0.014, facecolor=C["blue"], edgecolor="white", lw=0.5, zorder=8))
ax.add_patch(Circle((0.224, 0.177), 0.017, facecolor=C["orange"], edgecolor="white", lw=0.5, zorder=8))
ax.plot([0.132, 0.224], [0.177, 0.177], color=C["ink"], ls="--", lw=0.9, zorder=7)
txt(0.178, 0.196, r"$d_t$", size=8.3)
txt(0.178, 0.152, "target challenge band", size=6.8, color=C["muted"])

# Central interaction state and policy.
box(0.375, 0.425, 0.255, 0.150, fc="white", ec="#CCD2DA", lw=1.1, r=0.020, z=5)
txt(0.503, 0.548, "ZPD-regulated interaction state", size=8.2, weight="bold")
ax.add_patch(Rectangle((0.405, 0.450), 0.195, 0.055, facecolor="#F9FAFB", edgecolor="#D2D6DC", lw=0.7, zorder=6))
ax.add_patch(Circle((0.452, 0.478), 0.010, facecolor=C["blue"], edgecolor="white", lw=0.4, zorder=8))
ax.add_patch(Circle((0.548, 0.478), 0.012, facecolor=C["orange"], edgecolor="white", lw=0.4, zorder=8))
ax.plot([0.452, 0.548], [0.478, 0.478], color=C["ink"], lw=0.8, ls="--", zorder=7)
txt(0.500, 0.493, r"$d_t$", size=7.2)
txt(0.503, 0.435, "positions, boundary, recent motion history", size=6.7, color=C["muted"])

box(0.705, 0.443, 0.165, 0.118, fc=C["blue_l"], ec=C["blue"], lw=1.2, r=0.018, z=6)
txt(0.787, 0.525, "adaptive robot policy", size=8.0, weight="bold", color=C["blue"])
txt(0.787, 0.494, "PPO inference", size=7.0, color=C["muted"])
txt(0.787, 0.470, "robot motion command", size=6.7, color=C["muted"])

# Simulation learning components.
box(0.360, 0.710, 0.160, 0.070, "virtual patient\npopulation", fc=C["orange_l"], ec=C["orange"], lw=1.1, r=0.015, size=7.2, z=5)
box(0.565, 0.710, 0.145, 0.070, "interaction\nsimulator", fc="white", ec="#B7BDC5", lw=1.0, r=0.015, size=7.2, z=5)
box(0.755, 0.705, 0.120, 0.080, "ZPD\nfeedback", fc=C["purple_l"], ec=C["purple"], lw=1.1, r=0.015, size=7.2, z=5)
box(0.472, 0.615, 0.255, 0.055, "opponent pool and curriculum sampling", fc="white", ec="#D5D0E8", lw=0.85, r=0.014, size=7.0, color=C["muted"], z=5)
arrow(0.520, 0.745, 0.565, 0.745, color=C["orange"], lw=1.2)
arrow(0.710, 0.745, 0.755, 0.745, color=C["purple"], lw=1.2)
arrow(0.815, 0.705, 0.812, 0.562, color=C["purple"], lw=1.15, rad=-0.08)
arrow(0.650, 0.710, 0.555, 0.575, color=C["muted"], lw=1.05, rad=0.08)
arrow(0.600, 0.615, 0.705, 0.538, color=C["purple"], lw=1.05, rad=-0.08)
txt(0.743, 0.630, "train / update", size=6.5, color=C["purple"], rotation=-24)

# Real deployment components.
box(0.370, 0.250, 0.120, 0.065, "camera\nstream", fc="white", ec="#B7BDC5", lw=1.0, r=0.014, size=7.0, z=5)
box(0.525, 0.250, 0.145, 0.065, "state\nestimation", fc="white", ec="#B7BDC5", lw=1.0, r=0.014, size=7.0, z=5)
box(0.705, 0.245, 0.165, 0.075, "fixed-rate\nUR10 control", fc=C["green_l"], ec=C["green"], lw=1.1, r=0.014, size=7.2, z=5)
arrow(0.490, 0.283, 0.525, 0.283, color=C["green"], lw=1.2)
arrow(0.670, 0.283, 0.705, 0.283, color=C["green"], lw=1.2)
arrow(0.787, 0.443, 0.787, 0.320, color=C["blue"], lw=1.15)
curve((0.430, 0.250), (0.325, 0.190), (0.245, 0.250), (0.230, 0.320), color=C["green"], lw=1.05)
arrow(0.705, 0.268, 0.305, 0.405, color=C["green"], lw=1.05, rad=0.10)
txt(0.397, 0.197, "visual feedback", size=6.4, color=C["green"], rotation=10)
txt(0.515, 0.353, "actuation", size=6.4, color=C["green"], rotation=13)

# Shared flow.
arrow(0.307, 0.520, 0.375, 0.510, color=C["muted"], lw=1.15)
arrow(0.630, 0.500, 0.705, 0.500, color=C["blue"], lw=1.25)
arrow(0.705, 0.475, 0.630, 0.470, color=C["blue"], lw=1.05, rad=-0.10)

# Legend.
legend_x, legend_y = 0.052, 0.925
ax.add_patch(Circle((legend_x, legend_y), 0.0065, facecolor=C["blue"], edgecolor="none", zorder=20))
txt(legend_x + 0.020, legend_y, "robot / target", size=6.5, color=C["muted"], ha="left")
ax.add_patch(Circle((legend_x + 0.145, legend_y), 0.0065, facecolor=C["orange"], edgecolor="none", zorder=20))
txt(legend_x + 0.163, legend_y, "patient hand", size=6.5, color=C["muted"], ha="left")

outputs = [
    OUT_DIR / "fig_overview_system_framework.png",
    DRAFT_DIR / "fig_overview_system_framework_draft.png",
]
for path in outputs:
    fig.savefig(path, dpi=450, bbox_inches="tight", pad_inches=0.035, facecolor="white")
fig.savefig(OUT_DIR / "fig_overview_system_framework.pdf", bbox_inches="tight", pad_inches=0.035, facecolor="white")
fig.savefig(OUT_DIR / "fig_overview_system_framework.svg", bbox_inches="tight", pad_inches=0.035, facecolor="white")
plt.close(fig)

for path in outputs:
    print(path.as_posix())
print((OUT_DIR / "fig_overview_system_framework.pdf").as_posix())
print((OUT_DIR / "fig_overview_system_framework.svg").as_posix())
