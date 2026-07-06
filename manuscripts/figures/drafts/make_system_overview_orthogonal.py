from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch
from matplotlib.lines import Line2D

OUT_DIR = Path("manuscripts/figures/paper_ready")
DRAFT_DIR = Path("manuscripts/figures/drafts")
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
    "black": "#222222",
    "gray": "#5E5E5E",
    "border": "#333333",
    "light_border": "#8A8A8A",
    "blue": "#2E66A1",
    "blue_l": "#EAF2FB",
    "orange": "#C96D2D",
    "orange_l": "#FFF0E2",
    "green": "#3F8B4C",
    "green_l": "#EAF6EA",
    "purple": "#6E55A3",
    "purple_l": "#F1ECF8",
    "red": "#C7352C",
    "red_l": "#FCEAEA",
    "yellow_l": "#FFF8D9",
    "panel_l": "#FFFFFF",
    "photo": "#F3F4F6",
}

fig = plt.figure(figsize=(7.35, 5.05), dpi=450)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")


def txt(x, y, s, size=8, weight="normal", color=None, ha="center", va="center", z=20, **kw):
    ax.text(x, y, s, fontsize=size, fontweight=weight, color=color or C["black"], ha=ha, va=va, zorder=z, **kw)


def box(x, y, w, h, s="", fc="white", ec=None, lw=1.0, r=0.014, size=8, weight="normal", color=None, z=5, ls="solid"):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.006,rounding_size={r}",
        facecolor=fc,
        edgecolor=ec or C["border"],
        linewidth=lw,
        linestyle=ls,
        zorder=z,
    )
    ax.add_patch(p)
    if s:
        txt(x + w / 2, y + h / 2, s, size=size, weight=weight, color=color, z=z + 1)
    return p


def rect(x, y, w, h, fc="white", ec=None, lw=1.0, z=5, ls="solid"):
    p = Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec or C["border"], linewidth=lw, linestyle=ls, zorder=z)
    ax.add_patch(p)
    return p


def arrow(x1, y1, x2, y2, color=None, lw=1.25, ms=9, z=12):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>",
        mutation_scale=ms,
        linewidth=lw,
        color=color or C["black"],
        connectionstyle="arc3,rad=0",
        zorder=z,
    ))


def line(points, color=None, lw=1.15, z=10):
    xs, ys = zip(*points)
    ax.add_line(Line2D(xs, ys, color=color or C["black"], linewidth=lw, zorder=z))


def elbow(points, color=None, lw=1.15, ms=9, z=10):
    if len(points) > 2:
        line(points[:-1], color=color, lw=lw, z=z)
    arrow(points[-2][0], points[-2][1], points[-1][0], points[-1][1], color=color, lw=lw, ms=ms, z=z+1)

# Panel geometry: strict 2 x 2 grid.
left_x, right_x = 0.055, 0.535
bottom_y, top_y = 0.095, 0.555
panel_w, panel_h = 0.410, 0.365

panels = {
    "sim": (left_x, top_y, panel_w, panel_h),
    "policy": (right_x, top_y, panel_w, panel_h),
    "vision": (left_x, bottom_y, panel_w, panel_h),
    "real": (right_x, bottom_y, panel_w, panel_h),
}

# Big panels with aligned edges.
box(*panels["sim"], fc="#FCFCFC", ec=C["border"], lw=1.05, r=0.018, z=0)
box(*panels["policy"], fc="#FCFCFC", ec=C["border"], lw=1.05, r=0.018, z=0)
box(*panels["vision"], fc="#FCFCFC", ec=C["border"], lw=1.05, r=0.018, z=0)
box(*panels["real"], fc="#FCFCFC", ec=C["border"], lw=1.05, r=0.018, z=0)

# Panel labels and titles.
txt(left_x + 0.020, top_y + panel_h + 0.018, "a", size=9, weight="bold", ha="left")
txt(right_x + 0.020, top_y + panel_h + 0.018, "b", size=9, weight="bold", ha="left")
txt(left_x + 0.020, bottom_y + panel_h + 0.018, "c", size=9, weight="bold", ha="left")
txt(right_x + 0.020, bottom_y + panel_h + 0.018, "d", size=9, weight="bold", ha="left")
txt(left_x + panel_w / 2, top_y + panel_h - 0.035, "Simulation environment", size=10, weight="bold", color=C["red"])
txt(right_x + panel_w / 2, top_y + panel_h - 0.035, "Policy learning", size=10, weight="bold", color=C["red"])
txt(left_x + panel_w / 2, bottom_y + panel_h - 0.035, "Vision and state estimation", size=10, weight="bold", color=C["red"])
txt(right_x + panel_w / 2, bottom_y + panel_h - 0.035, "Real-world rehabilitation platform", size=10, weight="bold", color=C["red"])

# a) Simulation environment.
sx, sy, sw, sh = panels["sim"]
rect(sx + 0.035, sy + 0.070, 0.195, 0.215, fc="#F7FAFD", ec="#BFC7D0", lw=0.8)
# grid
for i in range(1, 7):
    ax.add_line(Line2D([sx + 0.035 + i * 0.195 / 7, sx + 0.035 + i * 0.195 / 7], [sy + 0.070, sy + 0.285], color="#E5E8ED", lw=0.45, zorder=3))
    ax.add_line(Line2D([sx + 0.035, sx + 0.230], [sy + 0.070 + i * 0.215 / 7, sy + 0.070 + i * 0.215 / 7], color="#E5E8ED", lw=0.45, zorder=3))
# path and objects
ax.plot([sx + 0.065, sx + 0.095, sx + 0.125, sx + 0.155, sx + 0.195], [sy + 0.120, sy + 0.175, sy + 0.155, sy + 0.210, sy + 0.235], color=C["blue"], lw=1.2, zorder=5)
ax.add_patch(Circle((sx + 0.065, sy + 0.120), 0.011, facecolor=C["blue"], edgecolor="white", lw=0.4, zorder=6))
ax.add_patch(Circle((sx + 0.195, sy + 0.235), 0.014, facecolor=C["orange"], edgecolor="white", lw=0.4, zorder=6))
ax.plot([sx + 0.065, sx + 0.195], [sy + 0.120, sy + 0.235], color=C["black"], lw=0.8, ls="--", zorder=5)
txt(sx + 0.132, sy + 0.190, r"$d_t$", size=7.0)
txt(sx + 0.132, sy + 0.045, "robot target and patient hand\nin the same ZPD task", size=6.7, color=C["gray"])
# observation/reward list
box(sx + 0.255, sy + 0.185, 0.125, 0.090, "Observation\n$O_t$", fc=C["blue_l"], ec=C["blue"], lw=0.9, r=0.012, size=7.0)
box(sx + 0.255, sy + 0.075, 0.125, 0.080, "Reward\n$r_t$", fc=C["purple_l"], ec=C["purple"], lw=0.9, r=0.012, size=7.0)
txt(sx + 0.318, sy + 0.158, "ZPD band\nboundaries\nsmoothness", size=6.2, color=C["gray"])

# b) Policy learning.
px, py, pw, ph = panels["policy"]
# policy block
box(px + 0.060, py + 0.105, 0.150, 0.185, fc=C["blue_l"], ec=C["blue"], lw=1.0, r=0.014)
txt(px + 0.135, py + 0.265, "adaptive\nrobot policy", size=7.6, weight="bold", color=C["blue"])
# simple neural net
for layer, xoff, n in [(0, 0.088, 3), (1, 0.135, 4), (2, 0.182, 2)]:
    ys = [py + 0.135 + i * 0.032 for i in range(n)]
    for yy in ys:
        ax.add_patch(Circle((px + xoff, yy), 0.009, facecolor="white", edgecolor=C["blue"], lw=0.8, zorder=7))
# connecting faint lines
for y1 in [py + 0.135, py + 0.167, py + 0.199]:
    for y2 in [py + 0.135, py + 0.167, py + 0.199, py + 0.231]:
        ax.add_line(Line2D([px + 0.097, px + 0.126], [y1, y2], color="#B9CBE0", lw=0.35, zorder=6))
for y1 in [py + 0.135, py + 0.167, py + 0.199, py + 0.231]:
    for y2 in [py + 0.151, py + 0.199]:
        ax.add_line(Line2D([px + 0.144, px + 0.173], [y1, y2], color="#B9CBE0", lw=0.35, zorder=6))
box(px + 0.250, py + 0.225, 0.115, 0.070, "virtual patient\npool", fc=C["orange_l"], ec=C["orange"], lw=0.9, r=0.012, size=6.8)
box(px + 0.250, py + 0.135, 0.115, 0.060, "PFSP\nsampling", fc=C["purple_l"], ec=C["purple"], lw=0.9, r=0.012, size=6.8)
box(px + 0.250, py + 0.055, 0.115, 0.050, "PPO update", fc=C["green_l"], ec=C["green"], lw=0.9, r=0.012, size=6.8)
arrow(px + 0.250, py + 0.260, px + 0.210, py + 0.235, color=C["orange"], lw=1.0, ms=8)
arrow(px + 0.250, py + 0.165, px + 0.210, py + 0.195, color=C["purple"], lw=1.0, ms=8)
arrow(px + 0.250, py + 0.080, px + 0.210, py + 0.128, color=C["green"], lw=1.0, ms=8)

# c) Vision and state estimation.
vx, vy, vw, vh = panels["vision"]
# image feed placeholder
rect(vx + 0.035, vy + 0.085, 0.160, 0.165, fc="#F5F5F5", ec="#BFC7D0", lw=0.8)
for i, (dx, dy, col) in enumerate([(0.040, 0.035, C["blue"]), (0.100, 0.090, C["orange"]), (0.070, 0.125, C["green"])]):
    ax.add_patch(Circle((vx + 0.035 + dx, vy + 0.085 + dy), 0.010, facecolor=col, edgecolor="white", lw=0.4, zorder=6))
# detection boxes
rect(vx + 0.060, vy + 0.112, 0.035, 0.028, fc="none", ec=C["blue"], lw=0.9)
rect(vx + 0.122, vy + 0.170, 0.038, 0.030, fc="none", ec=C["orange"], lw=0.9)
txt(vx + 0.115, vy + 0.060, "camera frame", size=6.8, color=C["gray"])
box(vx + 0.235, vy + 0.195, 0.125, 0.065, "YOLO +\nMediaPipe", fc=C["green_l"], ec=C["green"], lw=0.9, r=0.012, size=6.8)
box(vx + 0.235, vy + 0.095, 0.125, 0.065, "state\nestimator", fc=C["blue_l"], ec=C["blue"], lw=0.9, r=0.012, size=6.8)
arrow(vx + 0.195, vy + 0.200, vx + 0.235, vy + 0.225, color=C["green"], lw=1.0, ms=8)
arrow(vx + 0.298, vy + 0.195, vx + 0.298, vy + 0.160, color=C["blue"], lw=1.0, ms=8)
txt(vx + 0.298, vy + 0.045, "state = positions + velocity\nwith dead-reckoning gaps", size=6.3, color=C["gray"])

# d) Real-world platform placeholder.
rx, ry, rw, rh = panels["real"]
rect(rx + 0.035, ry + 0.075, 0.210, 0.205, fc=C["photo"], ec="#AEB7C2", lw=1.0, ls=(0, (4, 3)))
rect(rx + 0.055, ry + 0.098, 0.170, 0.160, fc="white", ec="#D3D8DF", lw=0.8)
ax.add_line(Line2D([rx + 0.078, rx + 0.205], [ry + 0.175, ry + 0.198], color="#C6CDD5", lw=4.0, solid_capstyle="round", zorder=6))
ax.add_patch(Circle((rx + 0.136, ry + 0.190), 0.016, facecolor="#D9DEE5", edgecolor="#B8C0CA", lw=0.7, zorder=7))
txt(rx + 0.140, ry + 0.185, "platform\nphoto slot", size=7.3, weight="bold", color="#7A8088")
txt(rx + 0.140, ry + 0.050, "insert annotated real setup photo", size=6.7, color=C["gray"])
box(rx + 0.275, ry + 0.205, 0.100, 0.055, "UR10\ncontroller", fc=C["green_l"], ec=C["green"], lw=0.9, r=0.012, size=6.7)
box(rx + 0.275, ry + 0.110, 0.100, 0.055, "magnetic\nactuation", fc=C["orange_l"], ec=C["orange"], lw=0.9, r=0.012, size=6.7)
arrow(rx + 0.325, ry + 0.205, rx + 0.325, ry + 0.165, color=C["green"], lw=1.0, ms=8)
arrow(rx + 0.275, ry + 0.138, rx + 0.245, ry + 0.158, color=C["orange"], lw=1.0, ms=8)

# Inter-panel straight connectors only.
# Top: simulation <-> policy
arrow(left_x + panel_w, top_y + 0.265, right_x, top_y + 0.265, color=C["blue"], lw=1.25, ms=9)
txt(0.500, top_y + 0.288, "observation $O_t$, reward $r_t$", size=6.8, color=C["blue"])
arrow(right_x, top_y + 0.205, left_x + panel_w, top_y + 0.205, color=C["orange"], lw=1.25, ms=9)
txt(0.500, top_y + 0.184, "action $a_t$", size=6.8, color=C["orange"])

# Right: policy -> real-world system
arrow(right_x + panel_w / 2, top_y, right_x + panel_w / 2, bottom_y + panel_h, color=C["green"], lw=1.25, ms=9)
txt(right_x + panel_w / 2 + 0.018, 0.500, "control command", size=6.8, color=C["green"], rotation=90)

# Bottom: real-world image feed -> vision
arrow(right_x, bottom_y + 0.250, left_x + panel_w, bottom_y + 0.250, color=C["green"], lw=1.25, ms=9)
txt(0.500, bottom_y + 0.272, "image feed", size=6.8, color=C["green"])
arrow(left_x + panel_w, bottom_y + 0.145, right_x, bottom_y + 0.145, color=C["blue"], lw=1.25, ms=9)
txt(0.500, bottom_y + 0.124, "real-world observation $O_t$", size=6.8, color=C["blue"])

# Left side: physical task ↔ simulation abstraction
arrow(left_x + panel_w / 2, bottom_y + panel_h, left_x + panel_w / 2, top_y, color=C["purple"], lw=1.10, ms=8)
txt(left_x + panel_w / 2 - 0.018, 0.500, "same ZPD task", size=6.6, color=C["purple"], rotation=90)

# Compact legend.
ax.add_patch(Circle((0.058, 0.962), 0.006, facecolor=C["blue"], edgecolor="none"))
txt(0.072, 0.962, "robot / microrobot", size=6.4, color=C["gray"], ha="left")
ax.add_patch(Circle((0.185, 0.962), 0.006, facecolor=C["orange"], edgecolor="none"))
txt(0.199, 0.962, "patient hand", size=6.4, color=C["gray"], ha="left")

for path in [
    OUT_DIR / "fig_overview_system_framework_orthogonal.png",
    DRAFT_DIR / "fig_overview_system_framework_orthogonal_draft.png",
]:
    fig.savefig(path, dpi=450, bbox_inches="tight", pad_inches=0.035, facecolor="white")
fig.savefig(OUT_DIR / "fig_overview_system_framework_orthogonal.pdf", bbox_inches="tight", pad_inches=0.035, facecolor="white")
fig.savefig(OUT_DIR / "fig_overview_system_framework_orthogonal.svg", bbox_inches="tight", pad_inches=0.035, facecolor="white")
plt.close(fig)

print((OUT_DIR / "fig_overview_system_framework_orthogonal.png").as_posix())
print((OUT_DIR / "fig_overview_system_framework_orthogonal.pdf").as_posix())
print((OUT_DIR / "fig_overview_system_framework_orthogonal.svg").as_posix())
print((DRAFT_DIR / "fig_overview_system_framework_orthogonal_draft.png").as_posix())
