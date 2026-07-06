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
    "font.size": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

C = {
    "black": "#222222",
    "gray": "#5F6368",
    "border": "#222222",
    "light_border": "#AEB7C2",
    "grid": "#DDE1E7",
    "blue": "#2F6FA5",
    "blue_l": "#E9F2FB",
    "orange": "#C76B2D",
    "orange_l": "#FFF0E3",
    "green": "#3F8C4D",
    "green_l": "#EAF6EA",
    "purple": "#7358A6",
    "purple_l": "#F1ECF8",
    "red": "#C92525",
    "red_l": "#FCEAEA",
    "photo": "#F3F4F6",
    "photo2": "#FFFFFF",
}

fig = plt.figure(figsize=(7.45, 5.32), dpi=450)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")


def txt(x, y, s, size=8, weight="normal", color=None, ha="center", va="center", z=20, **kw):
    ax.text(x, y, s, fontsize=size, fontweight=weight, color=color or C["black"], ha=ha, va=va, zorder=z, **kw)


def panel(x, y, w, h, label, title):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.018", facecolor="white", edgecolor=C["border"], linewidth=1.05, zorder=0)
    ax.add_patch(p)
    txt(x + 0.018, y + h + 0.021, label, size=9, weight="bold", ha="left")
    txt(x + w / 2, y + h - 0.032, title, size=10, weight="bold", color=C["red"])


def box(x, y, w, h, s="", fc="white", ec=None, lw=0.95, r=0.010, size=7.0, weight="normal", color=None, z=5, ls="solid"):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.005,rounding_size={r}", facecolor=fc, edgecolor=ec or C["border"], linewidth=lw, linestyle=ls, zorder=z)
    ax.add_patch(p)
    if s:
        txt(x + w / 2, y + h / 2, s, size=size, weight=weight, color=color, z=z + 1)
    return p


def rect(x, y, w, h, fc="white", ec=None, lw=0.9, z=5, ls="solid"):
    p = Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec or C["border"], linewidth=lw, linestyle=ls, zorder=z)
    ax.add_patch(p)
    return p


def arrow(x1, y1, x2, y2, color=None, lw=1.18, ms=8.5, z=15):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=ms, linewidth=lw, color=color or C["black"], connectionstyle="arc3,rad=0", zorder=z))


def label_box(x, y, s, color):
    txt(x, y, s, size=6.6, color=color, bbox=dict(boxstyle="round,pad=0.11", fc="white", ec="none", alpha=0.94))


def grid_rect(x, y, w, h):
    rect(x, y, w, h, fc="#F8FAFD", ec="#BBC4CE", lw=0.8, z=4)
    for k in range(1, 6):
        xx = x + w * k / 6
        yy = y + h * k / 6
        ax.add_line(Line2D([xx, xx], [y, y + h], color=C["grid"], lw=0.42, zorder=4))
        ax.add_line(Line2D([x, x + w], [yy, yy], color=C["grid"], lw=0.42, zorder=4))

# Geometry.
lx, rx = 0.052, 0.535
by, ty = 0.085, 0.575
pw, ph = 0.410, 0.355
panel(lx, ty, pw, ph, "a", "Simulation environment")
panel(rx, ty, pw, ph, "b", "RL policy setting")
panel(lx, by, pw, ph, "c", "Vision and state estimation")
panel(rx, by, pw, ph, "d", "Real-world rehabilitation platform")

# a) Simulation environment: task + CMD-DR.
sx, sy = lx, ty
grid_rect(sx + 0.035, sy + 0.078, 0.160, 0.160)
ax.plot([sx + 0.060, sx + 0.090, sx + 0.118, sx + 0.148, sx + 0.175], [sy + 0.110, sy + 0.150, sy + 0.135, sy + 0.185, sy + 0.205], color=C["blue"], lw=1.25, zorder=6)
ax.add_patch(Circle((sx + 0.060, sy + 0.110), 0.0105, facecolor=C["blue"], edgecolor="white", lw=0.4, zorder=7))
ax.add_patch(Circle((sx + 0.175, sy + 0.205), 0.0135, facecolor=C["orange"], edgecolor="white", lw=0.4, zorder=7))
ax.plot([sx + 0.060, sx + 0.175], [sy + 0.110, sy + 0.205], color=C["black"], lw=0.8, ls="--", zorder=6)
txt(sx + 0.119, sy + 0.166, r"$d_t$", size=7)
txt(sx + 0.115, sy + 0.052, "table-top pursuit task", size=6.7, color=C["gray"])
box(sx + 0.220, sy + 0.235, 0.150, 0.045, "CMD-DR virtual patient", fc=C["orange_l"], ec=C["orange"], size=6.8, weight="bold")
box(sx + 0.220, sy + 0.174, 0.150, 0.042, "intent: SPC / learned hand", fc="white", ec=C["orange"], size=6.2)
box(sx + 0.220, sy + 0.112, 0.150, 0.045, "motor layer: delay, noise,\nLPF, accel. limit", fc="white", ec=C["orange"], size=6.0)
box(sx + 0.220, sy + 0.050, 0.150, 0.043, "ZPD reward + boundary", fc=C["purple_l"], ec=C["purple"], size=6.2)
arrow(sx + 0.295, sy + 0.174, sx + 0.295, sy + 0.157, color=C["orange"], lw=0.9, ms=7)
arrow(sx + 0.295, sy + 0.112, sx + 0.295, sy + 0.093, color=C["purple"], lw=0.9, ms=7)

# b) RL policy setting.
px, py = rx, ty
box(px + 0.045, py + 0.215, 0.105, 0.060, "Observation\n$O_t$", fc=C["blue_l"], ec=C["blue"], size=6.9)
box(px + 0.045, py + 0.125, 0.105, 0.060, "Reward\n$r_t$", fc=C["purple_l"], ec=C["purple"], size=6.9)
box(px + 0.245, py + 0.165, 0.105, 0.065, "Action\n$a_t$", fc=C["orange_l"], ec=C["orange"], size=6.9)
box(px + 0.180, py + 0.070, 0.145, 0.045, "PPO training", fc=C["green_l"], ec=C["green"], size=6.8)

# Generic neural network placeholder; detailed dual-stream / auxiliary heads are reserved for a separate figure.
box(px + 0.168, py + 0.145, 0.060, 0.120, fc="white", ec=C["blue"], lw=0.9, r=0.012)
box(px + 0.205, py + 0.110, 0.020, 0.028, "π", fc=C["blue_l"], ec=C["blue"], size=7.0)
txt(px + 0.198, py + 0.285, "policy network", size=7.1, weight="bold", color=C["blue"])
for xoff, ys in [
    (0.182, [0.165, 0.195, 0.225]),
    (0.198, [0.158, 0.188, 0.218, 0.248]),
    (0.215, [0.175, 0.220]),
]:
    for yy in ys:
        ax.add_patch(Circle((px + xoff, py + yy), 0.0065, facecolor="white", edgecolor=C["blue"], lw=0.65, zorder=7))
for y1 in [0.165, 0.195, 0.225]:
    for y2 in [0.158, 0.188, 0.218, 0.248]:
        ax.add_line(Line2D([px + 0.188, px + 0.192], [py + y1, py + y2], color="#B5CBE2", lw=0.30, zorder=6))
for y1 in [0.158, 0.188, 0.218, 0.248]:
    for y2 in [0.175, 0.220]:
        ax.add_line(Line2D([px + 0.204, px + 0.209], [py + y1, py + y2], color="#B5CBE2", lw=0.30, zorder=6))
arrow(px + 0.150, py + 0.245, px + 0.168, py + 0.225, color=C["blue"], lw=0.9, ms=7)
arrow(px + 0.228, py + 0.200, px + 0.245, py + 0.198, color=C["orange"], lw=0.9, ms=7)
arrow(px + 0.150, py + 0.155, px + 0.180, py + 0.095, color=C["purple"], lw=0.9, ms=7)
arrow(px + 0.253, py + 0.115, px + 0.228, py + 0.145, color=C["green"], lw=0.9, ms=7)

# c) Vision/state estimation.
vx, vy = lx, by
rect(vx + 0.035, vy + 0.086, 0.160, 0.168, fc=C["photo"], ec="#BBC4CE", lw=0.8)
rect(vx + 0.056, vy + 0.112, 0.035, 0.026, fc="none", ec=C["blue"], lw=0.9)
rect(vx + 0.118, vy + 0.170, 0.042, 0.030, fc="none", ec=C["orange"], lw=0.9)
ax.add_patch(Circle((vx + 0.074, vy + 0.125), 0.0095, facecolor=C["blue"], edgecolor="white", lw=0.4, zorder=7))
ax.add_patch(Circle((vx + 0.139, vy + 0.185), 0.0125, facecolor=C["orange"], edgecolor="white", lw=0.4, zorder=7))
ax.add_patch(Circle((vx + 0.113, vy + 0.225), 0.0100, facecolor=C["green"], edgecolor="white", lw=0.4, zorder=7))
txt(vx + 0.115, vy + 0.060, "camera frame / detection overlay", size=6.5, color=C["gray"])
box(vx + 0.230, vy + 0.235, 0.135, 0.043, "async vision thread", fc=C["green_l"], ec=C["green"], size=6.5, weight="bold")
box(vx + 0.230, vy + 0.178, 0.135, 0.040, "YOLO microrobot", fc="white", ec=C["green"], size=6.1)
box(vx + 0.230, vy + 0.124, 0.135, 0.040, "MediaPipe hand", fc="white", ec=C["green"], size=6.1)
box(vx + 0.230, vy + 0.060, 0.135, 0.045, "state buffer +\ndead reckoning", fc=C["blue_l"], ec=C["blue"], size=6.0)
arrow(vx + 0.195, vy + 0.205, vx + 0.230, vy + 0.205, color=C["green"], lw=0.9, ms=7)
arrow(vx + 0.298, vy + 0.124, vx + 0.298, vy + 0.105, color=C["blue"], lw=0.8, ms=6)

# d) Real-world platform with placeholders.
rx0, ry0 = rx, by
rect(rx0 + 0.035, ry0 + 0.088, 0.200, 0.183, fc=C["photo"], ec="#AEB7C2", lw=1.0, ls=(0, (4, 3)))
rect(rx0 + 0.052, ry0 + 0.110, 0.166, 0.137, fc="white", ec="#D3D8DF", lw=0.8)
ax.add_line(Line2D([rx0 + 0.075, rx0 + 0.205], [ry0 + 0.177, ry0 + 0.202], color="#C6CDD5", lw=4.0, solid_capstyle="round", zorder=6))
ax.add_patch(Circle((rx0 + 0.134, ry0 + 0.192), 0.016, facecolor="#D9DEE5", edgecolor="#B8C0CA", lw=0.7, zorder=7))
txt(rx0 + 0.135, ry0 + 0.177, "setup photo\nplaceholder", size=7.0, weight="bold", color="#7A8088")
rect(rx0 + 0.255, ry0 + 0.090, 0.105, 0.070, fc=C["photo"], ec="#AEB7C2", lw=0.9, ls=(0, (4, 3)))
txt(rx0 + 0.308, ry0 + 0.125, "workspace\ninset", size=6.2, weight="bold", color="#7A8088")
box(rx0 + 0.255, ry0 + 0.230, 0.105, 0.046, "20 Hz\ncontrol thread", fc=C["blue_l"], ec=C["blue"], size=6.1)
box(rx0 + 0.255, ry0 + 0.176, 0.105, 0.040, "UR10 RTDE", fc=C["green_l"], ec=C["green"], size=6.1)
box(rx0 + 0.255, ry0 + 0.037, 0.105, 0.040, "magnetic\nactuation", fc=C["orange_l"], ec=C["orange"], size=5.9)
arrow(rx0 + 0.308, ry0 + 0.230, rx0 + 0.308, ry0 + 0.216, color=C["blue"], lw=0.8, ms=6)
arrow(rx0 + 0.308, ry0 + 0.176, rx0 + 0.308, ry0 + 0.160, color=C["green"], lw=0.8, ms=6)
arrow(rx0 + 0.255, ry0 + 0.057, rx0 + 0.235, ry0 + 0.130, color=C["orange"], lw=0.8, ms=6)
txt(rx0 + 0.135, ry0 + 0.060, "annotated real setup to be inserted", size=6.4, color=C["gray"])

# Inter-panel connectors, strictly straight.
arrow(lx + pw, ty + 0.248, rx, ty + 0.248, color=C["blue"], lw=1.2, ms=8.5)
label_box(0.500, ty + 0.270, r"$O_t, r_t$", C["blue"])
arrow(rx, ty + 0.176, lx + pw, ty + 0.176, color=C["orange"], lw=1.2, ms=8.5)
label_box(0.500, ty + 0.153, r"$a_t$", C["orange"])
arrow(rx + pw / 2, ty, rx + pw / 2, by + ph, color=C["green"], lw=1.2, ms=8.5)
label_box(rx + pw / 2 + 0.040, 0.506, "trained policy", C["green"])
arrow(rx, by + 0.245, lx + pw, by + 0.245, color=C["green"], lw=1.2, ms=8.5)
label_box(0.500, by + 0.267, "image feed", C["green"])
arrow(lx + pw, by + 0.142, rx, by + 0.142, color=C["blue"], lw=1.2, ms=8.5)
label_box(0.500, by + 0.119, r"real $O_t$", C["blue"])
arrow(lx + pw / 2, by + ph, lx + pw / 2, ty, color=C["purple"], lw=1.05, ms=8)
label_box(lx + pw / 2 - 0.050, 0.506, "same task", C["purple"])

# Legend.
ax.add_patch(Circle((0.055, 0.963), 0.006, facecolor=C["blue"], edgecolor="none"))
txt(0.070, 0.963, "robot / microrobot", size=6.4, color=C["gray"], ha="left")
ax.add_patch(Circle((0.185, 0.963), 0.006, facecolor=C["orange"], edgecolor="none"))
txt(0.200, 0.963, "patient hand", size=6.4, color=C["gray"], ha="left")

for path in [
    OUT_DIR / "fig_overview_system_framework_orthogonal_v5.png",
    DRAFT_DIR / "fig_overview_system_framework_orthogonal_v5_draft.png",
]:
    fig.savefig(path, dpi=450, bbox_inches="tight", pad_inches=0.035, facecolor="white")
fig.savefig(OUT_DIR / "fig_overview_system_framework_orthogonal_v5.pdf", bbox_inches="tight", pad_inches=0.035, facecolor="white")
fig.savefig(OUT_DIR / "fig_overview_system_framework_orthogonal_v5.svg", bbox_inches="tight", pad_inches=0.035, facecolor="white")
plt.close(fig)

print((OUT_DIR / "fig_overview_system_framework_orthogonal_v5.png").as_posix())
print((OUT_DIR / "fig_overview_system_framework_orthogonal_v5.pdf").as_posix())
print((OUT_DIR / "fig_overview_system_framework_orthogonal_v5.svg").as_posix())
print((DRAFT_DIR / "fig_overview_system_framework_orthogonal_v5_draft.png").as_posix())
