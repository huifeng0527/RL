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
    "grid": "#D6D9DE",
    "panel": "#FFFFFF",
    "blue": "#2F6FA5",
    "blue_l": "#E9F2FB",
    "orange": "#C76B2D",
    "orange_l": "#FFF0E3",
    "green": "#3F8C4D",
    "green_l": "#EAF6EA",
    "purple": "#7358A6",
    "purple_l": "#F1ECF8",
    "red": "#C92525",
    "photo": "#F3F4F6",
}

fig = plt.figure(figsize=(7.35, 5.20), dpi=450)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")


def txt(x, y, s, size=8, weight="normal", color=None, ha="center", va="center", z=20, **kw):
    ax.text(x, y, s, fontsize=size, fontweight=weight, color=color or C["black"], ha=ha, va=va, zorder=z, **kw)


def panel(x, y, w, h, label, title):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.018", facecolor=C["panel"], edgecolor=C["border"], linewidth=1.05, zorder=0)
    ax.add_patch(p)
    txt(x + 0.018, y + h + 0.021, label, size=9, weight="bold", ha="left")
    txt(x + w / 2, y + h - 0.034, title, size=10, weight="bold", color=C["red"])


def box(x, y, w, h, s="", fc="white", ec=None, lw=0.95, r=0.010, size=7.2, weight="normal", color=None, z=5, ls="solid"):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.005,rounding_size={r}", facecolor=fc, edgecolor=ec or C["border"], linewidth=lw, linestyle=ls, zorder=z)
    ax.add_patch(p)
    if s:
        txt(x + w / 2, y + h / 2, s, size=size, weight=weight, color=color, z=z + 1)
    return p


def rect(x, y, w, h, fc="white", ec=None, lw=0.9, z=5, ls="solid"):
    p = Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec or C["border"], linewidth=lw, linestyle=ls, zorder=z)
    ax.add_patch(p)
    return p


def arrow(x1, y1, x2, y2, color=None, lw=1.25, ms=9, z=15):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=ms, linewidth=lw, color=color or C["black"], connectionstyle="arc3,rad=0", zorder=z))


def label_box(x, y, s, color):
    txt(x, y, s, size=6.7, color=color, bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.92))

# Geometry with generous gutters.
lx, rx = 0.055, 0.535
by, ty = 0.090, 0.570
pw, ph = 0.405, 0.350
panel(lx, ty, pw, ph, "a", "Simulation environment")
panel(rx, ty, pw, ph, "b", "Policy learning")
panel(lx, by, pw, ph, "c", "Vision and state estimation")
panel(rx, by, pw, ph, "d", "Real-world rehabilitation platform")

# a) Simulation environment.
sx, sy = lx, ty
rect(sx + 0.045, sy + 0.085, 0.185, 0.190, fc="#F8FAFD", ec="#BBC4CE", lw=0.8)
for k in range(1, 6):
    x = sx + 0.045 + 0.185 * k / 6
    y = sy + 0.085 + 0.190 * k / 6
    ax.add_line(Line2D([x, x], [sy + 0.085, sy + 0.275], color="#E6E9EE", lw=0.45, zorder=4))
    ax.add_line(Line2D([sx + 0.045, sx + 0.230], [y, y], color="#E6E9EE", lw=0.45, zorder=4))
ax.plot([sx + 0.073, sx + 0.105, sx + 0.135, sx + 0.172, sx + 0.202], [sy + 0.116, sy + 0.166, sy + 0.145, sy + 0.205, sy + 0.232], color=C["blue"], lw=1.3, zorder=6)
ax.add_patch(Circle((sx + 0.073, sy + 0.116), 0.011, facecolor=C["blue"], edgecolor="white", lw=0.4, zorder=7))
ax.add_patch(Circle((sx + 0.202, sy + 0.232), 0.014, facecolor=C["orange"], edgecolor="white", lw=0.4, zorder=7))
ax.plot([sx + 0.073, sx + 0.202], [sy + 0.116, sy + 0.232], color=C["black"], lw=0.8, ls="--", zorder=6)
txt(sx + 0.140, sy + 0.180, r"$d_t$", size=7)
txt(sx + 0.137, sy + 0.054, "ZPD-regulated pursuit task", size=7.0, color=C["gray"])
box(sx + 0.265, sy + 0.215, 0.100, 0.060, "Obs.\n$O_t$", fc=C["blue_l"], ec=C["blue"], size=7.0)
box(sx + 0.265, sy + 0.125, 0.100, 0.060, "Reward\n$r_t$", fc=C["purple_l"], ec=C["purple"], size=7.0)
box(sx + 0.265, sy + 0.045, 0.100, 0.050, "ZPD band", fc="#FAFAFA", ec="#B7BDC5", size=6.8)

# b) Policy learning.
px, py = rx, ty
box(px + 0.050, py + 0.087, 0.150, 0.198, fc=C["blue_l"], ec=C["blue"], lw=1.0)
txt(px + 0.125, py + 0.260, "adaptive\nrobot policy", size=7.6, weight="bold", color=C["blue"])
# network glyph
for xoff, ys in [
    (0.085, [0.125, 0.165, 0.205]),
    (0.125, [0.115, 0.155, 0.195, 0.235]),
    (0.165, [0.145, 0.205]),
]:
    for yy in ys:
        ax.add_patch(Circle((px + xoff, py + yy), 0.0085, facecolor="white", edgecolor=C["blue"], lw=0.75, zorder=7))
for y1 in [0.125, 0.165, 0.205]:
    for y2 in [0.115, 0.155, 0.195, 0.235]:
        ax.add_line(Line2D([px + 0.094, px + 0.116], [py + y1, py + y2], color="#B5CBE2", lw=0.35, zorder=6))
for y1 in [0.115, 0.155, 0.195, 0.235]:
    for y2 in [0.145, 0.205]:
        ax.add_line(Line2D([px + 0.134, px + 0.156], [py + y1, py + y2], color="#B5CBE2", lw=0.35, zorder=6))
box(px + 0.245, py + 0.235, 0.110, 0.055, "virtual\npatients", fc=C["orange_l"], ec=C["orange"], size=6.8)
box(px + 0.245, py + 0.158, 0.110, 0.055, "PFSP", fc=C["purple_l"], ec=C["purple"], size=7.0)
box(px + 0.245, py + 0.080, 0.110, 0.055, "PPO\nupdate", fc=C["green_l"], ec=C["green"], size=6.8)
arrow(px + 0.245, py + 0.262, px + 0.200, py + 0.232, color=C["orange"], lw=1.0, ms=8)
arrow(px + 0.245, py + 0.185, px + 0.200, py + 0.185, color=C["purple"], lw=1.0, ms=8)
arrow(px + 0.245, py + 0.107, px + 0.200, py + 0.135, color=C["green"], lw=1.0, ms=8)

# c) Vision.
vx, vy = lx, by
rect(vx + 0.045, vy + 0.095, 0.170, 0.170, fc="#F8F8F8", ec="#BBC4CE", lw=0.8)
rect(vx + 0.072, vy + 0.125, 0.036, 0.028, fc="none", ec=C["blue"], lw=0.9)
rect(vx + 0.130, vy + 0.180, 0.040, 0.030, fc="none", ec=C["orange"], lw=0.9)
ax.add_patch(Circle((vx + 0.090, vy + 0.140), 0.010, facecolor=C["blue"], edgecolor="white", lw=0.4, zorder=7))
ax.add_patch(Circle((vx + 0.150, vy + 0.195), 0.012, facecolor=C["orange"], edgecolor="white", lw=0.4, zorder=7))
ax.add_patch(Circle((vx + 0.125, vy + 0.232), 0.010, facecolor=C["green"], edgecolor="white", lw=0.4, zorder=7))
txt(vx + 0.130, vy + 0.064, "camera frame", size=6.8, color=C["gray"])
box(vx + 0.250, vy + 0.205, 0.105, 0.055, "vision\nmodels", fc=C["green_l"], ec=C["green"], size=6.8)
box(vx + 0.250, vy + 0.115, 0.105, 0.055, "state\nestimator", fc=C["blue_l"], ec=C["blue"], size=6.8)
arrow(vx + 0.215, vy + 0.214, vx + 0.250, vy + 0.232, color=C["green"], lw=1.0, ms=8)
arrow(vx + 0.303, vy + 0.205, vx + 0.303, vy + 0.170, color=C["blue"], lw=1.0, ms=8)
txt(vx + 0.303, vy + 0.078, "position + velocity\nwith gap filling", size=6.3, color=C["gray"])

# d) Real platform.
rx0, ry0 = rx, by
rect(rx0 + 0.042, ry0 + 0.090, 0.205, 0.185, fc=C["photo"], ec="#AEB7C2", lw=1.0, ls=(0, (4, 3)))
rect(rx0 + 0.060, ry0 + 0.112, 0.170, 0.140, fc="white", ec="#D3D8DF", lw=0.8)
ax.add_line(Line2D([rx0 + 0.082, rx0 + 0.210], [ry0 + 0.180, ry0 + 0.205], color="#C6CDD5", lw=4.0, solid_capstyle="round", zorder=6))
ax.add_patch(Circle((rx0 + 0.140, ry0 + 0.194), 0.016, facecolor="#D9DEE5", edgecolor="#B8C0CA", lw=0.7, zorder=7))
txt(rx0 + 0.145, ry0 + 0.178, "photo slot", size=7.5, weight="bold", color="#7A8088")
txt(rx0 + 0.145, ry0 + 0.061, "insert annotated setup photo", size=6.7, color=C["gray"])
box(rx0 + 0.285, ry0 + 0.205, 0.087, 0.052, "UR10\ncontrol", fc=C["green_l"], ec=C["green"], size=6.5)
box(rx0 + 0.285, ry0 + 0.118, 0.087, 0.052, "magnetic\nactuation", fc=C["orange_l"], ec=C["orange"], size=6.5)
arrow(rx0 + 0.329, ry0 + 0.205, rx0 + 0.329, ry0 + 0.170, color=C["green"], lw=1.0, ms=8)
arrow(rx0 + 0.285, ry0 + 0.144, rx0 + 0.247, ry0 + 0.160, color=C["orange"], lw=1.0, ms=8)

# Inter-panel connectors: straight, aligned, with labels in the gutters.
# Top horizontal exchange.
arrow(lx + pw, ty + 0.248, rx, ty + 0.248, color=C["blue"], lw=1.25, ms=9)
label_box(0.500, ty + 0.270, r"$O_t, r_t$", C["blue"])
arrow(rx, ty + 0.174, lx + pw, ty + 0.174, color=C["orange"], lw=1.25, ms=9)
label_box(0.500, ty + 0.151, r"$a_t$", C["orange"])
# Right vertical command.
arrow(rx + pw / 2, ty, rx + pw / 2, by + ph, color=C["green"], lw=1.25, ms=9)
label_box(rx + pw / 2 + 0.042, 0.500, "command", C["green"])
# Bottom exchange.
arrow(rx, by + 0.245, lx + pw, by + 0.245, color=C["green"], lw=1.25, ms=9)
label_box(0.500, by + 0.268, "image feed", C["green"])
arrow(lx + pw, by + 0.142, rx, by + 0.142, color=C["blue"], lw=1.25, ms=9)
label_box(0.500, by + 0.119, r"real $O_t$", C["blue"])
# Left vertical task match.
arrow(lx + pw / 2, by + ph, lx + pw / 2, ty, color=C["purple"], lw=1.10, ms=8)
label_box(lx + pw / 2 - 0.050, 0.500, "same ZPD task", C["purple"])

# Legend outside panels.
ax.add_patch(Circle((0.055, 0.960), 0.006, facecolor=C["blue"], edgecolor="none"))
txt(0.070, 0.960, "robot / microrobot", size=6.4, color=C["gray"], ha="left")
ax.add_patch(Circle((0.185, 0.960), 0.006, facecolor=C["orange"], edgecolor="none"))
txt(0.200, 0.960, "patient hand", size=6.4, color=C["gray"], ha="left")

for path in [
    OUT_DIR / "fig_overview_system_framework_orthogonal_v3.png",
    DRAFT_DIR / "fig_overview_system_framework_orthogonal_v3_draft.png",
]:
    fig.savefig(path, dpi=450, bbox_inches="tight", pad_inches=0.035, facecolor="white")
fig.savefig(OUT_DIR / "fig_overview_system_framework_orthogonal_v3.pdf", bbox_inches="tight", pad_inches=0.035, facecolor="white")
fig.savefig(OUT_DIR / "fig_overview_system_framework_orthogonal_v3.svg", bbox_inches="tight", pad_inches=0.035, facecolor="white")
plt.close(fig)

print((OUT_DIR / "fig_overview_system_framework_orthogonal_v3.png").as_posix())
print((OUT_DIR / "fig_overview_system_framework_orthogonal_v3.pdf").as_posix())
print((OUT_DIR / "fig_overview_system_framework_orthogonal_v3.svg").as_posix())
print((DRAFT_DIR / "fig_overview_system_framework_orthogonal_v3_draft.png").as_posix())
