from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

OUT = Path("manuscripts/current_league_pfsp_window_7iter")


def add_image(ax, path, label=None, title=None):
    img = mpimg.imread(path)
    ax.imshow(img)
    ax.axis("off")
    if label:
        ax.text(0.0, 1.02, label, transform=ax.transAxes, fontsize=13, fontweight="bold", va="bottom")
    if title:
        ax.set_title(title, fontsize=10, pad=6)


# Composite 1: Cross-iteration validation.
fig = plt.figure(figsize=(11.0, 5.2), dpi=300)
gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.05)
ax = fig.add_subplot(gs[0, 0])
add_image(ax, OUT / "fig_cross_iter_validation_tis_heatmap_7iter.png", "(a)")
ax = fig.add_subplot(gs[0, 1])
add_image(ax, OUT / "fig_cross_iter_validation_summary_7iter.png", "(b)")
fig.suptitle("Cross-iteration validation of league-trained policies", fontsize=13, y=0.99)
fig.savefig(OUT / "paper_fig_cross_iter_validation_composite_7iter.png", bbox_inches="tight")
plt.close(fig)

# Composite 2: PFSP dynamics.
fig = plt.figure(figsize=(12.0, 8.2), dpi=300)
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.78], width_ratios=[1.12, 0.88], hspace=0.18, wspace=0.08)
ax = fig.add_subplot(gs[0, 0])
add_image(ax, OUT / "fig_pfsp_normalized_preference_heatmap_7iter.png", "(a)")
ax = fig.add_subplot(gs[0, 1])
add_image(ax, OUT / "fig_pfsp_opponent_age_preference_7iter.png", "(b)")
ax = fig.add_subplot(gs[1, :])
add_image(ax, OUT / "fig_pfsp_within_iteration_curves_7iter.png", "(c)")
fig.suptitle("PFSP sampling dynamics during league training", fontsize=13, y=0.99)
fig.savefig(OUT / "paper_fig_pfsp_dynamics_composite_7iter.png", bbox_inches="tight")
plt.close(fig)

# Composite 3: Auxiliary prediction visualization.
fig = plt.figure(figsize=(11.0, 6.0), dpi=300)
gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 0.9], wspace=0.04)
ax = fig.add_subplot(gs[0, 0])
add_image(ax, OUT / "fig_aux_future_prediction_trajectory_grid_7iter.png", "(a)")
ax = fig.add_subplot(gs[0, 1])
add_image(ax, OUT / "fig_aux_prediction_error_by_horizon_7iter.png", "(b)")
fig.suptitle("Auxiliary future-motion prediction visualization", fontsize=13, y=0.99)
fig.savefig(OUT / "paper_fig_aux_prediction_composite_7iter.png", bbox_inches="tight")
plt.close(fig)

# Contact sheet: all proposed paper composites.
fig = plt.figure(figsize=(9.0, 12.5), dpi=220)
gs = fig.add_gridspec(3, 1, hspace=0.08)
for i, (path, label) in enumerate([
    (OUT / "paper_fig_cross_iter_validation_composite_7iter.png", "Figure A. Cross-iteration validation"),
    (OUT / "paper_fig_pfsp_dynamics_composite_7iter.png", "Figure B. PFSP dynamics"),
    (OUT / "paper_fig_aux_prediction_composite_7iter.png", "Figure C. Auxiliary prediction"),
]):
    ax = fig.add_subplot(gs[i, 0])
    add_image(ax, path)
    ax.text(0.0, 1.01, label, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom")
fig.savefig(OUT / "paper_figures_contact_sheet_7iter.png", bbox_inches="tight")
plt.close(fig)

for path in [
    OUT / "paper_fig_cross_iter_validation_composite_7iter.png",
    OUT / "paper_fig_pfsp_dynamics_composite_7iter.png",
    OUT / "paper_fig_aux_prediction_composite_7iter.png",
    OUT / "paper_figures_contact_sheet_7iter.png",
]:
    print(path.as_posix())
