from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

OUT = Path("manuscripts/current_league_pfsp_window_7iter")


def add_image(ax, path, label=None):
    ax.imshow(mpimg.imread(path))
    ax.axis("off")
    if label:
        ax.text(0.0, 1.02, label, transform=ax.transAxes, fontsize=14, fontweight="bold", va="bottom")


# PFSP figure with within-iteration curves included.
fig = plt.figure(figsize=(12.0, 8.4), dpi=300)
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.8], width_ratios=[1.15, 1.0], hspace=0.08, wspace=0.04)
ax = fig.add_subplot(gs[0, 0])
add_image(ax, OUT / "fig_pfsp_normalized_preference_heatmap_7iter.png", "(a)")
ax = fig.add_subplot(gs[0, 1])
add_image(ax, OUT / "fig_pfsp_opponent_age_preference_7iter.png", "(b)")
ax = fig.add_subplot(gs[1, :])
add_image(ax, OUT / "fig_pfsp_within_iteration_curves_7iter.png", "(c)")
fig.suptitle("PFSP sampling dynamics during league training", fontsize=13, y=0.99)
fig.savefig(OUT / "paper_fig_pfsp_full_composite_7iter.png", bbox_inches="tight")
plt.close(fig)

# Recommended contact sheet with PFSP full composite.
fig = plt.figure(figsize=(9.2, 12.0), dpi=220)
gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.05, 1.0], hspace=0.06)
for i, (path, label) in enumerate([
    (OUT / "paper_fig_cross_iter_validation_composite_7iter.png", "Figure 2. Cross-iteration validation"),
    (OUT / "paper_fig_pfsp_full_composite_7iter.png", "Figure 3. PFSP sampling dynamics"),
    (OUT / "paper_fig_aux_prediction_composite_7iter.png", "Figure 4. Auxiliary prediction visualization"),
]):
    ax = fig.add_subplot(gs[i, 0])
    add_image(ax, path)
    ax.text(0.0, 1.01, label, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom")
fig.savefig(OUT / "paper_figures_recommended_contact_sheet_with_pfsp_curves_7iter.png", bbox_inches="tight")
plt.close(fig)

for path in [
    OUT / "paper_fig_pfsp_full_composite_7iter.png",
    OUT / "paper_figures_recommended_contact_sheet_with_pfsp_curves_7iter.png",
]:
    print(path.as_posix())
