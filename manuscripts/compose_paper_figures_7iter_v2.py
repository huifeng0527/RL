from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

OUT = Path("manuscripts/current_league_pfsp_window_7iter")


def add_image(ax, path, label=None):
    ax.imshow(mpimg.imread(path))
    ax.axis("off")
    if label:
        ax.text(0.0, 1.02, label, transform=ax.transAxes, fontsize=14, fontweight="bold", va="bottom")


# Main PFSP figure: keep only the two readable mechanism panels.
fig = plt.figure(figsize=(11.0, 4.8), dpi=300)
gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1.0], wspace=0.03)
ax = fig.add_subplot(gs[0, 0])
add_image(ax, OUT / "fig_pfsp_normalized_preference_heatmap_7iter.png", "(a)")
ax = fig.add_subplot(gs[0, 1])
add_image(ax, OUT / "fig_pfsp_opponent_age_preference_7iter.png", "(b)")
fig.suptitle("PFSP sampling preferences during league training", fontsize=13, y=0.99)
fig.savefig(OUT / "paper_fig_pfsp_main_composite_7iter.png", bbox_inches="tight")
plt.close(fig)

# Supplementary PFSP figure: within-iteration curves alone.
fig = plt.figure(figsize=(10.5, 3.8), dpi=300)
ax = fig.add_subplot(111)
add_image(ax, OUT / "fig_pfsp_within_iteration_curves_7iter.png", "")
fig.savefig(OUT / "paper_fig_pfsp_within_iteration_supplement_7iter.png", bbox_inches="tight")
plt.close(fig)

# A tighter contact sheet with the recommended main-paper figures.
fig = plt.figure(figsize=(9.2, 10.8), dpi=220)
gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 0.82, 1.0], hspace=0.08)
for i, (path, label) in enumerate([
    (OUT / "paper_fig_cross_iter_validation_composite_7iter.png", "Figure 2. Cross-iteration validation"),
    (OUT / "paper_fig_pfsp_main_composite_7iter.png", "Figure 3. PFSP sampling preferences"),
    (OUT / "paper_fig_aux_prediction_composite_7iter.png", "Figure 4. Auxiliary prediction visualization"),
]):
    ax = fig.add_subplot(gs[i, 0])
    add_image(ax, path)
    ax.text(0.0, 1.01, label, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom")
fig.savefig(OUT / "paper_figures_recommended_contact_sheet_7iter.png", bbox_inches="tight")
plt.close(fig)

for path in [
    OUT / "paper_fig_pfsp_main_composite_7iter.png",
    OUT / "paper_fig_pfsp_within_iteration_supplement_7iter.png",
    OUT / "paper_figures_recommended_contact_sheet_7iter.png",
]:
    print(path.as_posix())
