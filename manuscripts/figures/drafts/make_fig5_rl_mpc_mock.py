from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = Path(__file__).resolve().parent
PNG_PATH = OUT_DIR / "fig5_rl_vs_mpc_mock_data.png"
PDF_PATH = OUT_DIR / "fig5_rl_vs_mpc_mock_data.pdf"

N_PAIRS = 20
ZPD_LOW = 3.5
ZPD_HIGH = 5.5
HORIZON_S = 60.0

COLOR_RL = "#2a78d6"
COLOR_MPC = "#eb6834"
COLOR_ZPD = "#1baf7a"
COLOR_INK = "#171717"
COLOR_MUTED = "#74716b"
COLOR_PAIR = "#c7c4bd"
COLOR_AXIS = "#aaa79f"
COLOR_SURFACE = "#fcfcfb"


def bootstrap_mean_ci(values, rng, samples=10000):
    values = np.asarray(values, dtype=float)
    draws = rng.choice(values, size=(samples, values.size), replace=True).mean(axis=1)
    return float(values.mean()), np.percentile(draws, [2.5, 97.5])


def smooth_curve(values, window=5):
    kernel = np.array([1, 2, 3, 2, 1], dtype=float)
    kernel /= kernel.sum()
    return np.convolve(values, kernel, mode="same")


def bootstrap_curve(curves, rng, samples=4000):
    curves = np.asarray(curves, dtype=float)
    indices = rng.integers(0, curves.shape[0], size=(samples, curves.shape[0]))
    means = curves[indices].mean(axis=1)
    return curves.mean(axis=0), np.percentile(means, [2.5, 97.5], axis=0)


def style_axis(ax):
    ax.set_facecolor(COLOR_SURFACE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLOR_AXIS)
    ax.spines["bottom"].set_color(COLOR_AXIS)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.tick_params(colors=COLOR_MUTED, labelsize=9, width=0.8, length=4)
    ax.grid(False)


def paired_panel(ax, left, right, ylabel, ylim, delta_label, rng, cap=None):
    jitter = rng.uniform(-0.055, 0.055, size=N_PAIRS)
    for i in range(N_PAIRS):
        ax.plot(
            [jitter[i], 1 + jitter[i]],
            [left[i], right[i]],
            color=COLOR_PAIR,
            linewidth=0.9,
            alpha=0.75,
            zorder=1,
        )

    ax.scatter(
        jitter,
        left,
        s=43,
        color=COLOR_MPC,
        edgecolor=COLOR_SURFACE,
        linewidth=1.1,
        zorder=3,
        label="CV-MPC",
    )
    ax.scatter(
        1 + jitter,
        right,
        s=43,
        color=COLOR_RL,
        edgecolor=COLOR_SURFACE,
        linewidth=1.1,
        zorder=3,
        label="League RL",
    )

    for x, values, color in [(0, left, COLOR_MPC), (1, right, COLOR_RL)]:
        mean, ci = bootstrap_mean_ci(values, rng)
        ax.errorbar(
            x,
            mean,
            yerr=[[mean - ci[0]], [ci[1] - mean]],
            fmt="D",
            markersize=7.5,
            markerfacecolor=color,
            markeredgecolor=COLOR_INK,
            markeredgewidth=0.8,
            ecolor=COLOR_INK,
            elinewidth=1.5,
            capsize=4,
            capthick=1.4,
            zorder=5,
        )

    delta = np.asarray(right) - np.asarray(left)
    delta_mean, delta_ci = bootstrap_mean_ci(delta, rng)
    annotation = (
        f"{delta_label} = {delta_mean:+.2f}\n"
        f"95% CI [{delta_ci[0]:+.2f}, {delta_ci[1]:+.2f}]"
    )
    ax.text(
        0.04,
        0.96,
        annotation,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9.2,
        color=COLOR_INK,
        bbox=dict(boxstyle="round,pad=0.34", facecolor="#f1f0ec", edgecolor="none"),
    )

    if cap is not None:
        ax.axhline(cap, color=COLOR_MUTED, linewidth=1.0, linestyle=(0, (4, 3)), zorder=0)
        ax.text(
            1.16,
            cap - 0.8,
            f"{cap:.0f}-s limit",
            color=COLOR_MUTED,
            fontsize=8.3,
            ha="right",
            va="top",
        )

    ax.set_xlim(-0.28, 1.28)
    ax.set_ylim(*ylim)
    ax.set_xticks([0, 1], ["CV-MPC", "League RL"])
    ax.set_ylabel(ylabel, fontsize=10.5, color=COLOR_INK)
    style_axis(ax)
    return delta


def main():
    rng = np.random.default_rng(20260819)

    hand_stride = rng.uniform(0.30, 0.60, N_PAIRS)
    challenge = (hand_stride - 0.30) / 0.30

    mpc_tiz = np.clip(
        0.49 - 0.19 * challenge + rng.normal(0.0, 0.070, N_PAIRS),
        0.10,
        0.66,
    )
    tiz_gain = 0.15 - 0.025 * challenge + rng.normal(0.0, 0.055, N_PAIRS)
    tiz_gain[[3, 14]] -= np.array([0.18, 0.13])
    rl_tiz = np.clip(mpc_tiz + tiz_gain, 0.12, 0.82)

    mpc_duration = np.clip(
        49.0 - 18.0 * challenge + rng.normal(0.0, 6.0, N_PAIRS),
        16.0,
        HORIZON_S,
    )
    duration_gain = 10.0 - 1.5 * challenge + rng.normal(0.0, 5.0, N_PAIRS)
    duration_gain[[3, 9, 14]] -= np.array([12.0, 8.0, 10.0])
    rl_duration = np.clip(mpc_duration + duration_gain, 16.0, HORIZON_S)

    distance_bins = np.linspace(0.0, 10.0, 61)
    bin_centers = 0.5 * (distance_bins[:-1] + distance_bins[1:])
    mpc_curves = []
    rl_curves = []
    for i in range(N_PAIRS):
        mpc_steps = max(80, int(mpc_duration[i] * 20))
        rl_steps = max(80, int(rl_duration[i] * 20))

        mpc_center = 4.75 + rng.normal(0.0, 0.30)
        mpc_spread = 1.25 + 0.45 * challenge[i]
        mpc_dist = rng.normal(mpc_center, mpc_spread, mpc_steps)
        mpc_dist += 0.45 * np.sin(np.linspace(0, 5 * np.pi, mpc_steps) + rng.uniform(0, 2 * np.pi))

        rl_center = 4.55 + rng.normal(0.0, 0.16)
        rl_spread = 0.72 + 0.24 * challenge[i]
        rl_dist = rng.normal(rl_center, rl_spread, rl_steps)
        rl_dist += 0.22 * np.sin(np.linspace(0, 5 * np.pi, rl_steps) + rng.uniform(0, 2 * np.pi))

        mpc_hist, _ = np.histogram(np.clip(mpc_dist, 0, 10), bins=distance_bins, density=True)
        rl_hist, _ = np.histogram(np.clip(rl_dist, 0, 10), bins=distance_bins, density=True)
        mpc_curves.append(smooth_curve(mpc_hist))
        rl_curves.append(smooth_curve(rl_hist))

    mpc_curve, mpc_curve_ci = bootstrap_curve(mpc_curves, rng)
    rl_curve, rl_curve_ci = bootstrap_curve(rl_curves, rng)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelcolor": COLOR_INK,
        "text.color": COLOR_INK,
        "figure.facecolor": COLOR_SURFACE,
        "savefig.facecolor": COLOR_SURFACE,
    })

    fig, axes = plt.subplots(2, 2, figsize=(13.6, 8.8), constrained_layout=False)
    fig.subplots_adjust(left=0.075, right=0.975, bottom=0.085, top=0.935, wspace=0.28, hspace=0.34)

    delta_tiz = paired_panel(
        axes[0, 0],
        mpc_tiz,
        rl_tiz,
        "Horizon-normalized TIZ",
        (0.0, 0.90),
        "Mean paired ΔTIZ",
        rng,
    )
    delta_duration = paired_panel(
        axes[0, 1],
        mpc_duration,
        rl_duration,
        "Episode duration (s)",
        (0.0, 63.5),
        "Mean paired Δduration",
        rng,
        cap=HORIZON_S,
    )

    ax = axes[1, 0]
    ax.axvspan(ZPD_LOW, ZPD_HIGH, color=COLOR_ZPD, alpha=0.12, zorder=0)
    ax.axvline(ZPD_LOW, color=COLOR_ZPD, linewidth=1.1, alpha=0.85)
    ax.axvline(ZPD_HIGH, color=COLOR_ZPD, linewidth=1.1, alpha=0.85)
    ax.fill_between(
        bin_centers,
        mpc_curve_ci[0],
        mpc_curve_ci[1],
        color=COLOR_MPC,
        alpha=0.12,
        linewidth=0,
    )
    ax.fill_between(
        bin_centers,
        rl_curve_ci[0],
        rl_curve_ci[1],
        color=COLOR_RL,
        alpha=0.12,
        linewidth=0,
    )
    ax.plot(bin_centers, mpc_curve, color=COLOR_MPC, linewidth=2.2, label="CV-MPC")
    ax.plot(bin_centers, rl_curve, color=COLOR_RL, linewidth=2.2, label="League RL")
    ax.text(
        0.45 * (ZPD_LOW + ZPD_HIGH),
        ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 0.5,
        "",
    )
    ax.text(
        0.5 * (ZPD_LOW + ZPD_HIGH),
        0.96,
        "target ZPD",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=8.8,
        color="#166a4d",
    )
    ax.set_xlim(0, 10)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("TCP–virtual-hand distance (cm)", fontsize=10.5)
    ax.set_ylabel("Mean within-rollout occupancy density", fontsize=10.5)
    ax.legend(frameon=False, loc="upper right", fontsize=9, handlelength=2.2)
    style_axis(ax)

    ax = axes[1, 1]
    ax.axvline(0, color=COLOR_AXIS, linewidth=1.0, zorder=0)
    ax.axhline(0, color=COLOR_AXIS, linewidth=1.0, zorder=0)
    ax.scatter(
        delta_tiz,
        delta_duration,
        s=62,
        color="#275d83",
        edgecolor=COLOR_SURFACE,
        linewidth=1.4,
        alpha=0.92,
        zorder=3,
    )
    both_better = int(np.sum((delta_tiz > 0) & (delta_duration > 0)))
    ax.text(
        0.97,
        0.96,
        f"RL better on both\n{both_better}/{N_PAIRS} paired trials",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9.2,
        color=COLOR_INK,
        bbox=dict(boxstyle="round,pad=0.34", facecolor="#f1f0ec", edgecolor="none"),
    )
    ax.text(
        0.98,
        0.04,
        "higher TIZ, shorter duration",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.0,
        color=COLOR_MUTED,
    )
    ax.text(
        0.02,
        0.96,
        "longer duration, lower TIZ",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.0,
        color=COLOR_MUTED,
    )
    x_pad = max(0.025, 0.12 * (delta_tiz.max() - delta_tiz.min()))
    y_pad = max(2.0, 0.12 * (delta_duration.max() - delta_duration.min()))
    ax.set_xlim(delta_tiz.min() - x_pad, delta_tiz.max() + x_pad)
    ax.set_ylim(delta_duration.min() - y_pad, delta_duration.max() + y_pad)
    ax.set_xlabel("Paired ΔTIZ (League RL − CV-MPC)", fontsize=10.5)
    ax.set_ylabel("Paired Δduration (s)", fontsize=10.5)
    style_axis(ax)

    for label, ax in zip(["(a)", "(b)", "(c)", "(d)"], axes.ravel()):
        ax.text(
            -0.13,
            1.06,
            label,
            transform=ax.transAxes,
            fontsize=13,
            fontweight="bold",
            ha="left",
            va="bottom",
            color=COLOR_INK,
        )

    fig.text(
        0.5,
        0.986,
        "SYNTHETIC MOCK DATA — LAYOUT PREVIEW ONLY",
        ha="center",
        va="top",
        fontsize=10,
        fontweight="bold",
        color="#8a4f3c",
    )

    fig.savefig(PNG_PATH, dpi=300, bbox_inches="tight")
    fig.savefig(PDF_PATH, bbox_inches="tight")
    print(PNG_PATH)
    print(PDF_PATH)


if __name__ == "__main__":
    main()
