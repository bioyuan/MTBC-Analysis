"""Recreate Figures 1-4 from the public aggregate study tables."""

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator, PercentFormatter
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "01_data"
OUT = ROOT / "04_generated_figures"

GROUPS = {
    "Fungi": ("PJ", "CRY", "CAN", "OF", "ASP", "TM"),
    "Viruses": ("B19", "HCoVs", "RHV", "Bocav", "HAdV", "IFV", "RSV", "HPIVs", "HMPV"),
    "Bacteria": ("SPY", "SA", "HIN", "KPN", "MCAT", "SPN", "BP", "PA", "ABA", "LEP"),
    "Atypical pathogens": ("MP", "CPN", "CPS"),
}
STRATIFIED_ORDER = (
    "CRY", "PJ", "CAN", "ASP", "OF", "TM", "B19", "HCoVs", "RHV", "IFV", "HPIVs",
    "HAdV", "HMPV", "RSV", "Bocav", "KPN", "HIN", "SA", "SPY", "SPN", "MCAT",
    "PA", "ABA", "LEP", "BP", "CPS", "CPN", "MP",
)
CATEGORY_COLOR = {
    "Bacteria": "#387CB1",
    "Viruses": "#965CA4",
    "Fungi": "#BE4D60",
    "Atypical pathogens": "#438C61",
}
PATHOGEN_COLOR = {
    pathogen: CATEGORY_COLOR[group]
    for group, pathogens in GROUPS.items()
    for pathogen in pathogens
}
POSITIVE = "#BE4650"
NEGATIVE = "#818181"
NONSIGNIFICANT = "#151515"
PR_HIGH = "#B7433C"
PR_LOW = "#2E689E"

AGE = {
    "0-14": "Children (0-14 years)",
    "15-44": "Young adults (15-44 years)",
    "45-64": "Middle-aged adults (45-64 years)",
    "65+": "Older adults (65+ years)",
}
REGION = {
    "North": "North China",
    "Central-Southern": "Central-South China",
    "Southwest": "Southwest China",
    "East": "East China",
}


def configure_style():
    # Liberation Serif is metrically compatible on systems without Times New Roman.
    installed = set(font_manager.get_font_names())
    family = next(
        font for font in ("Times New Roman", "Liberation Serif", "DejaVu Serif")
        if font in installed
    )
    mpl.rcParams.update({
        "font.family": family,
        "font.size": 8,
        "axes.linewidth": 0.8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    })


def read_table(name, keys):
    return pd.read_csv(DATA / name).set_index(keys, verify_integrity=True)


def save_figure(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=600)
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)


def significance_stars(pvalue):
    for threshold, stars in ((0.0001, "****"), (0.001, "***"), (0.01, "**"), (0.05, "*")):
        if pvalue < threshold:
            return stars
    return ""


def effect_color(odds_ratio, pvalue):
    if pvalue >= 0.05:
        return NONSIGNIFICANT
    return POSITIVE if odds_ratio > 1 else NEGATIVE


def effect_legend(fig, y=0.975):
    handles = [
        Line2D([], [], color=color, marker="o", markersize=4, label=label)
        for color, label in (
            (POSITIVE, "OR > 1 (p < 0.05)"),
            (NEGATIVE, "OR < 1 (p < 0.05)"),
            (NONSIGNIFICANT, "Non-significant (p >= 0.05)"),
        )
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, y), ncol=3)


def prevalence_pvalue(row):
    a = row.mtbc_positive_positive
    b = row.mtbc_negative_positive
    return chi2_contingency([
        [a, row.mtbc_positive_total - a],
        [b, row.mtbc_negative_total - b],
    ]).pvalue


def plot_figure1(counts):
    rates = counts.assign(
        negative_rate=lambda d: d.mtbc_negative_positive / d.mtbc_negative_total,
        positive_rate=lambda d: d.mtbc_positive_positive / d.mtbc_positive_total,
    )
    left = rates.sort_values("negative_rate", ascending=False, kind="stable")
    right = rates.sort_values("positive_rate", ascending=False, kind="stable")
    ranks = np.arange(len(rates))

    fig = plt.figure(figsize=(8.7, 8.4))
    grid = fig.add_gridspec(
        1, 5, left=0.076, right=0.975, top=0.91, bottom=0.083,
        width_ratios=(2.65, 1.55, 2.65, 0.34, 2.4), wspace=0,
    )
    left_ax, rank_ax, right_ax, bridge_ax, pr_ax = [fig.add_subplot(grid[0, i]) for i in range(5)]
    for ax in fig.axes:
        ax.set_ylim(len(ranks) - 0.35, -0.65)
        ax.tick_params(axis="y", length=0, pad=4)

    def draw_bars(ax, ordered, column, invert):
        vals = ordered[column]
        limit = 0.36
        ax.barh(ranks, [limit] * len(ranks), height=0.94, color="#e4e4e4")
        ax.barh(ranks, vals, height=0.91, color=[PATHOGEN_COLOR[p] for p in ordered.index])
        ax.set_xlim((limit, 0) if invert else (0, limit))
        ax.set_xticks(np.arange(0, 0.31, 0.1))
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        for i, val in enumerate(vals):
            inside = val > 0.25
            x = val - 0.005 if inside else val + 0.005
            ha = ("left" if invert else "right") if inside else ("right" if invert else "left")
            ax.text(x, i, f"{val:.2%}", ha=ha, va="center",
                    color="white" if inside else "#111111", fontweight="bold" if inside else "normal")
        ax.set_yticks(ranks, ordered.index, fontweight="bold", fontsize=7)

    draw_bars(left_ax, left, "negative_rate", True)
    draw_bars(right_ax, right, "positive_rate", False)
    right_ax.yaxis.tick_right()
    right_ax.tick_params(axis="y", labelleft=False, labelright=False)
    left_ax.set_xlabel("MTBC-negative population", fontweight="bold")
    right_ax.set_xlabel("MTBC-positive population", fontweight="bold")
    right_rank = {p: i for i, p in enumerate(right.index)}
    for i, pathogen in enumerate(left.index):
        j = right_rank[pathogen]
        rank_ax.plot([0, 1], [i, j], color=PATHOGEN_COLOR[pathogen],
                     ls="--" if j > i else "-", lw=1.45)
        bridge_ax.plot([0, 1], [j, j], color=PATHOGEN_COLOR[pathogen], lw=0.8)
    for ax in (rank_ax, bridge_ax):
        ax.set_xlim(0, 1)
        ax.set_axis_off()

    ratios = right.positive_rate / right.negative_rate
    pr_colors = [PR_HIGH if val > 1 else PR_LOW for val in ratios]
    pr_ax.barh(ranks, ratios, height=0.70, color=pr_colors, edgecolor="#333333", linewidth=0.4)
    pr_ax.axvline(1, color="#222222", lw=0.8, ls="--")
    pr_ax.set_xlim(0, max(2.0, ratios.max() + 0.28))
    pr_ax.xaxis.set_major_locator(MultipleLocator(0.5))
    pr_ax.set_yticks([])
    pr_ax.set_xlabel("Prevalence ratio (PR)", fontweight="bold")
    for i, (value, color, pvalue) in enumerate(zip(ratios, pr_colors, right.apply(prevalence_pvalue, axis=1))):
        pr_ax.text(value - 0.025, i, f"{value:.2f}", ha="right", va="center",
                   fontweight="bold", color="white", fontsize=7)
        pr_ax.text(value + 0.035, i, significance_stars(pvalue), ha="left", va="center",
                   color=color, fontweight="bold", fontsize=7)
    fig.text(0.04, 0.974, "A  Pathogen detection and rank", fontweight="bold", va="top")
    fig.text(0.748, 0.974, "B  Prevalence ratios", fontweight="bold", va="top")
    fig.legend(
        handles=[Patch(color=color, label=group) for group, color in CATEGORY_COLOR.items()],
        loc="upper left", bbox_to_anchor=(0.066, 0.954), ncol=4, columnspacing=0.85, handletextpad=0.35,
    )
    fig.legend(handles=[Patch(color=PR_LOW, label="PR <= 1"), Patch(color=PR_HIGH, label="PR > 1")],
               loc="upper left", bbox_to_anchor=(0.752, 0.954), ncol=2, columnspacing=0.8)
    save_figure(fig, "01_figure1")


def format_count(row, group):
    n = row[f"mtbc_{group}_positive"]
    total = row[f"mtbc_{group}_total"]
    return f"{n:,} ({100 * n / total:.2f})"


def format_or(row):
    return f"{row['or']:.2f} ({row.ci_lower:.2f}-{row.ci_upper:.2f})"


def plot_figure2(counts):
    crude = read_table("02_crude_or.csv", "pathogen")
    adjusted = read_table("03_adjusted_or.csv", "pathogen")
    fig = plt.figure(figsize=(8.2, 8.9))
    grid = fig.add_gridspec(1, 3, left=0.03, right=0.985, top=0.876, bottom=0.066,
                           width_ratios=(4.9, 2.42, 1.54), wspace=0.02)
    table, forest, estimate = [fig.add_subplot(grid[0, i]) for i in range(3)]
    rows = [(group, None) for group in GROUPS]
    rows = [item for group, _ in rows for item in [(group, None), *((group, p) for p in GROUPS[group])]]
    for ax in (table, forest, estimate):
        ax.set_ylim(len(rows) - 0.5, -0.5)
        ax.set_yticks([])
    for ax in (table, estimate):
        ax.set_xlim(0, 1)
        ax.set_axis_off()

    table_columns = (0.04, 0.31, 0.55, 0.83)
    for x, label in zip(table_columns, ("Pathogen", "MTBC-negative\ncases (%)",
                                       "MTBC-positive\ncases (%)", "Crude OR\n(95% CI)")):
        table.text(x, 1.018, label, transform=table.transAxes, va="bottom",
                   ha="left" if x == table_columns[0] else "center", fontweight="bold")
    forest.set_title("Adjusted OR\n(95% CI)", fontweight="bold", pad=10)
    estimate.text(0.5, 1.018, "Adjusted OR\n(95% CI)", ha="center", va="bottom",
                  transform=estimate.transAxes, fontweight="bold")
    for i, (group, pathogen) in enumerate(rows):
        if pathogen is None:
            table.text(table_columns[0], i, f"{group} ({len(GROUPS[group])})",
                       ha="left", va="center", fontweight="bold", fontstyle="italic")
            continue
        count = counts.loc[pathogen]
        for x, value in zip(table_columns, (
            pathogen, format_count(count, "negative"), format_count(count, "positive"),
            format_or(crude.loc[pathogen]),
        )):
            table.text(x, i, value, va="center", ha="left" if x == table_columns[0] else "center")
        model = adjusted.loc[pathogen]
        color = effect_color(model["or"], model.pvalue)
        forest.plot([model.ci_lower, model.ci_upper], [i, i], color=color, lw=1.25)
        forest.scatter(model["or"], i, s=15, color=color, zorder=3)
        estimate.text(0.5, i, format_or(model), ha="center", va="center")
    forest.axvline(1, ls="--", lw=0.7, color="#6f6f6f")
    forest.set_xlim(0, 1.60)
    forest.xaxis.set_major_locator(MultipleLocator(0.5))
    forest.set_xlabel("Adjusted odds ratio")
    forest.spines[["right", "left", "top"]].set_visible(False)
    for y in (0.876, 0.066):
        fig.add_artist(Line2D([0.03, 0.985], [y, y], transform=fig.transFigure, color="#303030", lw=0.75))
    effect_legend(fig)
    save_figure(fig, "02_figure2")


def harmonize_age(value):
    return value[1:] if value[0].isalpha() else value


def stratum_estimates(estimate_file, count_file, group_key, age=False):
    estimates = pd.read_csv(DATA / estimate_file).rename(columns={group_key: "stratum"})
    counts = pd.read_csv(DATA / count_file)
    if age:
        estimates["stratum"] = estimates.stratum.map(harmonize_age)
        counts["stratum"] = counts.stratum.map(harmonize_age)
    keys = ["stratum", "pathogen"]
    merged = estimates.merge(
        counts[keys + ["mtbc_positive_positive"]], on=keys, how="left", validate="one_to_one",
    )
    if merged.mtbc_positive_positive.isna().any():
        raise ValueError(f"Missing stratum counts for {estimate_file}")
    # The reference plots suppress estimates based on <=5 MTBC-positive detections.
    return merged.loc[merged.mtbc_positive_positive > 5].set_index(keys, verify_integrity=True)


def draw_stratum(ax, estimates, stratum, label, ymax):
    d = estimates.loc[stratum].reindex(STRATIFIED_ORDER)
    ax.axhline(1, color="#767676", ls="--", lw=0.8)
    for i, (_, row) in enumerate(d.iterrows()):
        if pd.isna(row["or"]):
            continue
        color = effect_color(row["or"], row.pvalue)
        ax.plot([i, i], [row.ci_lower, row.ci_upper], color=color, lw=1.0)
        ax.scatter(i, row["or"], s=15, color=color, zorder=3)
    ax.set_xlim(-0.65, len(d) - 0.35)
    ax.set_ylim(0, ymax)
    ax.yaxis.set_major_locator(MultipleLocator(0.5 if ymax <= 2.6 else 1))
    ax.grid(axis="y", alpha=0.20, lw=0.55)
    ax.set_axisbelow(True)
    ax.text(0.012, 0.048, label, transform=ax.transAxes, ha="left", va="bottom", fontweight="bold")


def stratum_labels(ax):
    ax.set_xticks(np.arange(len(STRATIFIED_ORDER)), STRATIFIED_ORDER,
                  rotation=90, fontsize=7, fontweight="bold")
    ax.tick_params(axis="x", pad=2)


def plot_figure3():
    sex = stratum_estimates("05_sex_or.csv", "09_sex_counts.csv", "sex_group")
    age = stratum_estimates("04_age_or.csv", "07_age_counts.csv", "group", age=True)
    fig = plt.figure(figsize=(8.4, 10.6))
    grid = fig.add_gridspec(7, 1, left=0.095, right=0.98, bottom=0.064, top=0.93,
                           height_ratios=[1, 1, 0.20, 1, 1, 1, 1], hspace=0.14)
    axes = [fig.add_subplot(grid[i]) for i in (0, 1, 3, 4, 5, 6)]
    for ax, group in zip(axes[:2], ("Male", "Female")):
        draw_stratum(ax, sex, group, group, 2.5)
    for ax, (group, label), ymax in zip(axes[2:], AGE.items(), (4, 2.5, 2.5, 2.5)):
        draw_stratum(ax, age, group, label, ymax)
    for ax in axes:
        stratum_labels(ax)
        ax.tick_params(axis="x", labelbottom=ax is axes[-1])
    fig.text(0.096, 0.944, "A  Sex strata", fontweight="bold")
    fig.text(0.096, axes[2].get_position().y1 + 0.009, "B  Age strata", fontweight="bold")
    fig.supylabel("Odds ratio (95% CI)", x=0.022, fontsize=9)
    effect_legend(fig, y=0.992)
    save_figure(fig, "03_figure3")


def plot_figure4():
    region = stratum_estimates("06_region_or.csv", "08_region_counts.csv", "group")
    fig, axes = plt.subplots(4, 1, figsize=(8.4, 8), sharex=True)
    fig.subplots_adjust(left=0.095, right=0.98, bottom=0.082, top=0.91, hspace=0.15)
    for ax, (group, label) in zip(axes, REGION.items()):
        draw_stratum(ax, region, group, label, 2.5)
    stratum_labels(axes[-1])
    fig.text(0.096, 0.928, "A  Region strata", fontweight="bold")
    fig.supylabel("Odds ratio (95% CI)", x=0.022, fontsize=9)
    effect_legend(fig, y=0.991)
    save_figure(fig, "04_figure4")


def main():
    configure_style()
    OUT.mkdir(exist_ok=True)
    counts = read_table("01_overall_counts.csv", "pathogen")
    plot_figure1(counts)
    plot_figure2(counts)
    plot_figure3()
    plot_figure4()


if __name__ == "__main__":
    main()
