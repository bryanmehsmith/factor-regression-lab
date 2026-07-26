"""Matplotlib figures for the app and the CLI report."""

import numpy as np
import pandas as pd
import scipy.stats as stats
from matplotlib.figure import Figure
from statsmodels.graphics.tsaplots import plot_acf

from factor_lab.regression import FactorRegression

FIG_WIDTH = 10


def tstat_comparison(comparison: pd.DataFrame, threshold: float = 1.96) -> Figure:
    """Grouped bars of each term's t-statistic under every standard-error assumption.

    The centrepiece chart: the estimates behind these bars are identical, so any
    difference in height is purely the cost of a weaker assumption about the
    residuals. Terms whose bars straddle the threshold line are the ones whose
    significance depends on that assumption.
    """
    t_stats = comparison.xs("t_stat", axis=1, level="statistic")
    terms = t_stats.index.tolist()
    se_types = t_stats.columns.tolist()

    positions = np.arange(len(terms))
    bar_width = 0.8 / len(se_types)

    figure = Figure(figsize=(FIG_WIDTH, 4.5))
    axes = figure.add_subplot(111)

    for offset, se_type in enumerate(se_types):
        axes.bar(
            positions + offset * bar_width,
            t_stats[se_type].abs(),
            width=bar_width,
            label=se_type,
        )

    axes.axhline(
        threshold,
        color="crimson",
        linestyle="--",
        linewidth=1,
        label=f"|t| = {threshold} (5% significance)",
    )
    axes.set_xticks(positions + bar_width * (len(se_types) - 1) / 2)
    axes.set_xticklabels(terms, rotation=0)
    axes.set_ylabel("|t statistic|")
    axes.set_title("Same estimates, three assumptions about the residuals")
    axes.legend(fontsize=8)
    figure.tight_layout()
    return figure


def diagnostics_grid(regression: FactorRegression) -> Figure:
    """Residuals versus fitted, Q-Q, residual ACF, and residual histogram."""
    residuals = regression.residuals
    fitted = regression.fitted

    figure = Figure(figsize=(FIG_WIDTH, 7))
    top_left, top_right, bottom_left, bottom_right = (
        figure.add_subplot(221),
        figure.add_subplot(222),
        figure.add_subplot(223),
        figure.add_subplot(224),
    )

    top_left.scatter(fitted, residuals, s=8, alpha=0.5)
    top_left.axhline(0, color="black", linewidth=0.8)
    top_left.set_xlabel("Fitted value")
    top_left.set_ylabel("Residual")
    top_left.set_title("Residuals vs fitted (look for a funnel)")

    stats.probplot(residuals, dist="norm", plot=top_right)
    top_right.set_title("Normal Q-Q (look for tails off the line)")
    top_right.get_lines()[0].set_markersize(3)
    top_right.get_lines()[0].set_alpha(0.5)

    max_lags = int(min(24, max(1, len(residuals) // 4)))
    plot_acf(residuals, lags=max_lags, ax=bottom_left, zero=False)
    bottom_left.set_title("Residual ACF (bars outside the band are autocorrelation)")
    bottom_left.set_xlabel("Lag (months)")

    bottom_right.hist(residuals, bins=40, density=True, alpha=0.7)
    grid = np.linspace(residuals.min(), residuals.max(), 200)
    bottom_right.plot(
        grid,
        stats.norm.pdf(grid, residuals.mean(), residuals.std()),
        color="crimson",
        linewidth=1,
        label="Normal fit",
    )
    bottom_right.set_title("Residual distribution")
    bottom_right.set_xlabel("Residual")
    bottom_right.legend(fontsize=8)

    figure.tight_layout()
    return figure


def rolling_term(term: str, estimates: pd.DataFrame, full_sample: float | None = None) -> Figure:
    """One rolling coefficient with its 95% band, against the full-sample value."""
    figure = Figure(figsize=(FIG_WIDTH, 4))
    axes = figure.add_subplot(111)

    axes.plot(estimates.index, estimates["estimate"], linewidth=1.2, label=f"Rolling {term}")
    axes.fill_between(
        estimates.index,
        estimates["lower"],
        estimates["upper"],
        alpha=0.2,
        label="95% band",
    )
    if full_sample is not None:
        axes.axhline(
            full_sample,
            color="crimson",
            linestyle="--",
            linewidth=1,
            label="Full-sample estimate",
        )
    axes.axhline(0, color="black", linewidth=0.8)
    axes.set_ylabel("Annualized alpha" if term == "alpha" else "Beta")
    axes.set_title(f"Rolling {term}")
    axes.legend(fontsize=8)
    figure.tight_layout()
    return figure


def correlation_heatmap(correlations: pd.DataFrame) -> Figure:
    """Regressor correlation matrix, the raw material behind the VIFs."""
    figure = Figure(figsize=(6.5, 5.5))
    axes = figure.add_subplot(111)

    image = axes.imshow(correlations, cmap="RdBu_r", vmin=-1, vmax=1)
    labels = correlations.columns.tolist()
    axes.set_xticks(range(len(labels)))
    axes.set_xticklabels(labels, rotation=45, ha="right")
    axes.set_yticks(range(len(labels)))
    axes.set_yticklabels(labels)

    for row in range(len(labels)):
        for column in range(len(labels)):
            value = correlations.iloc[row, column]
            axes.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if abs(value) > 0.5 else "black",
            )

    figure.colorbar(image, ax=axes, shrink=0.8)
    axes.set_title("Factor correlations")
    figure.tight_layout()
    return figure


def cumulative_fit(regression: FactorRegression) -> Figure:
    """Cumulative actual excess return against what the factor model explains.

    The gap between the lines is cumulative alpha; it makes the intercept
    tangible in a way a coefficient table does not.
    """
    actual = regression.frame["excess"]
    explained = regression.fitted

    figure = Figure(figsize=(FIG_WIDTH, 4.5))
    axes = figure.add_subplot(111)
    axes.plot((1 + actual).cumprod(), linewidth=1.2, label="Actual excess return")
    axes.plot(
        (1 + explained).cumprod(),
        linewidth=1.2,
        linestyle="--",
        label=f"Explained by {regression.model}",
    )
    axes.set_yscale("log")
    axes.set_ylabel("Growth of 1 (log scale)")
    axes.set_title("Actual versus factor-explained excess return")
    axes.legend(fontsize=8)
    figure.tight_layout()
    return figure
