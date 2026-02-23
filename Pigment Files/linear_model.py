"""
linear_model.py
===============
UMCES Pigment Data Analysis

Usage
-----
    python linear_model.py
"""

import os
import glob
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — no GUI needed
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
CSV_DIR     = os.path.join(BASE_DIR, "csv_output")
PLOT_DIR    = os.path.join(BASE_DIR, "plots")
README_PATH = os.path.join(BASE_DIR, "README.md")

os.makedirs(PLOT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Response variable configuration
# key → (display label, output PNG filename)
# ─────────────────────────────────────────────────────────────────────────────
RESPONSE_VARS = {
    "totchl":    ("Total Chl a (ug/L)",  "totchl.png"),
    "zea_chl":   ("Zeaxanthin / Chl",    "zea_chl.png"),
    "fuco_chl":  ("Fucoxanthin / Chl",   "fuco_chl.png"),
    "perid_chl": ("Peridinin / Chl",     "perid_chl.png"),
}


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Data Loading & Cleaning
# ─────────────────────────────────────────────────────────────────────────────

def load_data(csv_dir: str) -> pd.DataFrame:
    """
    Load Sheet3 CSVs (the only files with proper column headers across all
    station groups).  Each Sheet3 file contains columns:
        STA, DATE, YEAR, MONTH, [TChl a], Perid, But fuco, Fuco, Zea
    plus annual-average duplicate columns ([TChl a].1, etc.) which are dropped.

    Returns a tidy DataFrame with standardised names and computed ratios.
    """
    # Collect both naming variants (Sheet3 vs sheet 3 for Stations 5,7,14)
    files = sorted(
        glob.glob(os.path.join(csv_dir, "*_Sheet3.csv")) +
        glob.glob(os.path.join(csv_dir, "*_sheet 3.csv"))
    )
    if not files:
        raise FileNotFoundError(f"No Sheet3 CSV files found in {csv_dir}")

    frames = []
    for fp in files:
        df = pd.read_csv(fp)
        # Keep only the primary pigment columns; skip .1 annual-average columns
        keep = [c for c in ["STA", "DATE", "YEAR", "MONTH",
                             "[TChl a]", "Perid", "But fuco", "Fuco", "Zea"]
                if c in df.columns]
        frames.append(df[keep])

    raw = pd.concat(frames, ignore_index=True)

    # ── Standardise column names ──────────────────────────────────────────────
    # Lower-case; replace brackets, parens, spaces, # with underscore; strip edges
    raw.columns = (
        raw.columns
           .str.lower()
           .str.replace(r"[\[\]\(\)\s#]+", "_", regex=True)
           .str.strip("_")
    )
    # After transformation:  STA→sta  [TChl a]→tchl_a  But fuco→but_fuco  etc.
    rename_map = {
        "sta":    "station",
        "tchl_a": "totchl",
        "perid":  "peridinin",
    }
    raw.rename(columns=rename_map, inplace=True)

    # ── Convert to numeric ────────────────────────────────────────────────────
    raw["station"] = pd.to_numeric(raw["station"], errors="coerce")
    raw.dropna(subset=["station"], inplace=True)
    raw["station"] = raw["station"].astype(int)

    for col in ["year", "totchl", "peridinin", "but_fuco", "fuco", "zea"]:
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")

    # ── Compute pigment / Chl ratios ──────────────────────────────────────────
    # Replace zero totchl with NaN to avoid division-by-zero
    valid_chl = raw["totchl"].replace(0, np.nan)
    raw["zea_chl"]   = raw["zea"]       / valid_chl
    raw["fuco_chl"]  = raw["fuco"]      / valid_chl
    raw["perid_chl"] = raw["peridinin"] / valid_chl

    # Clean up any infinities from the division
    raw.replace([np.inf, -np.inf], np.nan, inplace=True)

    # ── Drop rows missing any key response variable ───────────────────────────
    raw.dropna(subset=list(RESPONSE_VARS.keys()), inplace=True)

    # Remove exact duplicate station+date records (Sheet1 == Sheet2 in most files)
    raw.drop_duplicates(subset=["station", "date"], inplace=True)
    raw.reset_index(drop=True, inplace=True)

    print(f"Loaded {len(raw):,} observations  |  "
          f"{raw['station'].nunique()} stations: {sorted(raw['station'].unique())}")
    return raw


# ─────────────────────────────────────────────────────────────────────────────
# Linear regression helper
# ─────────────────────────────────────────────────────────────────────────────

def fit_ols(x_vals, y_vals) -> dict | None:
    """
    Ordinary-least-squares via scipy.stats.linregress.
    Returns a result dict or None when fewer than 3 paired finite observations.
    """
    x = np.asarray(x_vals, dtype=float)
    y = np.asarray(y_vals, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    if len(x) < 3:
        return None

    slope, intercept, r, p, se = stats.linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 300)
    y_line = slope * x_line + intercept

    return dict(
        slope=slope, intercept=intercept,
        r2=r ** 2, p_value=p, n=len(x), se=se,
        x_data=x, y_data=y,
        x_line=x_line, y_line=y_line,
    )


def sig_stars(p: float) -> str:
    """Return significance stars string for a p-value."""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Model Fitting
# ─────────────────────────────────────────────────────────────────────────────

def run_models(df: pd.DataFrame) -> dict:
    """
    For each response variable, fit:
      (a) Pooled model:        X = station number  (tests spatial gradient)
      (b) Per-station models:  X = calendar year   (tests temporal trend per site)

    Prints full summaries and returns a nested results dict:
        results[var]["pooled"]         → fit dict or None
        results[var]["by_station"]     → {station_id: fit dict or None}
    """
    results = {}

    for var, (label, _) in RESPONSE_VARS.items():
        results[var] = {}

        print(f"\n{'=' * 65}")
        print(f"  RESPONSE: {label}")
        print(f"{'=' * 65}")

        # ── (a) Pooled model: X = station ─────────────────────────────────────
        fit = fit_ols(df["station"], df[var])
        results[var]["pooled"] = fit

        if fit:
            print(f"\n  [POOLED — X = station number, n = {fit['n']}]")
            print(f"    slope     = {fit['slope']:+.6f}")
            print(f"    intercept = {fit['intercept']:.6f}")
            print(f"    R²        = {fit['r2']:.4f}")
            print(f"    p-value   = {fit['p_value']:.3e}  {sig_stars(fit['p_value'])}")
        else:
            print("  [POOLED]  insufficient data")

        # ── (b) Per-station temporal models: X = year ─────────────────────────
        results[var]["by_station"] = {}
        print(f"\n  [PER-STATION — X = year]")
        print(f"  {'STA':>4}  {'slope':>11}  {'intercept':>12}  "
              f"{'R²':>7}  {'p-value':>12}  {'n':>5}  sig")
        print(f"  {'-' * 63}")

        for sta in sorted(df["station"].unique()):
            sub   = df[df["station"] == sta]
            fit_s = fit_ols(sub["year"], sub[var])
            results[var]["by_station"][sta] = fit_s

            if fit_s:
                print(f"  {sta:>4}  {fit_s['slope']:>+11.5f}  "
                      f"{fit_s['intercept']:>12.4f}  {fit_s['r2']:>7.4f}  "
                      f"{fit_s['p_value']:>12.3e}  {fit_s['n']:>5}  "
                      f"{sig_stars(fit_s['p_value'])}")
            else:
                print(f"  {sta:>4}  -- too few data points --")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Plots
# ─────────────────────────────────────────────────────────────────────────────

def station_colormap(stations: pd.Series) -> dict:
    """Return {station_id: RGBA} from the tab20 colormap."""
    unique = sorted(stations.unique())
    cmap   = plt.cm.tab20
    n      = max(len(unique) - 1, 1)
    return {sta: cmap(i / n) for i, sta in enumerate(unique)}


def plot_variable(df, var, label, filename, fit, cmap):
    """
    Scatter plot with one point per observation, coloured by station.
    A small horizontal jitter prevents all points from stacking at the same
    integer X position.  The pooled OLS line is drawn in black.
    """
    rng    = np.random.default_rng(42)
    jitter = rng.uniform(-0.25, 0.25, len(df))

    fig, ax = plt.subplots(figsize=(10, 5))

    # Draw points per station so each gets its own legend entry
    for sta in sorted(df["station"].unique()):
        idx = df.index[df["station"] == sta]
        ax.scatter(
            df.loc[idx, "station"] + jitter[idx],
            df.loc[idx, var],
            color=cmap[sta], alpha=0.55, s=18, label=f"St {sta}",
        )

    # Overlay pooled regression line with annotation
    if fit:
        ax.plot(fit["x_line"], fit["y_line"],
                "k-", linewidth=2.0, zorder=5, label="OLS fit")
        annotation = (
            f"R² = {fit['r2']:.3f}   p = {fit['p_value']:.2e}   "
            f"{sig_stars(fit['p_value'])}\n"
            f"slope = {fit['slope']:+.4f}   intercept = {fit['intercept']:.4f}"
        )
        ax.text(0.02, 0.97, annotation,
                transform=ax.transAxes, fontsize=8.5, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8))

    ax.set_xlabel("Station Number", fontsize=11)
    ax.set_ylabel(label, fontsize=11)
    ax.set_title(f"{label} vs Station  (all observations pooled, n = {len(df)})",
                 fontsize=12)
    ax.set_xticks(sorted(df["station"].unique()))

    # Compact legend outside axes if there are many stations
    handles, labs = ax.get_legend_handles_labels()
    ax.legend(handles, labs, fontsize=7, ncol=3, loc="upper right",
              framealpha=0.7, markerscale=1.5)

    plt.tight_layout()
    out = os.path.join(PLOT_DIR, filename)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_overview(df, results, cmap):
    """
    2×2 faceted figure showing all four response variables in one image.
    Each panel has the pooled regression line and R²/p annotation.
    Saved as plots/all_pigments_overview.png.
    """
    rng    = np.random.default_rng(42)
    jitter = rng.uniform(-0.25, 0.25, len(df))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Pigment Variables vs Station  (all observations pooled)",
                 fontsize=13, fontweight="bold")

    for ax, (var, (label, _)) in zip(axes.flatten(), RESPONSE_VARS.items()):
        # Scatter points colour-coded by station
        for sta in sorted(df["station"].unique()):
            idx = df.index[df["station"] == sta]
            ax.scatter(
                df.loc[idx, "station"] + jitter[idx],
                df.loc[idx, var],
                color=cmap[sta], alpha=0.45, s=12,
            )

        # Regression line and title annotation
        fit = results[var]["pooled"]
        if fit:
            ax.plot(fit["x_line"], fit["y_line"], "k-", linewidth=1.6, zorder=5)
            ax.set_title(
                f"{label}\n"
                f"R²={fit['r2']:.3f}  p={fit['p_value']:.1e}  "
                f"{sig_stars(fit['p_value'])}",
                fontsize=9.5,
            )
        else:
            ax.set_title(label, fontsize=9.5)

        ax.set_xlabel("Station", fontsize=9)
        ax.set_ylabel(label, fontsize=9)
        ax.set_xticks(sorted(df["station"].unique()))
        ax.tick_params(axis="x", labelsize=7)

    plt.tight_layout()
    out = os.path.join(PLOT_DIR, "all_pigments_overview.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved overview: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  README generation
# ─────────────────────────────────────────────────────────────────────────────

def write_readme(df: pd.DataFrame, results: dict) -> None:
    """
    Auto-generate README.md with:
      - Dataset description
      - Per-variable pooled model results table + interpretation
      - Per-variable temporal trend summary (per-station)
      - Notable observation highlights (highest/lowest/most variable station)
      - Cross-pigment pattern summary
    """
    stations = sorted(df["station"].unique())
    yr_min, yr_max = int(df["year"].min()), int(df["year"].max())

    lines = [
        "# UMCES Pigment Data — Linear Model Analysis\n\n",
        "## Dataset Description\n\n",
        f"Oceanographic pigment monitoring data from **{df['station'].nunique()} "
        f"stations** (stations {stations[0]}–{stations[-1]}) spanning "
        f"**{yr_min}–{yr_max}** ({yr_max - yr_min + 1} years).  "
        f"After cleaning and deduplication, **{len(df):,} observations** were used.\n\n",
        "**Response variables analysed:**\n",
        "- `totchl` — Total Chlorophyll *a* concentration (μg/L)\n",
        "- `zea_chl` — Zeaxanthin normalised to Chl *a*\n",
        "- `fuco_chl` — Fucoxanthin normalised to Chl *a*\n",
        "- `perid_chl` — Peridinin normalised to Chl *a*\n\n",
        "**Predictor — pooled models:** Station number (numeric spatial proxy "
        "for along-transect position).\n",
        "**Predictor — per-station models:** Calendar year (temporal trend "
        "at each individual site).\n\n",
        "---\n\n",
        "## Results by Response Variable\n\n",
    ]

    for var, (label, fname) in RESPONSE_VARS.items():
        fit   = results[var]["pooled"]
        by_st = results[var]["by_station"]

        lines.append(f"### {label}\n\n")
        lines.append(f"**Plot:** [`plots/{fname}`](plots/{fname})\n\n")

        # ── Pooled model table ────────────────────────────────────────────────
        if fit:
            sig = sig_stars(fit["p_value"])
            lines += [
                "#### Pooled Model (X = station)\n\n",
                "| Metric | Value |\n",
                "|--------|-------|\n",
                f"| Slope | `{fit['slope']:+.6f}` |\n",
                f"| Intercept | `{fit['intercept']:.6f}` |\n",
                f"| R² | `{fit['r2']:.4f}` |\n",
                f"| p-value | `{fit['p_value']:.3e}` ({sig}) |\n",
                f"| n | `{fit['n']}` |\n\n",
            ]
            if fit["p_value"] < 0.05:
                direction = "increases" if fit["slope"] > 0 else "decreases"
                lines.append(
                    f"> **Interpretation:** Significant spatial trend detected "
                    f"(p = {fit['p_value']:.2e}, {sig}).  "
                    f"{label} **{direction} with increasing station number** "
                    f"(R² = {fit['r2']:.3f}).\n\n"
                )
            else:
                lines.append(
                    f"> **Interpretation:** No significant spatial trend "
                    f"(p = {fit['p_value']:.2e}, R² = {fit['r2']:.3f}).  "
                    f"Station position alone does not explain variation in {label}.\n\n"
                )
        else:
            lines.append("> Insufficient data for pooled model.\n\n")

        # ── Per-station temporal summary ──────────────────────────────────────
        sig_stas = [int(sta) for sta, f in by_st.items() if f and f["p_value"] < 0.05]
        neg_stas = [int(sta) for sta, f in by_st.items()
                    if f and f["p_value"] < 0.05 and f["slope"] < 0]
        pos_stas = [int(sta) for sta, f in by_st.items()
                    if f and f["p_value"] < 0.05 and f["slope"] > 0]

        lines.append(
            f"#### Per-Station Temporal Trends (X = year)\n\n"
            f"**{len(sig_stas)}/{len(by_st)}** stations show a significant "
            f"temporal trend (p < 0.05).\n\n"
        )
        if neg_stas:
            lines.append(f"- Declining over time: stations **{neg_stas}**\n")
        if pos_stas:
            lines.append(f"- Increasing over time: stations **{pos_stas}**\n")
        lines.append("\n")

        # ── Notable observations ──────────────────────────────────────────────
        grp      = df.groupby("station")[var]
        high_sta = grp.mean().idxmax()
        low_sta  = grp.mean().idxmin()
        cv       = (grp.std() / grp.mean()).dropna()
        cv_sta   = cv.idxmax() if len(cv) else "N/A"

        lines.append(
            f"#### Notable Observations\n\n"
            f"- **Highest mean {label}:** Station {high_sta} "
            f"(mean = {grp.mean()[high_sta]:.4f})\n"
            f"- **Lowest mean {label}:** Station {low_sta} "
            f"(mean = {grp.mean()[low_sta]:.4f})\n"
            f"- **Most variable station (CV):** Station {cv_sta}\n\n"
            f"---\n\n"
        )

    # ── Cross-pigment section ─────────────────────────────────────────────────
    lines.append("## Cross-Pigment Patterns\n\n")

    # Identify stations that are high outliers (> mean + 1 SD) across variables
    high_counts: dict = {}
    for var in RESPONSE_VARS:
        grp = df.groupby("station")[var].mean()
        threshold = grp.mean() + grp.std()
        for sta in grp[grp > threshold].index:
            high_counts[sta] = high_counts.get(sta, 0) + 1

    multi_high = [int(sta) for sta, cnt in sorted(high_counts.items()) if cnt >= 2]
    if multi_high:
        lines.append(
            f"Stations **{multi_high}** show consistently above-average values "
            f"across **multiple pigment variables**, suggesting elevated productivity "
            f"or a distinctive phytoplankton community at these locations.\n\n"
        )
    else:
        lines.append(
            "No single station was a consistent high outlier across "
            "all four pigment variables.\n\n"
        )

    # Summary of significant spatial gradients
    sig_spatial = [
        f"`{var}` (R²={results[var]['pooled']['r2']:.3f}, "
        f"p={results[var]['pooled']['p_value']:.2e})"
        for var in RESPONSE_VARS
        if results[var]["pooled"] and results[var]["pooled"]["p_value"] < 0.05
    ]
    if sig_spatial:
        lines.append(
            f"Significant along-transect spatial gradients (p < 0.05) were detected "
            f"for: {', '.join(sig_spatial)}.  This implies systematic spatial "
            f"structuring of phytoplankton communities along the monitoring transect.\n\n"
        )
    else:
        lines.append(
            "No response variable showed a significant pooled spatial gradient.  "
            "Within-station temporal variability likely dominates over "
            "between-station spatial differences in this dataset.\n\n"
        )

    with open(README_PATH, "w", encoding="utf-8") as fh:
        fh.writelines(lines)
    print(f"\nREADME written: {README_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 1. Load and clean data
    df = load_data(CSV_DIR)

    # 2. Fit models and print summaries
    results = run_models(df)

    # 3. Build colour map (consistent across all plots)
    cmap = station_colormap(df["station"])

    # 4. Generate individual variable plots
    print("\n-- Generating plots --")
    for var, (label, fname) in RESPONSE_VARS.items():
        plot_variable(df, var, label, fname, results[var]["pooled"], cmap)

    # 4b. Overview faceted figure
    plot_overview(df, results, cmap)

    # 5. Write README
    write_readme(df, results)

    print("\nDone.")
