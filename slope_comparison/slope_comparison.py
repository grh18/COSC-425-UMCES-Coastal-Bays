import matplotlib.pyplot as plt
import numpy
import statsmodels.api as sm
import csv
import os

YEARCOL   = 2
DATACOL   = 9
CFACTOR   = 3
BASE_PATH = os.path.join(os.path.dirname(__file__), "..", "Nutrient_Clusters_I_VII")
SCRIPT_DIR = os.path.dirname(__file__)

SEGMENTS = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI", 7: "VII"}

NUTRIENTS = {
    "phosphate":   "orange",
    "nitrate":     "cyan",
    "silicate":    "lime",
    "ammonium":    "violet",
    "chlorophyll": "green",
}


def load_data(filepath, yearfirst, yearlast):
    xlist, ylist = [], []
    yearsum, samplenum = 0.0, 0
    curyear = yearfirst

    with open(filepath, encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))

    for row in rows:
        if row[YEARCOL] == "YEAR" or not row[YEARCOL].strip().lstrip("-").isdigit():
            continue
        yr = int(row[YEARCOL])
        if yr < yearfirst:
            continue
        if yr > yearlast:
            break
        if row[DATACOL] == "":
            continue

        val = float(row[DATACOL].replace(",", "")) / CFACTOR

        if yr == curyear:
            yearsum += val
            samplenum += 1
        else:
            if samplenum:
                xlist.append(float(curyear))
                ylist.append(yearsum / samplenum)
            yearsum = val
            samplenum = 1
            curyear = yr

    if samplenum:
        xlist.append(float(curyear))
        ylist.append(yearsum / samplenum)

    return numpy.array(xlist), numpy.array(ylist)


def ols_slope(x, y):
    if len(x) < 3:
        return float("nan")
    xc = sm.add_constant(x)
    return sm.OLS(y, xc).fit().params[1]


def run(yearfirst, yearlast, out_filename):
    nutrient_names = list(NUTRIENTS.keys())
    cluster_nums   = list(SEGMENTS.keys())
    slopes = {n: [] for n in nutrient_names}

    print(f"\n--- {yearfirst}–{yearlast} ---")
    for cluster_num in cluster_nums:
        for nutrient in nutrient_names:
            filepath = os.path.join(BASE_PATH, f"Cluster{cluster_num}", f"{nutrient}.csv")
            if not os.path.exists(filepath):
                print(f"  missing {filepath}")
                slopes[nutrient].append(float("nan"))
                continue
            x, y = load_data(filepath, yearfirst, yearlast)
            slope = ols_slope(x, y)
            slopes[nutrient].append(slope)
            print(f"  Cluster {SEGMENTS[cluster_num]} | {nutrient:12s} slope = {slope:.6f}")

    n_clusters  = len(cluster_nums)
    n_nutrients = len(nutrient_names)
    bar_width   = 0.15
    x_pos       = numpy.arange(n_clusters)

    fig, ax = plt.subplots(figsize=(14, 6))

    for i, nutrient in enumerate(nutrient_names):
        offset = (i - n_nutrients / 2 + 0.5) * bar_width
        ax.bar(x_pos + offset, slopes[nutrient], width=bar_width,
               color=NUTRIENTS[nutrient], label=nutrient.capitalize(),
               edgecolor="black", linewidth=0.5)

    ax.axhline(0, color="white", linewidth=0.8, linestyle="--")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([SEGMENTS[c] for c in cluster_nums])
    ax.set_xlabel("Coastal Bay Segment (Cluster)")
    ax.set_ylabel("OLS Slope (micromolars / year)")
    ax.set_title(f"Nutrient Trend Slopes by Segment ({yearfirst}–{yearlast})\n"
                 "Positive = increasing, Negative = decreasing")
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("white")
    ax.legend(loc="upper right", facecolor="#2a2a4e", labelcolor="white")

    plt.tight_layout()
    out_path = os.path.join(SCRIPT_DIR, out_filename)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


run(1995, 2008, "slope_comparison_1995_2008.png")
run(1995, 2021, "slope_comparison_1995_2021.png")
