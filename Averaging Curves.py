"""
RT-QuIC average kinetic traces by diagnosis group
You need the Excel with time_hours column + one column per case
       and also the Metadata Excel with case and diagnosis columns
The output is figure with mean ± SD traces per group
"""

import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter1d
import matplotlib.ticker as mticker

# =========================================
# ====== CONFIGURATION ====================
# =========================================
WORK_DIR      = r"/Users/mw4217/Desktop/RTQuic calculations/CSF/"
CURVES_FILE   = "/Users/mw4217/Desktop/RTQuic calculations/CSF_curves.xlsx"
METADATA_FILE = "/Users/mw4217/Desktop/RTQuic calculations/CSF_metadata.xlsx"

CASE_COL      = "case"
GROUP_COL     = "diagnosis"

# Group display order
GROUP_ORDER   = ["MSA", "PD", "DLB", "iLBD", "Control"]

# Smoothing (0 means off)
SMOOTH_SIGMA  = 0.8

# =========================
# === SETUP ================
# =========================
PARAMETERS = {
    "font.family":          "sans-serif",
    "font.sans-serif":      ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size":            7,
    "axes.titlesize":       8,
    "axes.labelsize":       7,
    "xtick.labelsize":      6,
    "ytick.labelsize":      6,
    "legend.fontsize":      6,
    "axes.linewidth":       0.75,
    "xtick.major.width":    0.75,
    "ytick.major.width":    0.75,
    "xtick.major.size":     3,
    "ytick.major.size":     3,
    "xtick.direction":      "out",
    "ytick.direction":      "out",
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "axes.grid":            False,
    "figure.dpi":           300,
    "savefig.dpi":          300,
    "savefig.bbox":         "tight",
    "savefig.pad_inches":   0.05,
    "figure.facecolor":     "white",
    "axes.facecolor":       "white",
    "legend.frameon":       False,
}
mpl.rcParams.update(PARAMETERS)

GROUP_COLORS = {
    "MSA":     "#D62728",
    "PD":      "#2166AC",
    "DLB":     "#54A24B",
    "iLBD":    "#F4A700",
    "Control": "#888888",
}

GREY_DARK = "#333333"
GREY_MID  = "#888888"

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return os.path.abspath(path)

def clean_str(s):
    return (str(s)
            .replace("\xa0", " ")
            .replace("\u200b", "")
            .strip())

# ==============================================
# ====== LOAD CURVES ===========================
# ==============================================
WORK_DIR = ensure_dir(WORK_DIR)

print("[INFO] Loading curves...")
curves_raw = pd.read_excel(CURVES_FILE, engine="openpyxl")
curves_raw.columns = [clean_str(c) for c in curves_raw.columns]
curves_raw = curves_raw.loc[:, ~curves_raw.columns.str.startswith("unnamed")]

# First column = time
time_col  = curves_raw.columns[0]
curves_raw = curves_raw.dropna(subset=[time_col]).reset_index(drop=True)
time      = curves_raw[time_col].values.astype(float)

case_cols_raw   = [c for c in curves_raw.columns if c != time_col]
case_cols_clean = [c.replace(" MSA", "").replace("MSA", "").strip()
                   for c in case_cols_raw]

rename_dict = {old: new for old, new in
               zip(case_cols_raw, case_cols_clean)}
curves_raw  = curves_raw.rename(columns=rename_dict)
case_cols   = case_cols_clean

# Convert to numeric
for c in case_cols:
    curves_raw[c] = pd.to_numeric(curves_raw[c], errors="coerce")

print(f"  Timepoints : {len(time)}  ({time[0]:.2f} to {time[-1]:.2f} h)")
print(f"  Cases      : {len(case_cols)}")

# ==============================================
# ====== LOAD METADATA =========================
# ==============================================
print("[INFO] Loading metadata...")
meta = pd.read_excel(METADATA_FILE, engine="openpyxl")
meta.columns = [clean_str(c).lower().replace(" ", "_")
                for c in meta.columns]
meta[CASE_COL]  = meta[CASE_COL].apply(clean_str)
meta[GROUP_COL] = meta[GROUP_COL].apply(clean_str)

print(f"  Cases in metadata: {len(meta)}")
print(f"\n  Group counts:")
print(meta[GROUP_COL].value_counts().to_string())

# =====================Check matching=====================
matched = [c for c in case_cols if c in meta[CASE_COL].values]
missing = [c for c in case_cols if c not in meta[CASE_COL].values]
print(f"\n  Matched: {len(matched)} / {len(case_cols)} cases")
if missing:
    print(f"  [WARNING] Unmatched cases: {missing}")

# ==============================================
# ===== CURVE STATISTICS =======================
# ==============================================
def group_stats(case_list):
    """
    Return smoothed mean and SD across cases.
    Returns nan arrays if no valid cases.
    """
    valid = [c for c in case_list if c in curves_raw.columns]
    if not valid:
        return (np.full(len(time), np.nan),
                np.full(len(time), np.nan),
                0)
    data = curves_raw[valid].values.astype(float)
    mean = np.nanmean(data, axis=1)
    sd   = np.nanstd(data,  axis=1)
    if SMOOTH_SIGMA > 0:
        mean = gaussian_filter1d(mean, sigma=SMOOTH_SIGMA)
        sd   = gaussian_filter1d(sd,   sigma=SMOOTH_SIGMA)
    return mean, sd, len(valid)

group_stats_dict = {}
for grp in GROUP_ORDER:
    cases = meta.loc[meta[GROUP_COL] == grp, CASE_COL].tolist()
    mean, sd, n = group_stats(cases)
    group_stats_dict[grp] = {"mean": mean, "sd": sd, "n": n, "cases": cases}
    print(f"  {grp}: n={n} curves")

groups_with_data = [g for g in GROUP_ORDER
                    if group_stats_dict[g]["n"] > 0]

# ==============================================
# ======== FIGURES ===============================
# A — All groups mean ± SD overlaid
# B — Individual traces + mean per group
# C — One panel per group
# ==============================================
n_groups = len(groups_with_data)

fig = plt.figure(figsize=(300/25.4, 180/25.4))

gs_main = mpl.gridspec.GridSpec(
    2, 2,
    figure=fig,
    left=0.05, right=0.97,
    top=0.93,  bottom=0.08,
    wspace=0.9, hspace=0.55,
    height_ratios=[1, 1.2]
)

# ---- PANEL A: All groups mean ± SD overlaid ----
ax_A = fig.add_subplot(gs_main[0, 0])

for grp in groups_with_data:
    d   = group_stats_dict[grp]
    col = GROUP_COLORS.get(grp, GREY_MID)
    ax_A.fill_between(time,
                      d["mean"] - d["sd"],
                      d["mean"] + d["sd"],
                      color=col, alpha=0.12, zorder=1)
    ax_A.plot(time, d["mean"],
              color=col, lw=1.2, zorder=2,
              label=f"{grp} (n={d['n']})")

ax_A.set_xlabel("Time (hours)", fontsize=7, labelpad=3)
ax_A.set_ylabel("Fluorescence (RFU)", fontsize=7, labelpad=3)
ax_A.set_title("Mean ± SD by diagnosis", fontsize=8,
                fontweight="bold", pad=4)
ax_A.set_xlim(time[0], time[-1])
ax_A.set_ylim(-5000, 260000)
ax_A.legend(fontsize=5.5, frameon=False,
            loc="upper left",
            handlelength=1.2, labelspacing=0.3)
ax_A.tick_params(labelsize=6, length=3)
for spine in ax_A.spines.values():
    spine.set_linewidth(0.75)
ax_A.text(-0.18, 1.04, "A",
           transform=ax_A.transAxes,
           fontsize=10, fontweight="bold")

# ---- PANEL B: Individual traces + mean overlay ----
ax_B = fig.add_subplot(gs_main[0, 1])

for grp in groups_with_data:
    d   = group_stats_dict[grp]
    col = GROUP_COLORS.get(grp, GREY_MID)
    # Individual traces — thin and transparent
    valid = [c for c in d["cases"] if c in curves_raw.columns]
    for c in valid:
        ax_B.plot(time, curves_raw[c].values,
                  color=col, lw=0.35, alpha=0.3, zorder=1)
    # Mean on top — thick and solid
    ax_B.plot(time, d["mean"],
              color=col, lw=1.5, zorder=3,
              label=f"{grp} (n={d['n']})")

ax_B.set_xlabel("Time (hours)", fontsize=7, labelpad=3)
ax_B.set_ylabel("Fluorescence (RFU)", fontsize=7, labelpad=3)
ax_B.set_title("Individual traces by diagnosis", fontsize=8,
                fontweight="bold", pad=4)
ax_B.set_xlim(time[0], time[-1])
ax_B.legend(fontsize=5.5, frameon=False,
            loc="upper left",
            handlelength=1.2, labelspacing=0.3)
ax_B.tick_params(labelsize=6, length=3)
for spine in ax_B.spines.values():
    spine.set_linewidth(0.75)
ax_B.text(-0.18, 1.04, "B",
           transform=ax_B.transAxes,
           fontsize=10, fontweight="bold")

# ---- PANEL C: One panel per group ----
gs_small = mpl.gridspec.GridSpecFromSubplotSpec(
    1, n_groups,
    subplot_spec=gs_main[1, :],
    wspace=0.25,
)

panel_labels = "ABCDE"

for idx, grp in enumerate(groups_with_data):
    ax = fig.add_subplot(gs_small[0, idx])
    d   = group_stats_dict[grp]
    col = GROUP_COLORS.get(grp, GREY_MID)

    # Individual traces
    valid = [c for c in d["cases"] if c in curves_raw.columns]
    for c in valid:
        ax.plot(time, curves_raw[c].values,
                color=col, lw=0.4, alpha=0.35, zorder=1)

    # SD ribbon
    ax.fill_between(time,
                    d["mean"] - d["sd"],
                    d["mean"] + d["sd"],
                    color=col, alpha=0.2, zorder=2)

    # Mean
    ax.plot(time, d["mean"],
            color=col, lw=1.5, zorder=3)

    ax.set_xlim(time[0], time[-1])
    ax.set_ylim(-5000, 260000)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.set_xlabel("Time (hours)", fontsize=6, labelpad=2)
    ax.set_ylabel("Fluorescence (RFU)", fontsize=6, labelpad=2)
    ax.set_title(f"{grp}\n(n={d['n']})",
                  fontsize=7, fontweight="bold",
                  color=col, pad=3)
    ax.tick_params(labelsize=5.5, length=2.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.75)


    # Panel label
    ax.text(-0.2, 1.04, panel_labels[idx],
             transform=ax.transAxes,
             fontsize=10, fontweight="bold")

fig.suptitle("RT-QuIC kinetic traces by diagnosis",
             fontsize=9, fontweight="bold", y=0.97)

fig.savefig(
    os.path.join(WORK_DIR, "Figure_traces_by_diagnosis.png"),
    dpi=300, facecolor="white"
)
fig.savefig(
    os.path.join(WORK_DIR, "Figure_traces_by_diagnosis.pdf"),
    facecolor="white"
)
plt.close(fig)
print("\n[INFO] Saved: Figure_traces_by_diagnosis.png / .pdf")

# ==============================================
# ====== SAVE MEAN TRACES ======================
# ==============================================
mean_out = pd.DataFrame({"time_hours": time})
for grp in groups_with_data:
    d = group_stats_dict[grp]
    mean_out[f"{grp}_mean"] = d["mean"]
    mean_out[f"{grp}_sd"]   = d["sd"]

mean_out.to_excel(
    os.path.join(WORK_DIR, "mean_traces_by_diagnosis.xlsx"),
    index=False
)
print("[INFO] Saved: mean_traces_by_diagnosis.xlsx")

# ==============================================
# ====== CONSOLE SUMMARY =======================
# ==============================================
print("\n" + "="*45)
print("[SUMMARY] Cases per group:")
for grp in groups_with_data:
    print(f"  {grp:10s}: n={group_stats_dict[grp]['n']}")
print("="*45)