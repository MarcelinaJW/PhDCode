"""
RT-QuIC PCA + K-means clustering
"""

import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch, FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from scipy.stats import mannwhitneyu
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import seaborn as sns
import diptest

# ===============================================
# ====== CONFIGURATIONS =========================
# ===============================================
WORK_DIR     = r"/Users/mw4217/Desktop/RTQuic calculations/Seeding/"
RTQUIC_FILE  = "/Users/mw4217/Desktop/RTQuic calculations/RTQuIC Groups.xlsx"
SHEET_RTQUIC = 0
FEATURE_COLS = ["lag", "auc", "maxslope", "fmax", "slope"]
VARIANCE_THRESHOLD = 0.93
#0.9 csf, 0.95 bh

PARAMETERS = {
    # Font
    "font.family":          "sans-serif",
    "font.sans-serif":      ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size":            7,
    "axes.titlesize":       8,
    "axes.labelsize":       7,
    "xtick.labelsize":      6,
    "ytick.labelsize":      6,
    "legend.fontsize":      6,
    "figure.titlesize":     9,
    # Lines and ticks
    "axes.linewidth":       0.75,
    "xtick.major.width":    0.75,
    "ytick.major.width":    0.75,
    "xtick.minor.width":    0.5,
    "ytick.minor.width":    0.5,
    "xtick.major.size":     3,
    "ytick.major.size":     3,
    "xtick.minor.size":     1.5,
    "ytick.minor.size":     1.5,
    "xtick.direction":      "out",
    "ytick.direction":      "out",
    "lines.linewidth":      1.0,
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    # Grid — minimal
    "axes.grid":            False,
    # Figure
    "figure.dpi":           300,
    "savefig.dpi":          300,
    "savefig.bbox":         "tight",
    "savefig.pad_inches":   0.05,
    # White background
    "figure.facecolor":     "white",
    "axes.facecolor":       "white",
    # Legend
    "legend.frameon":       False,
    "legend.handlelength":  1.0,
    "legend.handleheight":  0.7,
    "legend.handletextpad": 0.4,
    "legend.borderpad":     0.3,
    "legend.labelspacing":  0.3,
}
mpl.rcParams.update(PARAMETERS)

#====================== Colour palette ======================
C1_COLOR  = "#2166AC"   # deep blue  — Cluster 1
C2_COLOR  = "#F4A700"   # golden yellow — Cluster 2
GREY_DARK = "#333333"
GREY_MID  = "#888888"
GREY_LITE = "#DDDDDD"

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return os.path.abspath(path)

def clean_col(c):
    return (
        str(c).strip()
        .replace("\u00A0", " ")
        .replace("\u200b", "")
        .lower()
        .replace(" ", "_")
    )

def find_col(cols, *candidates):
    for cand in candidates:
        if cand in cols:
            return cand
    for cand in candidates:
        hits = [c for c in cols if cand in c]
        if hits:
            return hits[0]
    return None

def p_to_stars(p):
    """Convert p-value to significance stars."""
    if p < 0.0001: return "****"
    if p < 0.001:  return "***"
    if p < 0.01:   return "**"
    if p < 0.05:   return "*"
    return "ns"

def add_significance_bar(ax, x1, x2, y, p, h=0.02):
    """
    Draw a significance bar between two groups.
    y: top of bar in data coordinates
    h: height of the descending ticks as fraction of y-range
    """
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    tick_h  = h * y_range
    stars   = p_to_stars(p)
    color   = GREY_DARK if stars != "ns" else GREY_MID

    ax.plot([x1, x1, x2, x2],
            [y - tick_h, y, y, y - tick_h],
            lw=0.75, color=color, solid_capstyle="round")
    ax.text((x1 + x2) / 2, y + 0.005 * y_range,
            stars, ha="center", va="bottom",
            fontsize=7 if stars != "ns" else 6,
            color=color,
            fontweight="bold" if stars != "ns" else "normal")

def format_p(p):
    if p < 0.0001:
        return "p < 0.0001"
    if p < 0.001:
        return f"p = {p:.4f}"
    return f"p = {p:.3f}"

# =========================================================
# ====== LOAD DATA & PROCESS DATA =========================
# =========================================================
WORK_DIR = ensure_dir(WORK_DIR)

rq = pd.read_excel(RTQUIC_FILE, engine="openpyxl",
                   sheet_name=SHEET_RTQUIC, header=0)
rq.columns = [clean_col(c) for c in rq.columns]
rq = rq.loc[:, ~rq.columns.str.startswith("unnamed")]
cols = list(rq.columns)

case     = find_col(cols, "case")
auc      = find_col(cols, "area_under_the_curve", "auc")
fmx      = find_col(cols, "fmax", "f_max")
lag      = find_col(cols, "lag_(1/time_to_threshold)", "lag", "1/time")
slope    = find_col(cols, "slope", "max_slope", "growth_rate")
maxslope = find_col(cols, "maxslope", "max_slope")

rename_rq = {}
if case:     rename_rq[case]     = "case"
if auc:      rename_rq[auc]      = "auc"
if fmx:      rename_rq[fmx]      = "fmax"
if lag:      rename_rq[lag]      = "lag"
if slope:    rename_rq[slope]    = "slope"
if maxslope: rename_rq[maxslope] = "maxslope"
rq = rq.rename(columns=rename_rq)

rq["case"] = rq["case"].astype(str).str.strip()
for c in ["auc", "fmax", "lag", "slope", "maxslope"]:
    rq[c] = pd.to_numeric(rq[c], errors="coerce")

#rq["log10_auc"]      = np.log10(rq["auc"].clip(lower=1e-12))
#rq["log10_fmax"]     = np.log10(rq["fmax"].clip(lower=1e-12))
#rq["log10_slope"]    = np.log10(rq["slope"].clip(lower=1e-12))
#rq["log10_maxslope"] = np.log10(rq["maxslope"].clip(lower=1e-12))

df = rq.dropna(subset=FEATURE_COLS).copy()

# Normalise the data
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(df[FEATURE_COLS].values)

# ======================PRINCIPAL COMPONENT ANALYSIS==============================
pca_full    = PCA(random_state=42).fit(X_scaled)
explained   = pca_full.explained_variance_ratio_
cumulative  = np.cumsum(explained)
n_comp_auto = int(np.argmax(cumulative >= VARIANCE_THRESHOLD) + 1)
N_COMPONENTS = 2  # fixed, not auto-selected

pca    = PCA(n_components=N_COMPONENTS, random_state=42)
X_pca  = pca.fit_transform(X_scaled)
pc_cols = [f"PC{i+1}" for i in range(n_comp_auto)]
for i, col in enumerate(pc_cols):
    df[col] = X_pca[:, i]

# K-means k=2 on PCA scores
km2           = KMeans(n_clusters=2, random_state=42, n_init=50)
df["cluster_raw"] = km2.fit_predict(X_pca)

cluster_lag_means = df.groupby("cluster_raw")["lag"].mean()
high_cluster      = int(cluster_lag_means.idxmax())

df["cluster"]    = np.where(df["cluster_raw"] == high_cluster,
                             "Cluster 1", "Cluster 2")
df["cluster_num"] = np.where(df["cluster_raw"] == high_cluster, 1, 2)

palette = {"Cluster 1": C1_COLOR, "Cluster 2": C2_COLOR}
colors  = df["cluster"].map(palette)

# Cluster metrics
sil_k2  = silhouette_score(X_pca, km2.labels_)
n_c1    = (df["cluster"] == "Cluster 1").sum()
n_c2    = (df["cluster"] == "Cluster 2").sum()

# MWU
mwu_results = {}
for feat in FEATURE_COLS:
    c1  = df.loc[df["cluster"] == "Cluster 1", feat].dropna()
    c2  = df.loc[df["cluster"] == "Cluster 2", feat].dropna()
    u, p = mannwhitneyu(c1, c2, alternative="two-sided")
    r   = 1 - (2 * u) / (len(c1) * len(c2))
    mwu_results[feat] = {"U": u, "p": p, "r": r}

print(f"[INFO] PCA: retained {n_comp_auto} components "
      f"({cumulative[n_comp_auto-1]*100:.1f}% variance)")
print(f"[INFO] K-means: Cluster 1 n={n_c1}, Cluster 2 n={n_c2}")
print(f"[INFO] Silhouette: {sil_k2:.3f}")

# =============================================================
# FIGURE A + B=================================================
# A: PC1 vs PC2 scatterplot====================================
# B: Kinetic parameter boxplots with MWU stats=================
# =============================================================
feat_labels = {
    "lag":            "Lag\n(1/time to threshold)",
    "auc":      "AUC",
    "maxslope": "Max slope",
    "fmax":     "Fmax",
    "slope":    "Slope",
}

n_feats  = len(FEATURE_COLS)


fig = plt.figure(figsize=(183/25.4, 200/25.4))

gs = GridSpec(
    2, 5,                    # 2 rows, 5 columns
    figure=fig,
    left=0.08, right=0.97,
    top=0.95,  bottom=0.08,
    wspace=0.25,
    hspace=0.25,
)

# ---- PANEL A: PC1 vs PC2 scatter
ax_scatter = fig.add_subplot(gs[0, 1:4])
ax_scatter.spines["top"].set_visible(True)
ax_scatter.spines["right"].set_visible(True)
ax_scatter.spines["top"].set_linewidth(0.75)
ax_scatter.spines["right"].set_linewidth(0.75)

ax_scatter.tick_params(
    top=True, right=True,
    which="both",
    direction="in",
    length=3,
    width=0.75
)

var1 = pca.explained_variance_ratio_[0] * 100
var2 = pca.explained_variance_ratio_[1] * 100

for grp, grp_col in palette.items():
    mask = df["cluster"] == grp
    ax_scatter.scatter(
        df.loc[mask, "PC1"],
        df.loc[mask, "PC2"],
        c=grp_col,
        s=28,
        alpha=0.88,
        edgecolors="white",
        linewidths=0.3,
        label=grp,
        zorder=3
    )

#==================== Cluster centroids====================
centroids = km2.cluster_centers_
for grp, grp_col in palette.items():
    raw_idx = high_cluster if grp == "Cluster 1" else 1 - high_cluster
    ax_scatter.scatter(
        centroids[raw_idx, 0],
        centroids[raw_idx, 1],
        c=grp_col,
        s=90,
        marker="*",
        edgecolors=GREY_DARK,
        linewidths=0.5,
        zorder=5
    )

#==================== Convex hull shading====================
from scipy.spatial import ConvexHull
for grp, grp_col in palette.items():
    mask = df["cluster"] == grp
    pts  = df.loc[mask, ["PC1", "PC2"]].values
    if len(pts) >= 3:
        try:
            hull  = ConvexHull(pts)
            verts = np.append(hull.vertices, hull.vertices[0])
            ax_scatter.fill(
                pts[verts, 0], pts[verts, 1],
                alpha=0.08, color=grp_col, zorder=1
            )
            ax_scatter.plot(
                pts[verts, 0], pts[verts, 1],
                lw=0.6, color=grp_col, alpha=0.4,
                linestyle="--", zorder=2
            )
        except Exception:
            pass

ax_scatter.set_xlabel(f"PC1 ({var1:.1f}%)", fontsize=7, labelpad=3)
ax_scatter.set_ylabel(f"PC2 ({var2:.1f}%)", fontsize=7, labelpad=3)
ax_scatter.set_title("PCA + K-means clustering", fontsize=8,
                      fontweight="bold", pad=4)

#==================== Silhouette annotation====================
ax_scatter.text(
    0.97, 0.03,
    f"Silhouette = {sil_k2:.3f}",
    transform=ax_scatter.transAxes,
    fontsize=5.5, ha="right", va="bottom",
    color=GREY_MID
)

#==================== Legend====================
handles = [
    Line2D([0], [0], marker="o", color="w",
           markerfacecolor=C1_COLOR, markersize=5,
           label=f"Cluster 1 (n={n_c1})"),
    Line2D([0], [0], marker="o", color="w",
           markerfacecolor=C2_COLOR, markersize=5,
           label=f"Cluster 2 (n={n_c2})"),
    Line2D([0], [0], marker="*", color="w",
           markerfacecolor=GREY_DARK, markersize=6,
           label="Centroid"),
]
ax_scatter.legend(
    handles=handles,
    loc="upper right",
    fontsize=5.5,
    frameon=False,
    handletextpad=0.3,
    borderpad=0,
)

# ====================Panel label====================
ax_scatter.text(-0.12, 1.04, "A",
                transform=ax_scatter.transAxes,
                fontsize=10, fontweight="bold")

# ---- PANEL B: Kinetic parameter boxplots ----
for i, feat in enumerate(FEATURE_COLS):
    ax = fig.add_subplot(gs[1, i])

    c1_vals = df.loc[df["cluster"] == "Cluster 1", feat].dropna().values
    c2_vals = df.loc[df["cluster"] == "Cluster 2", feat].dropna().values

    #==================== Boxplot====================
    bp = ax.boxplot(
        [c1_vals, c2_vals],
        positions=[0, 1],
        widths=0.45,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="white", linewidth=1.2),
        whiskerprops=dict(linewidth=0.75, color=GREY_DARK),
        capprops=dict(linewidth=0.75, color=GREY_DARK),
        boxprops=dict(linewidth=0.75),
        zorder=2
    )
    for patch, col in zip(bp["boxes"], [C1_COLOR, C2_COLOR]):
        patch.set_facecolor(col)
        patch.set_alpha(0.75)

    # ====================Jittered strip====================
    np.random.seed(42)
    for j, (vals, col) in enumerate(zip([c1_vals, c2_vals],
                                         [C1_COLOR, C2_COLOR])):
        jitter = np.random.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(j + jitter, vals,
                   color=col, s=8, alpha=0.7,
                   edgecolors="white", linewidths=0.2,
                   zorder=3)

    # ====================Significance bar====================
    mwu  = mwu_results[feat]
    y_max = max(np.max(c1_vals), np.max(c2_vals))
    y_min = min(np.min(c1_vals), np.min(c2_vals))
    y_rng = y_max - y_min
    y_bar = y_max + y_rng * 0.08
    ax.set_ylim(y_min - y_rng * 0.08, y_max + y_rng * 0.22)
    add_significance_bar(ax, 0, 1, y_bar, mwu["p"], h=0.03)

    # p-value text
     #ax.text(0.5, 0.97,
     #        format_p(mwu["p"]),
      #       transform=ax.transAxes,
      #       fontsize=5, ha="center", va="top",
       #      color=GREY_MID)

    #==================== Axes formatting====================
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Cluster 1", "Cluster 2"], fontsize=6)
    ax.set_ylabel(feat_labels[feat], fontsize=6.5, labelpad=3)
    ax.tick_params(axis="y", labelsize=6, length=2.5)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.6, 1.6)

    #==================== Effect size====================
    ax.text(
        0.97, 0.03,
        f"r = {abs(mwu['r']):.2f}",
        transform=ax.transAxes,
        fontsize=5.5, ha="right", va="bottom",
        color=GREY_MID
    )

    if i == 0:
        ax.text(-0.35, 1.04, "B",
                transform=ax.transAxes,
                fontsize=10, fontweight="bold")

# ====================Save====================
fig.savefig(
    os.path.join(WORK_DIR, "Figure_PCA_Kmeans.png"),
    dpi=300, facecolor="white"
)
fig.savefig(
    os.path.join(WORK_DIR, "Figure_PCA_Kmeans.pdf"),
    facecolor="white"
)
plt.close(fig)
print("[INFO] Saved: Figure_PCA_Kmeans.png / .pdf")

# =============================================================
# K-selection + Scree + Loadings
# =============================================================

k_range = range(2, 7)
inertias, sil_sc, ch_sc, db_sc = [], [], [], []
for k in k_range:
    km_    = KMeans(n_clusters=k, random_state=42, n_init=50)
    labs_  = km_.fit_predict(X_pca)
    inertias.append(km_.inertia_)
    sil_sc.append(silhouette_score(X_pca, labs_))
    ch_sc.append(calinski_harabasz_score(X_pca, labs_))
    db_sc.append(davies_bouldin_score(X_pca, labs_))

# PCA loadings
pca_all  = PCA(n_components=len(FEATURE_COLS), random_state=42).fit(X_scaled)
loadings = pd.DataFrame(
    pca_all.components_.T,
    index=FEATURE_COLS,
    columns=[f"PC{i+1}" for i in range(len(FEATURE_COLS))]
)
feat_label_short = {
    "lag":            "Lag",
    "auc":      "AUC",
    "maxslope": "MaxSlope",
    "fmax":     "Fmax",
    "slope":    "Slope",
}

fig_s = plt.figure(figsize=(183/25.4, 140/25.4))
gs_s  = GridSpec(
    2, 4,
    figure=fig_s,
    left=0.08, right=0.97,
    top=0.91,  bottom=0.11,
    wspace=0.55, hspace=0.55,
)

# Scree Plot
ax_scree = fig_s.add_subplot(gs_s[0, 0])
ax_scree.plot(range(1, len(FEATURE_COLS)+1), explained*100,
              "o-", color=GREY_DARK, lw=1, ms=4)
ax_scree.axvline(n_comp_auto, color=C1_COLOR, ls="--", lw=0.8,
                  label=f"PC{n_comp_auto}")
ax_scree.set_xlabel("Principal component", fontsize=7, labelpad=3)
ax_scree.set_ylabel("Explained variance (%)", fontsize=7, labelpad=3)
ax_scree.set_title("Scree plot", fontsize=8, fontweight="bold", pad=4)
ax_scree.set_xticks(range(1, len(FEATURE_COLS)+1))
ax_scree.legend(fontsize=5.5, frameon=False)
ax_scree.text(-0.22, 1.04, "A", transform=ax_scree.transAxes,
               fontsize=10, fontweight="bold")

# Cumulative variance plot
ax_cum = fig_s.add_subplot(gs_s[0, 1])
ax_cum.plot(range(1, len(FEATURE_COLS)+1), cumulative*100,
            "o-", color=GREY_DARK, lw=1, ms=4)
ax_cum.axhline(VARIANCE_THRESHOLD*100, color=GREY_MID,
               ls="--", lw=0.8, label=f"{VARIANCE_THRESHOLD*100:.0f}%")
ax_cum.axvline(n_comp_auto, color=C1_COLOR, ls="--", lw=0.8)
ax_cum.set_xlabel("Number of PCs", fontsize=7, labelpad=3)
ax_cum.set_ylabel("Cumulative variance (%)", fontsize=7, labelpad=3)
ax_cum.set_title("Cumulative variance", fontsize=8, fontweight="bold", pad=4)
ax_cum.set_xticks(range(1, len(FEATURE_COLS)+1))
ax_cum.legend(fontsize=5.5, frameon=False)
ax_cum.text(-0.22, 1.04, "B", transform=ax_cum.transAxes,
             fontsize=10, fontweight="bold")

# Elbow (inertia) plot
ax_elbow = fig_s.add_subplot(gs_s[0, 2])
ax_elbow.plot(list(k_range), inertias,
              "o-", color=GREY_DARK, lw=1, ms=4)
ax_elbow.axvline(2, color=C1_COLOR, ls="--", lw=0.8, label="k = 2")
ax_elbow.set_xlabel("Number of clusters (k)", fontsize=7, labelpad=3)
ax_elbow.set_ylabel("Inertia", fontsize=7, labelpad=3)
ax_elbow.set_title("Elbow method", fontsize=8, fontweight="bold", pad=4)
ax_elbow.set_xticks(list(k_range))
ax_elbow.legend(fontsize=5.5, frameon=False)
ax_elbow.text(-0.22, 1.04, "C", transform=ax_elbow.transAxes,
               fontsize=10, fontweight="bold")

# Silhouette plot
ax_sil = fig_s.add_subplot(gs_s[0, 3])
ax_sil.plot(list(k_range), sil_sc,
            "o-", color=GREY_DARK, lw=1, ms=4)
ax_sil.axvline(2, color=C1_COLOR, ls="--", lw=0.8, label="k = 2")
ax_sil.set_xlabel("Number of clusters (k)", fontsize=7, labelpad=3)
ax_sil.set_ylabel("Silhouette score", fontsize=7, labelpad=3)
ax_sil.set_title("Silhouette score", fontsize=8, fontweight="bold", pad=4)
ax_sil.set_xticks(list(k_range))
ax_sil.legend(fontsize=5.5, frameon=False)
ax_sil.text(-0.22, 1.04, "D", transform=ax_sil.transAxes,
             fontsize=10, fontweight="bold")

# PCA loadings heatmap
ax_load = fig_s.add_subplot(gs_s[1, :2])
load_data   = loadings.values
feat_names  = [feat_label_short[f] for f in FEATURE_COLS]
pc_names    = [f"PC{i+1}\n({explained[i]*100:.0f}%)"
               for i in range(len(FEATURE_COLS))]

im = ax_load.imshow(load_data, cmap="RdBu_r",
                     vmin=-1, vmax=1, aspect="auto")
ax_load.set_xticks(range(len(FEATURE_COLS)))
ax_load.set_xticklabels(pc_names, fontsize=6)
ax_load.set_yticks(range(len(FEATURE_COLS)))
ax_load.set_yticklabels(feat_names, fontsize=6)
ax_load.set_title("PCA feature loadings", fontsize=8, fontweight="bold", pad=4)

for row_i in range(len(FEATURE_COLS)):
    for col_j in range(len(FEATURE_COLS)):
        val = load_data[row_i, col_j]
        txt_col = "white" if abs(val) > 0.6 else GREY_DARK
        ax_load.text(col_j, row_i, f"{val:.2f}",
                     ha="center", va="center",
                     fontsize=5.5, color=txt_col)

cbar = fig_s.colorbar(im, ax=ax_load, fraction=0.025,
                       pad=0.02, shrink=0.8)
cbar.set_label("Loading", fontsize=6)
cbar.ax.tick_params(labelsize=5.5)
ax_load.spines["top"].set_visible(True)
ax_load.spines["right"].set_visible(True)
ax_load.text(-0.12, 1.04, "E", transform=ax_load.transAxes,
              fontsize=10, fontweight="bold")

# Davies-Bouldin score
ax_db = fig_s.add_subplot(gs_s[1, 2])
ax_db.plot(list(k_range), db_sc,
           "o-", color=GREY_DARK, lw=1, ms=4)
ax_db.axvline(2, color=C1_COLOR, ls="--", lw=0.8, label="k = 2")
ax_db.set_xlabel("Number of clusters (k)", fontsize=7, labelpad=3)
ax_db.set_ylabel("Davies–Bouldin score", fontsize=7, labelpad=3)
ax_db.set_title("Davies–Bouldin score\n(lower = better)",
                 fontsize=8, fontweight="bold", pad=4)
ax_db.set_xticks(list(k_range))
ax_db.legend(fontsize=5.5, frameon=False)
ax_db.text(-0.22, 1.04, "F", transform=ax_db.transAxes,
            fontsize=10, fontweight="bold")

# Calinski-Harabasz score
ax_ch = fig_s.add_subplot(gs_s[1, 3])
ax_ch.plot(list(k_range), ch_sc,
           "o-", color=GREY_DARK, lw=1, ms=4)
ax_ch.axvline(2, color=C1_COLOR, ls="--", lw=0.8, label="k = 2")
ax_ch.set_xlabel("Number of clusters (k)", fontsize=7, labelpad=3)
ax_ch.set_ylabel("Calinski–Harabasz score", fontsize=7, labelpad=3)
ax_ch.set_title("Calinski–Harabasz score\n(higher = better)",
                 fontsize=8, fontweight="bold", pad=4)
ax_ch.set_xticks(list(k_range))
ax_ch.legend(fontsize=5.5, frameon=False)
ax_ch.text(-0.22, 1.04, "G", transform=ax_ch.transAxes,
            fontsize=10, fontweight="bold")

fig_s.savefig(
    os.path.join(WORK_DIR, "FigureS_Methods.png"),
    dpi=300, facecolor="white"
)
fig_s.savefig(
    os.path.join(WORK_DIR, "FigureS_Methods.pdf"),
    facecolor="white"
)
plt.close(fig_s)
print("[INFO] Saved: FigureS_Methods.png / .pdf")
print(f"\n[PCA] Variance explained:")
print(f"  PC1: {pca.explained_variance_ratio_[0]*100:.1f}%")
print(f"  PC2: {pca.explained_variance_ratio_[1]*100:.1f}%")
print(f"  Total: {sum(pca.explained_variance_ratio_)*100:.1f}%")
# Print loadings clearly
print("\n[PCA] Feature loadings:")
print(f"{'Feature':20s}  {'PC1':>8}  {'PC2':>8}")
print("-" * 42)
for feat, pc1, pc2 in zip(FEATURE_COLS,
                            pca.components_[0],
                            pca.components_[1]):
    print(f"{feat:20s}  {pc1:>8.3f}  {pc2:>8.3f}")

print("\n[DONE] Both figures saved to:", WORK_DIR)

# =======================================================
# ====== SAVE CLUSTER ASSIGNMENTS========================
# =======================================================
cluster_out = df[["case", "PC1", "PC2", "cluster"]].copy()

for feat in FEATURE_COLS:
    cluster_out[feat] = df[feat]

if "assignment_confidence" in df.columns:
    cluster_out["assignment_confidence"] = df["assignment_confidence"]
    cluster_out["signed_confidence"]     = df["signed_confidence"]

cluster_out = cluster_out.sort_values(["cluster", "case"]).reset_index(drop=True)

#=============Save==========================
cluster_out.to_csv(
    os.path.join(WORK_DIR, "cluster_assignments.csv"),
    index=False
)

with pd.ExcelWriter(
    os.path.join(WORK_DIR, "cluster_assignments.xlsx"),
    engine="openpyxl"
) as writer:
    cluster_out.to_excel(writer, sheet_name="cluster_assignments", index=False)


    summary = pd.DataFrame({
        "cluster":     ["Cluster 1", "Cluster 2"],
        "n":           [n_c1, n_c2],
        "silhouette":  [sil_k2, sil_k2],
        "PC1_var_pct": [pca.explained_variance_ratio_[0]*100,
                        pca.explained_variance_ratio_[0]*100],
        "PC2_var_pct": [pca.explained_variance_ratio_[1]*100,
                        pca.explained_variance_ratio_[1]*100],
    })
    summary.to_excel(writer, sheet_name="summary", index=False)

print(f"[INFO] Saved cluster assignments to: {WORK_DIR}")
print(f"  • cluster_assignments.csv")
print(f"  • cluster_assignments.xlsx")
print(f"\n[INFO] Cluster breakdown:")
print(cluster_out[["case", "cluster"]].to_string(index=False))





















"""
RT-QuIC averaged kinetic trace plotting Colours traces by K-means cluster assignment
"""

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter1d

# ===================================================
# ====== CONFIGURATIONS =====================
# ===================================================
WORK_DIR       = r"/Users/mw4217/Desktop/RTQuic calculations/Seeding/"
CURVES_FILE    = "/Users/mw4217/Desktop/RTQuic calculations/RTQuIC_curvesBH.xlsx"
CLUSTERS_FILE  = os.path.join(WORK_DIR, "cluster_assignments.csv")


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

C1_COLOR  = "#2166AC"
C2_COLOR  = "#F4A700"
GREY_MID  = "#888888"
GREY_DARK = "#333333"

# ======================================
# ====== LOAD DATA =====================
# ======================================
print("[INFO] Loading curves...")
curves_raw = pd.read_excel(CURVES_FILE, engine="openpyxl")

# Fix non-breaking spaces in column names
curves_raw.columns = [
    str(c).replace("\xa0", " ").replace("\u200b", "").strip()
    for c in curves_raw.columns
]

# First column is time
time_col  = curves_raw.columns[0]

curves_raw = curves_raw.dropna(subset=[time_col]).reset_index(drop=True)

time      = curves_raw[time_col].values.astype(float)
case_cols = [c for c in curves_raw.columns if c != time_col]

print(f"  Timepoints : {len(time)}  ({time[0]:.2f} to {time[-1]:.2f} h)")
print(f"  Cases      : {len(case_cols)}")

# Convert all case columns to numeric
for c in case_cols:
    curves_raw[c] = pd.to_numeric(curves_raw[c], errors="coerce")

# ======================================
# === NORMALISE TO %RFU ================
# ======================================
curves = curves_raw.copy()

print("\n[INFO] Normalised value ranges (first 3 cases):")
for c in case_cols[:3]:
    print(f"  {c}: min={curves[c].min():.1f}  max={curves[c].max():.1f}")

# ======================================
# === FIX CLUSTER MERGE ================
# ======================================
print("\n[INFO] Loading cluster assignments...")
clusters = pd.read_csv(CLUSTERS_FILE)
clusters["case"] = (clusters["case"].astype(str)
                    .str.replace("\xa0", " ")
                    .str.strip())

# Also clean case_cols for matching
case_cols_clean = [
    str(c).replace("\xa0", " ").strip()
    for c in case_cols
]

# Rebuild curves with column names
curves.columns = [time_col] + case_cols_clean
case_cols = case_cols_clean

cluster_map = dict(zip(clusters["case"], clusters["cluster"]))
cluster_map = {
    k.replace(" MSA", "").strip(): v
    for k, v in cluster_map.items()
}

case_cols = [c.replace(" MSA", "").strip() for c in case_cols]

# Check matching
missing = [c for c in case_cols if c not in cluster_map]
matched = [c for c in case_cols if c in cluster_map]
if missing:
    print(f"  [WARNING] {len(missing)} cases still unmatched: {missing}")
print(f"  Matched: {len(matched)} / {len(case_cols)} cases")

c1_cases = [c for c in case_cols if cluster_map.get(c) == "Cluster 1"]
c2_cases = [c for c in case_cols if cluster_map.get(c) == "Cluster 2"]
print(f"  Cluster 1: {len(c1_cases)} cases")
print(f"  Cluster 2: {len(c2_cases)} cases")
# ==============================================
# ===== COMPUTE MEANS and SD ===================
# ==============================================
def cluster_stats(cases):
    """Return mean and SD across cases at each timepoint."""
    if not cases:
        return np.full(len(time), np.nan), np.full(len(time), np.nan)
    data = curves[cases].values.astype(float)
    mean = np.nanmean(data, axis=1)
    sd   = np.nanstd(data, axis=1)
    return mean, sd

c1_mean, c1_sd = cluster_stats(c1_cases)
c2_mean, c2_sd = cluster_stats(c2_cases)

# Optional: light gaussian smoothing for display
# Set sigma=0 to disable
SMOOTH_SIGMA = 0.8
c1_mean_s = gaussian_filter1d(c1_mean, sigma=SMOOTH_SIGMA)
c2_mean_s = gaussian_filter1d(c2_mean, sigma=SMOOTH_SIGMA)

# ======================================
# ======== FIGURE =======================
# Left:  individual traces coloured by cluster
# Right: mean ± SD per cluster
# ======================================
fig, axes = plt.subplots(1, 2, figsize=(130/25.4, 65/25.4))

palette = {"Cluster 1": C1_COLOR, "Cluster 2": C2_COLOR}

# ---- PANEL A: Individual traces ----=============
ax = axes[0]

for case in c2_cases:
    ax.plot(time, curves[case].values,
            color=C2_COLOR, lw=0.5, alpha=0.4, zorder=2)
for case in c1_cases:
    ax.plot(time, curves[case].values,
            color=C1_COLOR, lw=0.5, alpha=0.4, zorder=3)

# Overlay cluster means on top
ax.plot(time, c1_mean_s, color=C1_COLOR, lw=1.5, zorder=5)
ax.plot(time, c2_mean_s, color=C2_COLOR, lw=1.5, zorder=4)

ax.set_xlabel("Time (hours)", fontsize=7, labelpad=3)
ax.set_ylabel("Fluorescence (RFU)", fontsize=7, labelpad=3)
ax.set_title("Individual traces", fontsize=8, fontweight="bold", pad=4)
ax.set_xlim(0, 70)
ax.set_ylim(bottom=0)
ax.tick_params(labelsize=6, length=3)
for spine in ax.spines.values():
    spine.set_linewidth(0.75)

handles = [
    Line2D([0], [0], color=C1_COLOR, lw=1.5,
           label=f"Cluster 1 (n={len(c1_cases)})"),
    Line2D([0], [0], color=C2_COLOR, lw=1.5,
           label=f"Cluster 2 (n={len(c2_cases)})"),
]
ax.legend(handles=handles, fontsize=6, frameon=False,
          loc="upper left", handlelength=1.2)
ax.text(-0.18, 1.04, "A", transform=ax.transAxes,
        fontsize=10, fontweight="bold")

# ---- PANEL B: Mean ± sd ----=============
ax2 = axes[1]

#sd shading
ax2.fill_between(time,
                 c1_mean_s - c1_sd,
                 c1_mean_s + c1_sd,
                 color=C1_COLOR, alpha=0.15, zorder=2)
ax2.fill_between(time,
                 c2_mean_s - c2_sd,
                 c2_mean_s + c2_sd,
                 color=C2_COLOR, alpha=0.15, zorder=2)

# Mean lines
ax2.plot(time, c1_mean_s, color=C1_COLOR, lw=1.5, zorder=4,
         label=f"Cluster 1 (n={len(c1_cases)})")
ax2.plot(time, c2_mean_s, color=C2_COLOR, lw=1.5, zorder=3,
         label=f"Cluster 2 (n={len(c2_cases)})")

ax2.set_xlabel("Time (hours)", fontsize=7, labelpad=3)
ax2.set_ylabel("Fluorescence (RFU)", fontsize=7, labelpad=3)
ax2.set_title("Mean \u00b1 SD", fontsize=8, fontweight="bold", pad=4)
ax2.set_xlim(0, 70)
ax2.set_ylim(bottom=0)
ax2.tick_params(labelsize=6, length=3)
for spine in ax2.spines.values():
    spine.set_linewidth(0.75)
ax2.legend(fontsize=6, frameon=False,
           loc="upper left", handlelength=1.2)
ax2.text(-0.18, 1.04, "A", transform=ax2.transAxes,
         fontsize=10, fontweight="bold")

fig.tight_layout()
fig.savefig(os.path.join(WORK_DIR, "Figure_kinetic_traces.png"),
            dpi=300, facecolor="white")
fig.savefig(os.path.join(WORK_DIR, "Figure_kinetic_traces.pdf"),
            facecolor="white")
plt.close(fig)
print("[INFO] Saved: Figure_kinetic_traces.png / .pdf")

# =========================
# === ALSO SAVE MEAN TRACES
# =========================
mean_traces = pd.DataFrame({
    "time_hours":      time,
    "cluster1_mean":   c1_mean,
    "cluster1_sd":    c1_sd,
    "cluster2_mean":   c2_mean,
    "cluster2_sd":    c2_sd,
})
mean_traces.to_excel(
    os.path.join(WORK_DIR, "mean_traces_by_cluster.xlsx"),
    index=False
)
print("[INFO] Saved: mean_traces_by_cluster.xlsx")