"""
Bangladesh DoE Station Map — White Theme, Dramatic Stations
============================================================
Reads BD.json (local file, same directory as script).
White/print-friendly background with dramatic station effects.

Usage:
    pip install geopandas matplotlib
    python Fig1_Station_Map_geopandas.py

Output:
    Fig1_Station_Map.png  (300 dpi, publication-ready)
"""

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

# ── PATH ─────────────────────────────────────────────────────
GEOJSON_PATH = "BD.json"

# ── 4 SELECTED STATIONS ──────────────────────────────────────
SELECTED = [
    {
        "name": "Agrabad", "location": "Chittagong",
        "lat": 22.32315, "lon": 91.80221,
        "color": "#D62728", "glow": "#FF9896",
        "marker": "o", "dx": 0.44, "dy": 0.10, "ha": "left",
    },
    {
        "name": "Darus Salam", "location": "Dhaka",
        "lat": 23.78086, "lon": 90.35565,
        "color": "#1565C0", "glow": "#90CAF9",
        "marker": "s", "dx": -0.44, "dy": 0.12, "ha": "right",
    },
    {
        "name": "Red Crescent Office", "location": "Sylhet",
        "lat": 24.88885, "lon": 91.86733,
        "color": "#2E7D32", "glow": "#A5D6A7",
        "marker": "^", "dx": 0.44, "dy": 0.10, "ha": "left",
    },
    {
        "name": "Uttar Bagura Road", "location": "Barishal",
        "lat": 22.70982, "lon": 90.36250,
        "color": "#6A1B9A", "glow": "#CE93D8",
        "marker": "D", "dx": -0.44, "dy": -0.32, "ha": "right",
    },
]

# ── EXCLUDED STATIONS ─────────────────────────────────────────
EXCLUDED = [
    {"name": "US Embassy",            "lat": 23.79696, "lon": 90.42190},
    {"name": "Farmgate",              "lat": 23.75973, "lon": 90.38941},
    {"name": "Gazipur",               "lat": 23.99419, "lon": 90.42199},
    {"name": "Khanpur",               "lat": 23.62649, "lon": 90.50769},
    {"name": "Khulshi",               "lat": 22.36164, "lon": 91.79988},
    {"name": "Baira",                 "lat": 22.84236, "lon": 89.53981},
    {"name": "Sopura",                "lat": 24.38043, "lon": 88.60509},
    {"name": "DOE Office Mymensingh", "lat": 24.76260, "lon": 90.40184},
    {"name": "Rangpur BTV",           "lat": 25.74207, "lon": 89.21918},
    {"name": "AERI East Chandana",    "lat": 23.95501, "lon": 90.27962},
    {"name": "Narsingdhi Sadar",      "lat": 23.86220, "lon": 90.68120},
    {"name": "Court Area Comilla",    "lat": 23.47228, "lon": 91.18074},
    {"name": "Savar",                 "lat": 23.95501, "lon": 90.27962},
]

# ── DIVISION PALETTE — light pastel, print-safe ───────────────
DIV_PALETTE = {
    "dhaka"      : "#FFF9C4",
    "chittagong" : "#E0F7FA",
    "rajshahi"   : "#E8F5E9",
    "khulna"     : "#FCE4EC",
    "barishal"   : "#F3E5F5",
    "sylhet"     : "#E3F2FD",
    "rangpur"    : "#FFF3E0",
    "mymensingh" : "#E8EAF6",
}
FALLBACK_COLOR = "#F5F5F5"

# ── LOAD ─────────────────────────────────────────────────────
print(f"Loading: {GEOJSON_PATH}")
gdf = gpd.read_file(GEOJSON_PATH)
print(f"  Rows    : {len(gdf)}")
print(f"  CRS     : {gdf.crs}")
print(f"  Columns : {list(gdf.columns)}")

if gdf.crs is None:
    gdf = gdf.set_crs("EPSG:4326")
elif gdf.crs.to_epsg() != 4326:
    gdf = gdf.to_crs("EPSG:4326")

NAME_COL = None
for c in ["NAME_1", "ADM1_EN", "DIV_NAME", "DIVISION", "division",
          "name", "Name", "bibhag", "NAME"]:
    if c in gdf.columns:
        NAME_COL = c
        break

if NAME_COL:
    print(f"  Name col  : {NAME_COL}")
    print(f"  Divisions : {sorted(gdf[NAME_COL].unique())}")
else:
    print(f"  WARNING: no name column detected.")
    print(f"  Available columns: {list(gdf.columns)}")

# ── FIGURE — white background ─────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 10))
fig.patch.set_facecolor("white")
ax.set_facecolor("#DDEEFF")   # soft blue ocean / Bay of Bengal

# ── 1. Division polygons — light pastel fills ─────────────────
if NAME_COL:
    for _, row in gdf.iterrows():
        raw  = str(row[NAME_COL]).lower().strip()
        fill = FALLBACK_COLOR
        for key, col in DIV_PALETTE.items():
            if key in raw:
                fill = col
                break
        gpd.GeoDataFrame([row], geometry="geometry", crs=gdf.crs).plot(
            ax=ax, color=fill, edgecolor="#546E7A",
            linewidth=0.75, zorder=2,
        )
else:
    gdf.plot(ax=ax, color="#E8F5E9",
             edgecolor="#546E7A", linewidth=0.75, zorder=2)

# ── 2. Division name labels — dark italic, subtle ─────────────
if NAME_COL:
    for _, row in gdf.iterrows():
        cx    = row.geometry.centroid.x
        cy    = row.geometry.centroid.y
        label = str(row[NAME_COL]).replace(" Division", "").strip()
        ax.text(cx, cy, label,
                fontsize=7, color="#37474F",
                ha="center", va="center",
                style="italic", alpha=0.70,
                fontweight="normal", zorder=3)

# ── 3. Excluded stations — X markers, muted grey-red ─────────
for st in EXCLUDED:
    # soft halo ring
    ax.scatter(st["lon"], st["lat"],
               s=160, color="none",
               edgecolors="#EF9A9A", linewidths=0.7,
               alpha=0.5, zorder=5, marker="o")
    # X marker body
    ax.scatter(st["lon"], st["lat"],
               s=75, color="#FFEBEE", marker="X",
               edgecolors="#C62828", linewidths=0.9,
               alpha=0.90, zorder=6)

# ── 4. Selected stations — dramatic concentric rings ──────────
for st in SELECTED:
    lon, lat = st["lon"], st["lat"]
    col      = st["color"]
    glow     = st["glow"]

    # Layer 1 — outermost soft wash
    ax.scatter(lon, lat, s=2200, color=glow,
               alpha=0.10, zorder=7, edgecolors="none")
    # Layer 2 — medium halo
    ax.scatter(lon, lat, s=1000, color=glow,
               alpha=0.18, zorder=7, edgecolors="none")
    # Layer 3 — tight colour fill
    ax.scatter(lon, lat, s=450, color=col,
               alpha=0.22, zorder=7, edgecolors="none")
    # Layer 4 — crisp outer ring
    ax.scatter(lon, lat, s=280, color="none",
               edgecolors=col, linewidths=1.8,
               alpha=0.95, zorder=8)
    # Layer 5 — solid bright core
    ax.scatter(lon, lat, s=140, color=col, marker=st["marker"],
               edgecolors="white", linewidths=1.6,
               alpha=1.0, zorder=9)

    # Annotation — coloured banner, dark text for readability
    ax.annotate(
        f"  {st['name']}  \n  {st['location']}  ",
        xy=(lon, lat),
        xytext=(lon + st["dx"], lat + st["dy"]),
        fontsize=8.5, ha=st["ha"], va="center",
        color="white", fontweight="bold",
        arrowprops=dict(
            arrowstyle="-|>",
            color=col, lw=1.4,
            mutation_scale=12,
            shrinkA=10, shrinkB=5,
        ),
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor=col, alpha=0.90,
            edgecolor="white", linewidth=0.8,
        ),
        zorder=10,
    )

# ── 5. Legend ─────────────────────────────────────────────────
selected_handles = [
    Line2D([0], [0],
           marker=st["marker"], color="w",
           markerfacecolor=st["color"],
           markeredgecolor="white", markersize=11,
           label=f"\u2713  {st['name']} ({st['location']})")
    for st in SELECTED
]
excluded_handle = Line2D(
    [0], [0], marker="X", color="w",
    markerfacecolor="#FFEBEE",
    markeredgecolor="#C62828",
    markersize=9,
    label=f"\u2717  Excluded DoE stations  (n={len(EXCLUDED)})",
)

leg = ax.legend(
    handles=selected_handles + [excluded_handle],
    loc="lower left", fontsize=8.5,
    framealpha=0.95,
    facecolor="white",
    edgecolor="#546E7A",
    title="DoE Monitoring Stations",
    title_fontsize=9,
)
leg.get_title().set_color("#1A237E")
leg.get_title().set_fontweight("bold")

# ── 6. Count badges ───────────────────────────────────────────
ax.text(
    92.72, 26.60,
    f"Selected\nn = {len(SELECTED)}",
    fontsize=8, color="#1B5E20",
    ha="right", va="top", fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#E8F5E9",
              edgecolor="#2E7D32", linewidth=1.0, alpha=0.92),
    zorder=10,
)
ax.text(
    92.72, 25.88,
    f"Excluded\nn = {len(EXCLUDED)}",
    fontsize=8, color="#B71C1C",
    ha="right", va="top", fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFEBEE",
              edgecolor="#C62828", linewidth=1.0, alpha=0.92),
    zorder=10,
)

# ── 7. Axes ───────────────────────────────────────────────────
ax.set_xlim(87.9, 92.8)
ax.set_ylim(20.4, 26.8)
ax.set_aspect("equal")
ax.set_xlabel("Longitude (\u00b0E)", fontsize=11, color="#212121")
ax.set_ylabel("Latitude (\u00b0N)",  fontsize=11, color="#212121")
ax.tick_params(labelsize=9, colors="#212121")
for spine in ax.spines.values():
    spine.set_edgecolor("#546E7A")
    spine.set_linewidth(0.8)

ax.set_title(
    "Figure 1: Location of DoE Air Quality Monitoring Stations\n"
    "Selected for Analysis in Bangladesh (2014\u20132021)",
    fontsize=11, fontweight="bold", pad=14, color="#1A237E",
)
ax.grid(True, linestyle="--", linewidth=0.3,
        alpha=0.4, color="#90A4AE", zorder=1)

plt.tight_layout()
OUT = "Fig1_Station_Map.png"
plt.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {OUT}")