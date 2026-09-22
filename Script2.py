# ============================================================
# SINGLE FOCUS CARTOGRAPHIC DISTORTION
# F1 = 240°–270°
# x-stretch + height distortion + cartographic styling
# ============================================================

from pathlib import Path
import os
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

from shapely.ops import transform as shapely_transform
from shapely.geometry import LineString
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")

# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = Path(
    r"C:\IGN_French_Visa\Arcachon_Project\Polar_Log_Python\Full_Reshape_Polar_Pipeline"
)

OUT_ROOT = Path(
    r"C:\IGN_French_Visa\Arcachon_Project\Polar_Log_Python\Single_Focus_Final_Cartography"
)

OUT_ROOT.mkdir(parents=True, exist_ok=True)

# Automatically use latest full_reshape_pipeline_* folder
runs = [p for p in BASE_DIR.glob("full_reshape_pipeline_*") if p.is_dir()]

if not runs:
    raise FileNotFoundError("No full_reshape_pipeline_* folder found.")

RUN_DIR = sorted(runs, key=os.path.getmtime)[-1]
LAYER_DIR = RUN_DIR / "02_transformed_layers"

if not LAYER_DIR.exists():
    raise FileNotFoundError(f"Cannot find transformed layers folder: {LAYER_DIR}")

OUT_DIR = OUT_ROOT / f"single_focus_F1_240_270_{RUN_DIR.name}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("Using transformed run:")
print(RUN_DIR)

print("\nUsing layer folder:")
print(LAYER_DIR)

print("\nSaving output to:")
print(OUT_DIR)


# ============================================================
# 2. SINGLE FOCUS SETTINGS
# ============================================================

FOCUS_RANGES = [
    ("F1", 240, 270)
]

# Horizontal scale
BASE_X_SCALE = 0.60
FOCUS_X_SCALE = 2.50

# Height distortion
CONTEXT_HEIGHT_SCALE = 0.65
FOCUS_HEIGHT_SCALE = 1.30

# Smooth transition before and after focus
TRANSITION_DEG = 15

STRIP_HEIGHT = 1000


# ============================================================
# 3. FIND AND LOAD TRANSFORMED LAYERS
# ============================================================

BAD_WORDS = [
    "fallback",
    "failed",
    "centroid",
    "representative",
    "rep_point",
    "reppoint"
]

VECTOR_EXTS = [".geojson", ".shp", ".gpkg"]


def all_vector_files(folder):
    return [p for p in folder.rglob("*") if p.suffix.lower() in VECTOR_EXTS]


FILES = all_vector_files(LAYER_DIR)

print("\nAvailable files:")
for f in FILES:
    print(" -", f.name)


def read_gdf(path):
    try:
        gdf = gpd.read_file(path)
        gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
        gdf.crs = None
        return gdf
    except Exception as e:
        print(f"Could not read {path.name}: {e}")
        return None


def geometry_family(gdf):
    if gdf is None or len(gdf) == 0:
        return "empty"

    types = set(gdf.geometry.geom_type.unique())

    if any(t in types for t in ["Polygon", "MultiPolygon"]):
        return "polygon"

    if any(t in types for t in ["LineString", "MultiLineString"]):
        return "line"

    if any(t in types for t in ["Point", "MultiPoint"]):
        return "point"

    return "mixed"


def choose_layer(label, keyword_options, expected_family=None, allow_bad=False):
    candidates = []

    for keys in keyword_options:
        keys = [k.lower() for k in keys]

        for f in FILES:
            name = f.stem.lower()

            if all(k in name for k in keys):
                if not allow_bad and any(bad in name for bad in BAD_WORDS):
                    continue

                candidates.append(f)

    if not candidates:
        print(f"{label}: not found")
        return None

    # Prefer GeoJSON first
    candidates = sorted(
        candidates,
        key=lambda p: (
            0 if p.suffix.lower() == ".geojson" else 1,
            len(p.name)
        )
    )

    for path in candidates:
        gdf = read_gdf(path)

        if gdf is None or len(gdf) == 0:
            continue

        fam = geometry_family(gdf)

        if expected_family is None or fam == expected_family:
            print(f"{label}: {path.name} | {fam} | {len(gdf)} features")
            return gdf

    print(f"{label}: candidates found, but no matching geometry family")
    return None


layers = {
    "flood": choose_layer(
        "flood",
        [["flood", "polygons"], ["flood", "polygon"], ["flood"]],
        expected_family="polygon"
    ),

    "aoi": choose_layer(
        "aoi",
        [["27", "aoi"], ["aoi", "polygon"], ["aio"], ["aoi"]],
        expected_family="polygon"
    ),

    "coastline": choose_layer(
        "coastline",
        [["coastline"], ["coast"]],
        expected_family="line",
        allow_bad=True
    ),

    "buildings": choose_layer(
        "buildings",
        [["building", "polygons"], ["building", "polygon"], ["building"]],
        expected_family="polygon"
    ),

    "roads": choose_layer(
        "roads",
        [["roads"], ["road"]],
        expected_family="line",
        allow_bad=True
    ),

    "railways": choose_layer(
        "railways",
        [["railways"], ["rail"]],
        expected_family="line",
        allow_bad=True
    ),

    "waterbodies": choose_layer(
        "waterbodies",
        [["waterbody"], ["waterbodies"]],
        expected_family="polygon"
    ),

    "traffic": choose_layer(
        "traffic",
        [["traffic"]],
        expected_family="point",
        allow_bad=True
    ),

    "transport": choose_layer(
        "transport facilities",
        [["transport", "facilities"], ["transport"]],
        expected_family="point",
        allow_bad=True
    )
}

layers = {k: v for k, v in layers.items() if v is not None}

print("\nLoaded layers:")
print(list(layers.keys()))


# ============================================================
# 4. DETECT FLOOD OCCURRENCE FIELD
# ============================================================

flood_field = None

if "flood" in layers:
    candidates = [
        "nb_scn",
        "NB_SCN",
        "Flooded_Scenarios",
        "Flooded Scenarios",
        "flooded_scenarios",
        "flooded_sc",
        "count",
        "COUNT",
        "gridcode",
        "GRIDCODE",
        "value",
        "VALUE"
    ]

    for col in candidates:
        if col in layers["flood"].columns:
            flood_field = col
            break

print("\nFlood field used:", flood_field)


# ============================================================
# 5. SCALE FUNCTIONS
# ============================================================

def smoothstep(t):
    return t * t * (3 - 2 * t)


def build_x_scale_mapping(focus_ranges, base_scale, focus_scale, transition_deg):
    theta_grid = np.linspace(0, 360, 3601)
    scale_grid = np.ones_like(theta_grid) * base_scale

    for label, start, end in focus_ranges:

        focus_mask = (theta_grid >= start) & (theta_grid <= end)
        scale_grid[focus_mask] = np.maximum(scale_grid[focus_mask], focus_scale)

        # Left transition
        left_start = max(0, start - transition_deg)
        left_end = start
        left_mask = (theta_grid >= left_start) & (theta_grid < left_end)

        if left_mask.any() and left_end > left_start:
            t = (theta_grid[left_mask] - left_start) / (left_end - left_start)
            scale_grid[left_mask] = base_scale + (focus_scale - base_scale) * smoothstep(t)

        # Right transition
        right_start = end
        right_end = min(360, end + transition_deg)
        right_mask = (theta_grid > right_start) & (theta_grid <= right_end)

        if right_mask.any() and right_end > right_start:
            t = 1 - ((theta_grid[right_mask] - right_start) / (right_end - right_start))
            scale_grid[right_mask] = base_scale + (focus_scale - base_scale) * smoothstep(t)

    xnew_grid = np.zeros_like(theta_grid)
    dtheta = np.diff(theta_grid)

    xnew_grid[1:] = np.cumsum(
        0.5 * (scale_grid[:-1] + scale_grid[1:]) * dtheta
    )

    def theta_to_xnew(theta):
        return np.interp(theta, theta_grid, xnew_grid)

    return theta_grid, scale_grid, xnew_grid, theta_to_xnew


def build_height_scale_function(focus_ranges, context_height, focus_height, transition_deg):
    theta_grid = np.linspace(0, 360, 3601)
    height_grid = np.ones_like(theta_grid) * context_height

    for label, start, end in focus_ranges:

        focus_mask = (theta_grid >= start) & (theta_grid <= end)
        height_grid[focus_mask] = np.maximum(height_grid[focus_mask], focus_height)

        # Left transition
        left_start = max(0, start - transition_deg)
        left_end = start
        left_mask = (theta_grid >= left_start) & (theta_grid < left_end)

        if left_mask.any() and left_end > left_start:
            t = (theta_grid[left_mask] - left_start) / (left_end - left_start)
            height_grid[left_mask] = context_height + (focus_height - context_height) * smoothstep(t)

        # Right transition
        right_start = end
        right_end = min(360, end + transition_deg)
        right_mask = (theta_grid > right_start) & (theta_grid <= right_end)

        if right_mask.any() and right_end > right_start:
            t = 1 - ((theta_grid[right_mask] - right_start) / (right_end - right_start))
            height_grid[right_mask] = context_height + (focus_height - context_height) * smoothstep(t)

    def theta_to_height(theta):
        return np.interp(theta, theta_grid, height_grid)

    return theta_grid, height_grid, theta_to_height


theta_grid, x_scale_grid, xnew_grid, theta_to_xnew = build_x_scale_mapping(
    FOCUS_RANGES,
    BASE_X_SCALE,
    FOCUS_X_SCALE,
    TRANSITION_DEG
)

theta_h_grid, height_grid, theta_to_height = build_height_scale_function(
    FOCUS_RANGES,
    CONTEXT_HEIGHT_SCALE,
    FOCUS_HEIGHT_SCALE,
    TRANSITION_DEG
)


# ============================================================
# 6. APPLY X + HEIGHT DISTORTION TO GEOMETRIES
# ============================================================

def fix_geometry(geom):
    if geom is None or geom.is_empty:
        return None

    try:
        if geom.is_valid:
            return geom

        fixed = geom.buffer(0)

        if fixed is not None and not fixed.is_empty:
            return fixed

        return geom

    except Exception:
        return geom


def valid_bounds(geom):
    if geom is None or geom.is_empty:
        return False

    try:
        b = geom.bounds
        return all(np.isfinite(v) for v in b)
    except Exception:
        return False


def transform_geometry_mixed_height(geom):
    if geom is None or geom.is_empty:
        return None

    def func(x, y, z=None):
        x_new = theta_to_xnew(x)
        h = theta_to_height(x)
        y_new = y * h

        if z is None:
            return x_new, y_new

        return x_new, y_new, z

    try:
        out = shapely_transform(func, geom)
        out = fix_geometry(out)

        if not valid_bounds(out):
            return None

        return out

    except Exception:
        return None


mixed_layers = {}

for name, gdf in layers.items():

    print("Transforming to mixed x/y space:", name)

    out = gdf.copy()
    out.crs = None

    out["geometry"] = out.geometry.apply(transform_geometry_mixed_height)

    out = out[out.geometry.notna() & ~out.geometry.is_empty].copy()
    out = out[out.geometry.apply(valid_bounds)].copy()
    out.crs = None

    if len(out) > 0:
        mixed_layers[name] = out

        out_path = OUT_DIR / f"{name}_single_focus_F1_240_270.geojson"

        try:
            out.to_file(out_path, driver="GeoJSON")
            print("Saved:", out_path.name)
        except Exception as e:
            print("Could not save:", name, e)

print("\nMixed layers created:")
print(list(mixed_layers.keys()))


# ============================================================
# 7. HELPER: SELECT FEATURES INSIDE FOCUS AREA BY BOUNDS
# ============================================================

def select_by_x_bounds(gdf, xmin, xmax):
    if gdf is None or len(gdf) == 0:
        return None

    bounds = gdf.geometry.bounds

    selected = gdf[
        (bounds["maxx"] >= xmin) &
        (bounds["minx"] <= xmax)
    ].copy()

    if len(selected) == 0:
        return None

    selected.crs = None
    return selected


focus_label, focus_start, focus_end = FOCUS_RANGES[0]
focus_xmin = theta_to_xnew(focus_start)
focus_xmax = theta_to_xnew(focus_end)


# ============================================================
# 8. FINAL CARTOGRAPHIC PLOT
# ============================================================

STYLE = {
    "focus_fill": "#FFF2B2",
    "focus_alpha": 0.35,

    "transition_fill": "#E6E6E6",
    "transition_alpha": 0.30,

    "waterbody_face": "#D8EEF7",
    "waterbody_edge": "#8ECAE6",

    "flood_cmap": "Blues",
    "flood_alpha": 0.78,

    "building_face": "#7A6F65",
    "building_alpha": 0.65,

    "roads": "#F4A261",
    "roads_lw": 0.45,
    "roads_alpha": 0.55,

    "railways": "#252525",
    "railways_lw": 0.80,
    "railways_alpha": 0.85,

    "coastline": "#00CFE3",
    "coastline_lw": 1.60,

    "aoi": "#E34A33",
    "aoi_lw": 1.25,

    "traffic": "#7B3294",
    "transport": "#008837",

    "min_boundary": "#08306B",
    "max_boundary": "#CB181D",
}

fig, ax = plt.subplots(figsize=(24, 6.8))

# ------------------------------------------------------------
# Background transition and focus zones
# ------------------------------------------------------------
for label, start, end in FOCUS_RANGES:
    left_start = max(0, start - TRANSITION_DEG)
    left_end = start

    right_start = end
    right_end = min(360, end + TRANSITION_DEG)

    ax.axvspan(
        theta_to_xnew(left_start),
        theta_to_xnew(left_end),
        color=STYLE["transition_fill"],
        alpha=STYLE["transition_alpha"],
        linewidth=0,
        zorder=0
    )

    ax.axvspan(
        theta_to_xnew(right_start),
        theta_to_xnew(right_end),
        color=STYLE["transition_fill"],
        alpha=STYLE["transition_alpha"],
        linewidth=0,
        zorder=0
    )

    ax.axvspan(
        theta_to_xnew(start),
        theta_to_xnew(end),
        color=STYLE["focus_fill"],
        alpha=STYLE["focus_alpha"],
        linewidth=0,
        zorder=0
    )

    ax.axvline(theta_to_xnew(start), color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.axvline(theta_to_xnew(end), color="black", linestyle="--", linewidth=1.0, alpha=0.7)

    ax.text(
        (theta_to_xnew(start) + theta_to_xnew(end)) / 2,
        FOCUS_HEIGHT_SCALE * STRIP_HEIGHT + 35,
        f"{label}: {int(start)}°–{int(end)}°",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.88)
    )

# ------------------------------------------------------------
# Waterbodies first
# ------------------------------------------------------------
if "waterbodies" in mixed_layers:
    mixed_layers["waterbodies"].plot(
        ax=ax,
        color=STYLE["waterbody_face"],
        edgecolor=STYLE["waterbody_edge"],
        linewidth=0.25,
        alpha=0.50,
        zorder=1
    )

# ------------------------------------------------------------
# Flood occurrence polygons
# ------------------------------------------------------------
flood_sm = None

if "flood" in mixed_layers:
    flood_gdf = mixed_layers["flood"].copy()

    if flood_field is not None and flood_field in flood_gdf.columns:
        flood_gdf[flood_field] = pd.to_numeric(flood_gdf[flood_field], errors="coerce")
        vals = flood_gdf[flood_field].dropna()

        if len(vals) > 0:
            norm = colors.Normalize(vmin=float(vals.min()), vmax=float(vals.max()))
            cmap = cm.get_cmap(STYLE["flood_cmap"])
            flood_sm = cm.ScalarMappable(norm=norm, cmap=cmap)

            flood_gdf.plot(
                ax=ax,
                column=flood_field,
                cmap=STYLE["flood_cmap"],
                linewidth=0.08,
                edgecolor="#08519C",
                alpha=STYLE["flood_alpha"],
                zorder=2
            )
        else:
            flood_gdf.plot(
                ax=ax,
                color="#6BAED6",
                linewidth=0,
                alpha=STYLE["flood_alpha"],
                zorder=2
            )
    else:
        flood_gdf.plot(
            ax=ax,
            color="#6BAED6",
            linewidth=0,
            alpha=STYLE["flood_alpha"],
            zorder=2
        )

# ------------------------------------------------------------
# Context roads and railways
# ------------------------------------------------------------
if "roads" in mixed_layers:
    mixed_layers["roads"].plot(
        ax=ax,
        color=STYLE["roads"],
        linewidth=STYLE["roads_lw"],
        alpha=STYLE["roads_alpha"],
        zorder=3
    )

if "railways" in mixed_layers:
    mixed_layers["railways"].plot(
        ax=ax,
        color=STYLE["railways"],
        linewidth=STYLE["railways_lw"],
        alpha=STYLE["railways_alpha"],
        zorder=4
    )

# ------------------------------------------------------------
# Buildings only inside focused area
# ------------------------------------------------------------
if "buildings" in mixed_layers:
    b_focus = select_by_x_bounds(
        mixed_layers["buildings"],
        focus_xmin,
        focus_xmax
    )

    if b_focus is not None:
        b_focus.plot(
            ax=ax,
            color=STYLE["building_face"],
            linewidth=0,
            alpha=STYLE["building_alpha"],
            zorder=5
        )

# ------------------------------------------------------------
# AOI and coastline on top
# ------------------------------------------------------------
if "aoi" in mixed_layers:
    mixed_layers["aoi"].boundary.plot(
        ax=ax,
        color=STYLE["aoi"],
        linewidth=STYLE["aoi_lw"],
        alpha=0.95,
        zorder=6
    )

if "coastline" in mixed_layers:
    mixed_layers["coastline"].plot(
        ax=ax,
        color=STYLE["coastline"],
        linewidth=STYLE["coastline_lw"],
        alpha=1.0,
        zorder=7
    )

# ------------------------------------------------------------
# Traffic and transport points only inside focused area
# ------------------------------------------------------------
if "traffic" in mixed_layers:
    traffic_focus = select_by_x_bounds(
        mixed_layers["traffic"],
        focus_xmin,
        focus_xmax
    )

    if traffic_focus is not None:
        traffic_focus.plot(
            ax=ax,
            color=STYLE["traffic"],
            marker="o",
            markersize=20,
            alpha=0.90,
            zorder=8
        )

if "transport" in mixed_layers:
    transport_focus = select_by_x_bounds(
        mixed_layers["transport"],
        focus_xmin,
        focus_xmax
    )

    if transport_focus is not None:
        transport_focus.plot(
            ax=ax,
            color=STYLE["transport"],
            marker="^",
            markersize=36,
            alpha=0.95,
            zorder=9
        )

# ------------------------------------------------------------
# Height-distorted boundaries
# ------------------------------------------------------------
ax.plot(
    xnew_grid,
    np.zeros_like(xnew_grid),
    color=STYLE["min_boundary"],
    linewidth=1.8,
    zorder=10
)

ax.plot(
    xnew_grid,
    STRIP_HEIGHT * height_grid,
    color=STYLE["max_boundary"],
    linewidth=1.8,
    zorder=10
)

# ------------------------------------------------------------
# Axes and ticks
# ------------------------------------------------------------
theta_ticks = [0, 30, 60, 90, 120, 150, 180, 210, 225, 240, 270, 285, 300, 330, 360]
x_ticks = [theta_to_xnew(t) for t in theta_ticks]

ax.set_xticks(x_ticks)
ax.set_xticklabels([f"{t}°" for t in theta_ticks], fontsize=9)

ax.set_xlim(theta_to_xnew(0), theta_to_xnew(360))
ax.set_ylim(0, FOCUS_HEIGHT_SCALE * STRIP_HEIGHT + 90)

ax.set_aspect("auto")

ax.set_xlabel("Original angular position θ shown on a modified x-scale", fontsize=10)
ax.set_ylabel("Height-distorted normalized radial position", fontsize=10)

ax.set_title(
    "Single-focus deformation strip: x-scale and height distortion for F1",
    fontsize=13,
    fontweight="bold"
)

ax.grid(True, linestyle="--", alpha=0.18)

ax.text(
    0.01,
    0.03,
    "Note: x-scale and vertical height are locally distorted for cartographic emphasis.",
    transform=ax.transAxes,
    fontsize=8,
    bbox=dict(facecolor="white", edgecolor="none", alpha=0.85)
)

# ------------------------------------------------------------
# Colorbar
# ------------------------------------------------------------
if flood_sm is not None:
    cbar = plt.colorbar(
        flood_sm,
        ax=ax,
        orientation="vertical",
        fraction=0.025,
        pad=0.015
    )
    cbar.set_label("Flooded scenarios", fontsize=9)

# ------------------------------------------------------------
# Legend
# ------------------------------------------------------------
legend_handles = [
    Patch(facecolor=STYLE["focus_fill"], edgecolor="none", alpha=STYLE["focus_alpha"], label="Focused area"),
    Patch(facecolor=STYLE["transition_fill"], edgecolor="none", alpha=STYLE["transition_alpha"], label="Transition area"),
    Patch(facecolor=STYLE["waterbody_face"], edgecolor=STYLE["waterbody_edge"], alpha=0.50, label="Waterbodies"),
    Patch(facecolor="#6BAED6", edgecolor="#08519C", alpha=STYLE["flood_alpha"], label="Flood occurrence"),
    Patch(facecolor=STYLE["building_face"], edgecolor="none", alpha=STYLE["building_alpha"], label="Buildings in F1"),
    Line2D([0], [0], color=STYLE["roads"], lw=1.4, label="Roads"),
    Line2D([0], [0], color=STYLE["railways"], lw=1.4, label="Railways"),
    Line2D([0], [0], color=STYLE["aoi"], lw=1.5, label="AOI boundary"),
    Line2D([0], [0], color=STYLE["coastline"], lw=1.8, label="Coastline"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=STYLE["traffic"], markersize=8, label="Traffic points in F1"),
    Line2D([0], [0], marker="^", color="w", markerfacecolor=STYLE["transport"], markersize=9, label="Transport facilities in F1"),
    Line2D([0], [0], color=STYLE["min_boundary"], lw=1.8, label="Minimum boundary"),
    Line2D([0], [0], color=STYLE["max_boundary"], lw=1.8, label="Height-distorted maximum boundary"),
]

ax.legend(
    handles=legend_handles,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.38),
    ncol=6,
    fontsize=8,
    frameon=True
)

fig.subplots_adjust(bottom=0.30)

# --------------------------------------------
# 9. SAVE OUTPUTS
# -------------------------------------------------

out_png = OUT_DIR / "single_focus_F1_240_270_x_height_cartography.png"
out_pdf = OUT_DIR / "single_focus_F1_240_270_x_height_cartography.pdf"
out_svg = OUT_DIR / "single_focus_F1_240_270_x_height_cartography.svg"

fig.savefig(out_png, dpi=300, bbox_inches="tight")
fig.savefig(out_pdf, bbox_inches="tight")
fig.savefig(out_svg, bbox_inches="tight")

plt.show()

print("Saved final outputs:")
print(out_png)
print(out_pdf)
print(out_svg)