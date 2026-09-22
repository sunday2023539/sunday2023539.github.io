# ----------------------------------------------------------
# THREE SEPARATE FULL-CONTEXT FOCUS MAPS 
#
# ------------------------------------------------------------

from pathlib import Path
import os
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
from matplotlib.font_manager import FontProperties
from matplotlib.colors import LinearSegmentedColormap

from shapely.ops import transform as shapely_transform, unary_union
from shapely import affinity

from matplotlib.lines import Line2D
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")

# -------------------------------------------
# AUTOMATIC THREE-MAP DRIVER
# ----------------------------------------

ALL_FOCUS_CONFIGS = [
    ("F1", 110, 145),
    ("F2", 240, 270),
    ("F3", 315, 335),
]

def generate_full_context_focus_map(ACTIVE_FOCUS_ID):
    # -----------------------------------------------------------------------------------
    # 1. PATHS
    # ------------------------------------------------------------------------------------
    
    BASE_DIR = Path(
        r"C:\IGN_French_Visa\Arcachon_Project\Polar_Log_Python\Full_Reshape_Polar_Pipeline"
    )
    
    OUT_ROOT = Path(
        r"C:\IGN_French_Visa\Arcachon_Project\Polar_Log_Python\Three_Separate"
    )
    
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    
    runs = [p for p in BASE_DIR.glob("full_reshape_pipeline_*") if p.is_dir()]
    
    if not runs:
        raise FileNotFoundError("No full_reshape_pipeline_* folder found.")
    
    RUN_DIR = sorted(runs, key=os.path.getmtime)[-1]
    LAYER_DIR = RUN_DIR / "02_transformed_layers"
    
    if not LAYER_DIR.exists():
        raise FileNotFoundError(f"Cannot find transformed layers folder: {LAYER_DIR}")
    
    print("Using transformed run:")
    print(RUN_DIR)
    
    print("\nUsing layer folder:")
    print(LAYER_DIR)
    
    # ============================================================
    # 2. FOCUS SETTINGS
    # ============================================================
    
    focus_matches = [item for item in ALL_FOCUS_CONFIGS if item[0] == ACTIVE_FOCUS_ID]
    
    if len(focus_matches) != 1:
        raise ValueError(f"Unknown focus identifier: {ACTIVE_FOCUS_ID}")
    
    FOCUS_RANGES = focus_matches
    focus_label, focus_start, focus_end = FOCUS_RANGES[0]
    FOCUS_NAME = f"{focus_label}_{focus_start}_{focus_end}"
    
    OUT_DIR = OUT_ROOT / f"three_separate_full_context_maps_{RUN_DIR.name}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\nSaving output to:")
    print(OUT_DIR)
    
    # ============================================================
    # 3. GENERAL SETTINGS
    # ============================================================
    
    BASE_X_SCALE = 0.60
    FOCUS_X_SCALE = 2.50
    
    CONTEXT_HEIGHT_SCALE = 0.65
    FOCUS_HEIGHT_SCALE = 1.30
    
    TRANSITION_DEG = 15
    STRIP_HEIGHT = 1000
    
    POP_FIELD = "ind"
    COMMUNE_FIELD = "nom_com"
    POP_SLICE_DEG = 5
    
    MAX_BUILDINGS_TRANSFORM = 60000
    MAX_BUILDINGS_DRAW = 22000
    
    # Urbanized area settings
    MAX_BUILDINGS_FOR_URBAN = 50000
    URBAN_BUFFER_X = 1.15
    URBAN_BUFFER_Y = 24.0
    URBAN_SIMPLIFY = 1.10
    
    # Sea settings
    SEA_COAST_QUANTILE = 0.20
    SEA_SMOOTH_WINDOW = 19
    
    # Landmark settings
    LANDMARK_SPACE = 245
    LANDMARK_Y = -145
    
    try:
        EMOJI_FONT = FontProperties(family="Segoe UI Emoji")
    except:
        EMOJI_FONT = None
    
    # ============================================================
    # 4. STYLE
    # ============================================================
    
    HEAT_CMAP = LinearSegmentedColormap.from_list(
        "custom_heat",
        [
            "#FFFDE7",  # very low flood occurrence
            "#FFF176",  # yellow
            "#FFB74D",  # orange
            "#FF7043",  # deep orange
            "#E53935"   # red = high flood occurrence
        ]
    )
    
    STYLE = {
        # Sea
        "sea_face": "#B9DFF2",
        "sea_alpha": 0.65,
    
        # Focus background
        "focus_fill": "#FFF2B2",
        "focus_edge": "#666666",
        "focus_alpha": 0.16,
    
        "transition_fill": "#E6E6E6",
        "transition_alpha": 0.12,
    
        # Waterbodies
        "waterbody_face": "#377EB8",
        "waterbody_edge": "#174A73",
        "waterbody_alpha": 0.62,
    
        # Urbanized area
        "urban_face": "#B7A99A",
        "urban_edge": "#6F6258",
        "urban_alpha": 0.18,
        "urban_boundary_alpha": 0.15,
    
        # Flood occurrence - HEAT PALETTE
        "flood_cmap": HEAT_CMAP,
        "flood_alpha": 0.78,
        "flood_edge": "#B22222",
    
        # Buildings
        "building_face": "#2F2924",
        "building_edge": "#FFFFFF",
        "building_alpha": 0.96,
    
        # Roads
        "road_context": "#B88945",
        "road_context_lw": 0.42,
        "road_context_alpha": 0.55,
    
        "road_focus": "#F07C2F",
        "road_focus_lw": 0.86,
        "road_focus_alpha": 0.90,
    
        # Railways
        "railways": "#1F1F1F",
        "railways_lw": 0.95,
        "railways_alpha": 0.88,
    
        # Coastline
        "coastline": "#7A1F5C",
        "coastline_lw": 2.20,
    
        # Landmark
        "landmark_line": "#4A4A4A",
    
        # Boundaries
        "min_boundary": "#08306B",
        "max_boundary": "#CB181D",
    
        # Population
        "pop_context": "#C7C7C7",
        "pop_focus": "#4D4D4D"
    }
    
    # All focus areas use the same restrained cartographic treatment.
    # They are identified by their labels and boundaries, not extra colours.
    FOCUS_COLORS = {label: STYLE["focus_edge"] for label, _, _ in FOCUS_RANGES}
    
    # Approximate geographic orientation around the unwrapped basin shoreline.
    # These positions were calibrated from the supplied original basin map.
    ORIENTATION_MARKS = [
        (20, "W"),
        (55, "NW"),
        (145, "N"),
        (195, "NE"),
        (235, "E"),
        (270, "SE"),
        (300, "S"),
        (335, "SW"),
    ]
    
    # ============================================================
    # 5. FIND AND LOAD TRANSFORMED LAYERS
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
    
    print("\nAvailable transformed files:")
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
    
        "coastline": choose_layer(
            "coastline",
            [["coastline"], ["coast"]],
            expected_family="line",
            allow_bad=True
        ),
    
        "population": choose_layer(
            "population grid",
            [["population", "grid"], ["population"]],
            expected_family="polygon"
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
        )
    }
    
    layers = {k: v for k, v in layers.items() if v is not None}
    
    print("\nLoaded layers:")
    print(list(layers.keys()))
    
    # ============================================================
    # 6. DETECT IMPORTANT FIELDS
    # ============================================================
    
    flood_field = None
    
    if "flood" in layers:
        flood_candidates = [
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
    
        for col in flood_candidates:
            if col in layers["flood"].columns:
                flood_field = col
                break
    
    print("\nFlood field used:", flood_field)
    
    if "population" in layers:
        print("\nPopulation columns:")
        print(list(layers["population"].columns))
    
        if POP_FIELD not in layers["population"].columns:
            print(f"WARNING: population field '{POP_FIELD}' not found.")
    
        if COMMUNE_FIELD not in layers["population"].columns:
            print(f"WARNING: commune field '{COMMUNE_FIELD}' not found.")
    
    # ============================================================
    # 7. SCALE FUNCTIONS
    # ============================================================
    
    def smoothstep(t):
        return t * t * (3 - 2 * t)
    
    
    def build_x_scale_mapping(focus_ranges, base_scale, focus_scale, transition_deg):
        theta_grid = np.linspace(0, 360, 3601)
        scale_grid = np.ones_like(theta_grid) * base_scale
    
        for label, start, end in focus_ranges:
    
            focus_mask = (theta_grid >= start) & (theta_grid <= end)
            scale_grid[focus_mask] = np.maximum(scale_grid[focus_mask], focus_scale)
    
            left_start = max(0, start - transition_deg)
            left_end = start
            left_mask = (theta_grid >= left_start) & (theta_grid < left_end)
    
            if left_mask.any() and left_end > left_start:
                t = (theta_grid[left_mask] - left_start) / (left_end - left_start)
                scale_grid[left_mask] = base_scale + (focus_scale - base_scale) * smoothstep(t)
    
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
    
            left_start = max(0, start - transition_deg)
            left_end = start
            left_mask = (theta_grid >= left_start) & (theta_grid < left_end)
    
            if left_mask.any() and left_end > left_start:
                t = (theta_grid[left_mask] - left_start) / (left_end - left_start)
                height_grid[left_mask] = context_height + (focus_height - context_height) * smoothstep(t)
    
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
    
    FOCUS_X_RANGES = [
        (label, theta_to_xnew(start), theta_to_xnew(end))
        for label, start, end in FOCUS_RANGES
    ]
    
    # ============================================================
    # 8. GEOMETRY TRANSFORMATION
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
    
        if name == "buildings" and len(out) > MAX_BUILDINGS_TRANSFORM:
            print(f"Sampling buildings before transform: {MAX_BUILDINGS_TRANSFORM} / {len(out)}")
            out = out.sample(MAX_BUILDINGS_TRANSFORM, random_state=42).copy()
    
        out["geometry"] = out.geometry.apply(transform_geometry_mixed_height)
    
        out = out[out.geometry.notna() & ~out.geometry.is_empty].copy()
        out = out[out.geometry.apply(valid_bounds)].copy()
        out.crs = None
    
        if len(out) > 0:
            mixed_layers[name] = out
    
    print("\nMixed layers created:")
    print(list(mixed_layers.keys()))
    
    # ============================================================
    # 9. SELECTION HELPERS
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
    
    
    def select_by_focus_ranges(gdf):
        """Select features intersecting at least one of the three focus ranges."""
        if gdf is None or len(gdf) == 0:
            return None
    
        bounds = gdf.geometry.bounds
        mask = np.zeros(len(gdf), dtype=bool)
    
        for _, xmin, xmax in FOCUS_X_RANGES:
            mask |= ((bounds["maxx"] >= xmin) & (bounds["minx"] <= xmax)).to_numpy()
    
        selected = gdf.loc[mask].copy()
        if len(selected) == 0:
            return None
    
        selected.crs = None
        return selected
    
    
    def detect_road_class_field(gdf):
        possible = [
            "highway",
            "fclass",
            "class",
            "road_type",
            "type",
            "TYPE",
            "nature",
            "NATURE",
            "category",
            "CATEGORY"
        ]
    
        for col in possible:
            if col in gdf.columns:
                return col
    
        return None
    
    
    def filter_context_roads(gdf):
        if gdf is None or len(gdf) == 0:
            return None
    
        road_field = detect_road_class_field(gdf)
    
        if road_field is None:
            print("No road class field found. Using all roads as visible context.")
            return gdf.copy()
    
        major_keywords = [
            "motorway",
            "trunk",
            "primary",
            "secondary",
            "tertiary",
            "autoroute",
            "principale",
            "national",
            "departemental",
            "départemental",
            "route"
        ]
    
        temp = gdf.copy()
        road_text = temp[road_field].astype(str).str.lower()
    
        mask = road_text.apply(
            lambda value: any(k in value for k in major_keywords)
        )
    
        roads_context = temp[mask].copy()
    
        if len(roads_context) == 0:
            print("No filtered roads found. Using all roads as context.")
            return temp
    
        print(f"Road class field used: {road_field}")
        print(f"Context roads selected: {len(roads_context)} / {len(temp)}")
    
        roads_context.crs = None
        return roads_context
    
    
    roads_context = None
    roads_focus = None
    buildings_focus = None
    
    if "roads" in mixed_layers:
        roads_context = filter_context_roads(mixed_layers["roads"])
        roads_focus = select_by_focus_ranges(mixed_layers["roads"])
    
    if "buildings" in mixed_layers:
        buildings_focus = select_by_focus_ranges(mixed_layers["buildings"])
    
    # ============================================================
    # 10. SEA BACKGROUND BELOW COASTLINE
    # ============================================================
    
    def extract_line_coords(geom):
        coords = []
    
        if geom is None or geom.is_empty:
            return coords
    
        gtype = geom.geom_type
    
        if gtype == "LineString":
            coords.extend(list(geom.coords))
    
        elif gtype == "MultiLineString":
            for part in geom.geoms:
                coords.extend(list(part.coords))
    
        elif gtype == "GeometryCollection":
            for part in geom.geoms:
                coords.extend(extract_line_coords(part))
    
        return coords
    
    
    def make_sea_line_from_coastline(coastline_gdf, xmin, xmax, ymax, n_bins=1000):
        if coastline_gdf is None or len(coastline_gdf) == 0:
            return None, None
    
        xs = []
        ys = []
    
        for geom in coastline_gdf.geometry:
            for x, y, *rest in extract_line_coords(geom):
                if np.isfinite(x) and np.isfinite(y):
                    if xmin <= x <= xmax and -200 <= y <= ymax:
                        xs.append(x)
                        ys.append(y)
    
        if len(xs) < 20:
            print("Not enough coastline coordinates to create sea background.")
            return None, None
    
        xs = np.array(xs)
        ys = np.array(ys)
    
        bins = np.linspace(xmin, xmax, n_bins + 1)
        centers = (bins[:-1] + bins[1:]) / 2
        sea_y = np.full(len(centers), np.nan)
    
        bin_ids = np.digitize(xs, bins) - 1
    
        for i in range(len(centers)):
            values = ys[bin_ids == i]
    
            if len(values) > 0:
                sea_y[i] = np.nanquantile(values, SEA_COAST_QUANTILE)
    
        s = pd.Series(sea_y)
        s = s.interpolate(limit_direction="both")
    
        if SEA_SMOOTH_WINDOW > 1:
            s = s.rolling(
                window=SEA_SMOOTH_WINDOW,
                center=True,
                min_periods=1
            ).median()
    
        sea_y = s.to_numpy()
        sea_y = np.clip(sea_y, 0, ymax)
    
        return centers, sea_y
    
    # ============================================================
    # 11. URBANIZED AREA FROM BUILDINGS
    # ============================================================
    
    def anisotropic_buffer_geom(geom, x_radius=1.15, y_radius=24.0):
        if geom is None or geom.is_empty:
            return None
    
        try:
            factor = x_radius / y_radius
    
            g1 = affinity.scale(
                geom,
                xfact=1.0,
                yfact=factor,
                origin=(0, 0)
            )
    
            g2 = g1.buffer(
                x_radius,
                resolution=5,
                cap_style=1,
                join_style=1
            )
    
            g3 = affinity.scale(
                g2,
                xfact=1.0,
                yfact=1.0 / factor,
                origin=(0, 0)
            )
    
            g3 = fix_geometry(g3)
    
            return g3
    
        except Exception:
            return None
    
    
    def create_urbanized_area(building_gdf):
        if building_gdf is None or len(building_gdf) == 0:
            return None
    
        temp = building_gdf.copy()
        temp.crs = None
    
        if len(temp) > MAX_BUILDINGS_FOR_URBAN:
            print(f"Sampling buildings for urbanized area: {MAX_BUILDINGS_FOR_URBAN} / {len(temp)}")
            temp = temp.sample(MAX_BUILDINGS_FOR_URBAN, random_state=42).copy()
    
        print("Creating urbanized area from building footprints...")
    
        buffers = []
    
        for geom in temp.geometry:
            b = anisotropic_buffer_geom(
                geom,
                x_radius=URBAN_BUFFER_X,
                y_radius=URBAN_BUFFER_Y
            )
    
            if b is not None and not b.is_empty:
                buffers.append(b)
    
        if len(buffers) == 0:
            return None
    
        merged = unary_union(buffers)
        merged = fix_geometry(merged)
    
        if URBAN_SIMPLIFY > 0:
            merged = merged.simplify(
                URBAN_SIMPLIFY,
                preserve_topology=True
            )
    
        urban = gpd.GeoDataFrame(
            {"layer": ["urbanized_area"]},
            geometry=[merged],
            crs=None
        )
    
        urban = urban[urban.geometry.notna() & ~urban.geometry.is_empty].copy()
        urban.crs = None
    
        return urban
    
    
    urbanized_area = None
    
    if "buildings" in mixed_layers:
        urbanized_area = create_urbanized_area(mixed_layers["buildings"])
    
    if urbanized_area is not None:
        urban_path = OUT_DIR / f"urbanized_area_{FOCUS_NAME}.geojson"
        urbanized_area.to_file(urban_path, driver="GeoJSON")
        print("Urbanized area saved:")
        print(urban_path)
    
    # ============================================================
    # 12. POPULATION BAR CHART DATA
    # ============================================================
    
    def make_population_slices(pop_gdf, slice_deg=5):
        if pop_gdf is None or len(pop_gdf) == 0:
            return pd.DataFrame()
    
        if POP_FIELD not in pop_gdf.columns:
            print(f"Population field '{POP_FIELD}' not found.")
            return pd.DataFrame()
    
        temp = pop_gdf.copy()
        temp.crs = None
    
        temp["_theta"] = temp.geometry.representative_point().x
        temp["_pop"] = pd.to_numeric(temp[POP_FIELD], errors="coerce").fillna(0)
    
        bins = np.arange(0, 360 + slice_deg, slice_deg)
        rows = []
    
        for start, end in zip(bins[:-1], bins[1:]):
    
            inside = temp[
                (temp["_theta"] >= start) &
                (temp["_theta"] < end)
            ]
    
            pop_sum = inside["_pop"].sum()
    
            x0 = theta_to_xnew(start)
            x1 = theta_to_xnew(end)
    
            rows.append({
                "theta_start": start,
                "theta_end": end,
                "theta_mid": (start + end) / 2,
                "population": pop_sum,
                "x0": x0,
                "x1": x1,
                "xmid": (x0 + x1) / 2,
                "xwidth": x1 - x0,
                "in_focus": any(
                    start >= f_start and end <= f_end
                    for _, f_start, f_end in FOCUS_RANGES
                )
            })
    
        return pd.DataFrame(rows)
    
    
    pop_slices = pd.DataFrame()
    
    if "population" in layers:
        pop_slices = make_population_slices(
            layers["population"],
            slice_deg=POP_SLICE_DEG
        )
    
    pop_csv = OUT_DIR / f"population_slices_{FOCUS_NAME}.csv"
    pop_slices.to_csv(pop_csv, index=False)
    
    print("\nPopulation slices saved:")
    print(pop_csv)
    
    # ============================================================
    # 13. COMMUNE LABELS FROM nom_com
    # ============================================================
    
    def make_commune_labels(pop_gdf):
        if pop_gdf is None or len(pop_gdf) == 0:
            return pd.DataFrame()
    
        if COMMUNE_FIELD not in pop_gdf.columns:
            print(f"Commune field '{COMMUNE_FIELD}' not found.")
            return pd.DataFrame()
    
        temp = pop_gdf.copy()
        temp.crs = None
    
        rep = temp.geometry.representative_point()
        temp["_theta"] = rep.x
        temp["_y"] = rep.y
    
        if POP_FIELD in temp.columns:
            temp["_pop"] = pd.to_numeric(temp[POP_FIELD], errors="coerce").fillna(0)
        else:
            temp["_pop"] = 1
    
        rows = []
    
        for name, group in temp.groupby(COMMUNE_FIELD):
    
            group = group[
                (group["_theta"] >= 0) &
                (group["_theta"] <= 360)
            ].copy()
    
            if len(group) == 0:
                continue
    
            weights = group["_pop"].to_numpy(dtype=float)
    
            if np.nansum(weights) <= 0:
                weights = np.ones(len(group))
    
            theta_mean = np.average(group["_theta"], weights=weights)
            y_mean = np.average(group["_y"], weights=weights)
    
            x_label = theta_to_xnew(theta_mean)
            y_label = y_mean * theta_to_height(theta_mean)
    
            rows.append({
                "nom_com": str(name),
                "theta": theta_mean,
                "x": x_label,
                "y": y_label,
                "population": group["_pop"].sum(),
                "in_or_near_focus": any(
                    (theta_mean >= f_start - 15) and (theta_mean <= f_end + 15)
                    for _, f_start, f_end in FOCUS_RANGES
                )
            })
    
        labels = pd.DataFrame(rows)
    
        if len(labels) == 0:
            return labels
    
        labels = labels.sort_values("population", ascending=False).reset_index(drop=True)
    
        near_focus = labels[labels["in_or_near_focus"] == True].copy()
        top_others = labels[labels["in_or_near_focus"] == False].head(5).copy()
    
        final_labels = pd.concat([near_focus, top_others], ignore_index=True)
        final_labels = final_labels.drop_duplicates(subset=["nom_com"]).head(10)
    
        return final_labels
    
    
    commune_labels = pd.DataFrame()
    
    if "population" in layers:
        commune_labels = make_commune_labels(layers["population"])
    
    labels_csv = OUT_DIR / f"commune_labels_{FOCUS_NAME}.csv"
    commune_labels.to_csv(labels_csv, index=False)
    
    print("\nCommune labels:")
    if len(commune_labels) > 0:
        print(commune_labels[["nom_com", "theta", "population"]])
    else:
        print("No labels")
    
    print("\nCommune labels saved:")
    print(labels_csv)
    
    # ============================================================
    # 14. MANUAL LANDMARKS FOR ALL THREE FOCUS AREAS
    # ============================================================
    
    LANDMARKS_BY_FOCUS = {
        "F1": [
            {"emoji": "⚓", "name": "Jetée d'Arès", "theta": 116.0, "y_base": 650, "color": "#2A9DF4"},
            {"emoji": "🛸", "name": "Ovniport d'Arès", "theta": 126.0, "y_base": 770, "color": "#8E6CBB"},
            {"emoji": "🌿", "name": "Réserve des Prés Salés", "theta": 138.0, "y_base": 880, "color": "#2CA25F"},
        ],
        "F2": [
            {"emoji": "⚓", "name": "Port d'Audenge", "theta": 245.0, "y_base": 660, "color": "#2A9DF4"},
            {"emoji": "🛶", "name": "Port de Biganos", "theta": 258.0, "y_base": 720, "color": "#C97B2B"},
            {"emoji": "🐦", "name": "Réserve ornithologique du Teich", "theta": 268.0, "y_base": 760, "color": "#2CA25F"},
        ],
        "F3": [
            {"emoji": "🚉", "name": "Arcachon Station", "theta": 329.0, "y_base": 760, "color": "#F4A261"},
            {"emoji": "⚓", "name": "Port / Marina", "theta": 323.0, "y_base": 350, "color": "#2A9DF4"},
            {"emoji": "🌳", "name": "Parc Mauresque", "theta": 331.5, "y_base": 845, "color": "#2CA25F"},
        ],
    }
    
    landmark_rows = []
    
    focus_lookup = {label: (start, end) for label, start, end in FOCUS_RANGES}
    
    for focus_id, focus_landmarks in LANDMARKS_BY_FOCUS.items():
        if focus_id not in focus_lookup:
            continue
    
        focus_start, focus_end = focus_lookup[focus_id]
        focus_xmin = theta_to_xnew(focus_start)
        focus_xmax = theta_to_xnew(focus_end)
        label_x_positions = np.linspace(
            focus_xmin + 0.15 * (focus_xmax - focus_xmin),
            focus_xmax - 0.15 * (focus_xmax - focus_xmin),
            len(focus_landmarks)
        )
    
        for i, lm in enumerate(focus_landmarks, start=1):
            theta = lm["theta"]
            y_base = lm["y_base"]
            landmark_rows.append({
                "focus": focus_id,
                "number": i,
                "emoji": lm["emoji"],
                "name": lm["name"],
                "theta": theta,
                "y_base": y_base,
                "x": theta_to_xnew(theta),
                "y": y_base * theta_to_height(theta),
                "label_x": label_x_positions[i - 1],
                "label_y": LANDMARK_Y,
                "color": lm["color"]
            })
    
    landmarks = pd.DataFrame(landmark_rows)
    
    landmark_csv = OUT_DIR / f"manual_emoji_landmarks_{FOCUS_NAME}.csv"
    landmarks.to_csv(landmark_csv, index=False)
    
    print("\nManual emoji landmarks saved:")
    print(landmark_csv)
    print(landmarks)
    
    # ============================================================
    # 15. FINAL CARTOGRAPHIC PLOT
    # ============================================================
    
    # Wide panoramic cartographic canvas.  Its aspect ratio matches the requested
    # stretched strip and is scaled to fit an A3 landscape sheet when printing.
    fig = plt.figure(figsize=(26, 9.2))
    
    gs = gridspec.GridSpec(
        4,
        1,
        height_ratios=[1.00, 5.10, 0.34, 0.26],
        hspace=0.13
    )
    
    ax_pop = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[1, 0], sharex=ax_pop)
    ax_spacer = fig.add_subplot(gs[2, 0])
    cax = fig.add_subplot(gs[3, 0])
    
    ax_spacer.set_axis_off()
    
    # Compass labels are placed inside this dedicated blank row below the map.
    # This is NOT an x-axis: it has no line, ticks or degree labels.
    ax_spacer.set_xlim(theta_to_xnew(0), theta_to_xnew(360))
    ax_spacer.set_ylim(0, 1)
    
    for theta_value, direction_label in ORIENTATION_MARKS:
        ax_spacer.text(
            theta_to_xnew(theta_value),
            0.62,
            direction_label,
            transform=ax_spacer.get_xaxis_transform(),
            ha="center",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color="#222222",
            clip_on=True,
            zorder=50,
        )
    
    # ------------------------------------------------------------
    # Background focus and transition zones
    # ------------------------------------------------------------
    
    for label, start, end in FOCUS_RANGES:
        left_start = max(0, start - TRANSITION_DEG)
        left_end = start
    
        right_start = end
        right_end = min(360, end + TRANSITION_DEG)
    
        for target_ax in [ax_pop, ax]:
            target_ax.axvspan(
                theta_to_xnew(left_start),
                theta_to_xnew(left_end),
                color=STYLE["transition_fill"],
                alpha=STYLE["transition_alpha"],
                linewidth=0,
                zorder=0
            )
    
            target_ax.axvspan(
                theta_to_xnew(right_start),
                theta_to_xnew(right_end),
                color=STYLE["transition_fill"],
                alpha=STYLE["transition_alpha"],
                linewidth=0,
                zorder=0
            )
    
            target_ax.axvspan(
                theta_to_xnew(start),
                theta_to_xnew(end),
                facecolor=STYLE["focus_fill"],
                alpha=STYLE["focus_alpha"],
                linewidth=0.8,
                edgecolor=STYLE["focus_edge"],
                zorder=0
            )
    
    # Landmark area below strip
    ax.axhspan(
        -LANDMARK_SPACE,
        0,
        color="#FAFAFA",
        alpha=0.95,
        zorder=0
    )
    
    # ------------------------------------------------------------
    # Population bar chart
    # ------------------------------------------------------------
    
    if len(pop_slices) > 0:
    
        context_bars = pop_slices[pop_slices["in_focus"] == False]
        focus_bars = pop_slices[pop_slices["in_focus"] == True]
    
        ax_pop.bar(
            context_bars["xmid"],
            context_bars["population"],
            width=context_bars["xwidth"] * 0.88,
            color=STYLE["pop_context"],
            edgecolor="none",
            alpha=0.85,
            align="center"
        )
    
        ax_pop.bar(
            focus_bars["xmid"],
            focus_bars["population"],
            width=focus_bars["xwidth"] * 0.88,
            color=STYLE["pop_focus"],
            edgecolor="none",
            alpha=0.95,
            align="center"
        )
    
        pop_max = pop_slices["population"].max()
    
        if pop_max > 0:
            ax_pop.set_ylim(0, pop_max * 1.18)
    
        ax_pop.set_ylabel("Population", fontsize=9)
        ax_pop.set_title("")
        ax_pop.grid(True, axis="y", linestyle="--", alpha=0.20)
        ax_pop.tick_params(axis="x", labelbottom=False)
        ax_pop.tick_params(axis="y", labelsize=8)
    
    else:
        ax_pop.text(
            0.5,
            0.5,
            "Population layer unavailable",
            ha="center",
            va="center",
            transform=ax_pop.transAxes
        )
        ax_pop.set_axis_off()
    
    # ------------------------------------------------------------
    # Sea background below coastline
    # ------------------------------------------------------------
    
    sea_x = None
    sea_y = None
    
    if "coastline" in mixed_layers:
        sea_x, sea_y = make_sea_line_from_coastline(
            mixed_layers["coastline"],
            xmin=theta_to_xnew(0),
            xmax=theta_to_xnew(360),
            ymax=FOCUS_HEIGHT_SCALE * STRIP_HEIGHT + 90,
            n_bins=1100
        )
    
    if sea_x is not None and sea_y is not None:
        ax.fill_between(
            sea_x,
            0,
            sea_y,
            color=STYLE["sea_face"],
            alpha=STYLE["sea_alpha"],
            linewidth=0,
            zorder=0.6
        )
    
    # ------------------------------------------------------------
    # Waterbodies
    # ------------------------------------------------------------
    
    if "waterbodies" in mixed_layers:
        mixed_layers["waterbodies"].plot(
            ax=ax,
            color=STYLE["waterbody_face"],
            edgecolor=STYLE["waterbody_edge"],
            linewidth=0.18,
            alpha=STYLE["waterbody_alpha"],
            zorder=1
        )
    
    # ------------------------------------------------------------
    # Urbanized area
    # ------------------------------------------------------------
    
    if urbanized_area is not None and len(urbanized_area) > 0:
        urbanized_area.plot(
            ax=ax,
            facecolor=STYLE["urban_face"],
            edgecolor="none",
            linewidth=0,
            alpha=STYLE["urban_alpha"],
            zorder=1.3
        )
    
        urbanized_area.boundary.plot(
            ax=ax,
            color=STYLE["urban_edge"],
            linewidth=0.18,
            alpha=STYLE["urban_boundary_alpha"],
            zorder=1.4
        )
    
    # ------------------------------------------------------------
    # Flood occurrence using HEAT palette
    # ------------------------------------------------------------
    
    flood_sm = None
    
    if "flood" in mixed_layers:
        flood_gdf = mixed_layers["flood"].copy()
    
        if flood_field is not None and flood_field in flood_gdf.columns:
            flood_gdf[flood_field] = pd.to_numeric(flood_gdf[flood_field], errors="coerce")
            vals = flood_gdf[flood_field].dropna()
    
            if len(vals) > 0:
                norm = colors.Normalize(vmin=float(vals.min()), vmax=float(vals.max()))
                cmap = STYLE["flood_cmap"]
                flood_sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    
                flood_gdf.plot(
                    ax=ax,
                    column=flood_field,
                    cmap=cmap,
                    linewidth=0.025,
                    edgecolor=STYLE["flood_edge"],
                    alpha=STYLE["flood_alpha"],
                    zorder=2
                )
            else:
                flood_gdf.plot(
                    ax=ax,
                    color="#FFB74D",
                    linewidth=0,
                    alpha=STYLE["flood_alpha"],
                    zorder=2
                )
        else:
            flood_gdf.plot(
                ax=ax,
                color="#FFB74D",
                linewidth=0,
                alpha=STYLE["flood_alpha"],
                zorder=2
            )
    
    # ------------------------------------------------------------
    # Roads visible everywhere
    # ------------------------------------------------------------
    
    if roads_context is not None and len(roads_context) > 0:
        roads_context.plot(
            ax=ax,
            color=STYLE["road_context"],
            linewidth=STYLE["road_context_lw"],
            alpha=STYLE["road_context_alpha"],
            zorder=3
        )
    
    # Detailed roads inside the three focus areas
    if roads_focus is not None and len(roads_focus) > 0:
        roads_focus.plot(
            ax=ax,
            color=STYLE["road_focus"],
            linewidth=STYLE["road_focus_lw"],
            alpha=STYLE["road_focus_alpha"],
            zorder=4
        )
    
    # ------------------------------------------------------------
    # Railways
    # ------------------------------------------------------------
    
    if "railways" in mixed_layers:
        mixed_layers["railways"].plot(
            ax=ax,
            color=STYLE["railways"],
            linewidth=STYLE["railways_lw"],
            alpha=STYLE["railways_alpha"],
            zorder=5
        )
    
    # ------------------------------------------------------------
    # Buildings inside the three focus areas only
    # ------------------------------------------------------------
    
    if buildings_focus is not None and len(buildings_focus) > 0:
    
        if len(buildings_focus) > MAX_BUILDINGS_DRAW:
            buildings_focus_plot = buildings_focus.sample(MAX_BUILDINGS_DRAW, random_state=42).copy()
        else:
            buildings_focus_plot = buildings_focus
    
        buildings_focus_plot.plot(
            ax=ax,
            facecolor=STYLE["building_face"],
            edgecolor=STYLE["building_edge"],
            linewidth=0.045,
            alpha=STYLE["building_alpha"],
            zorder=6
        )
    
    # ------------------------------------------------------------
    # Coastline
    # ------------------------------------------------------------
    
    if "coastline" in mixed_layers:
        mixed_layers["coastline"].plot(
            ax=ax,
            color=STYLE["coastline"],
            linewidth=STYLE["coastline_lw"],
            alpha=1.0,
            zorder=8
        )
    
    # ------------------------------------------------------------
    # Minimum and maximum boundaries
    # ------------------------------------------------------------
    
    ax.plot(
        xnew_grid,
        np.zeros_like(xnew_grid),
        color=STYLE["min_boundary"],
        linewidth=1.8,
        zorder=11
    )
    
    ax.plot(
        xnew_grid,
        STRIP_HEIGHT * height_grid,
        color=STYLE["max_boundary"],
        linewidth=1.7,
        alpha=0.90,
        zorder=11
    )
    
    # Focus limit lines and labels
    for focus_label, focus_start, focus_end in FOCUS_RANGES:
        for limit in (focus_start, focus_end):
            ax.axvline(
                theta_to_xnew(limit),
                color=FOCUS_COLORS[focus_label],
                linestyle="-",
                linewidth=1.35,
                alpha=0.95,
                zorder=12
            )
    
        ax.text(
            (theta_to_xnew(focus_start) + theta_to_xnew(focus_end)) / 2,
            FOCUS_HEIGHT_SCALE * STRIP_HEIGHT + 35,
            f"{focus_label}: {focus_start}°–{focus_end}°",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color=FOCUS_COLORS[focus_label],
            bbox=dict(facecolor="white", edgecolor=FOCUS_COLORS[focus_label], alpha=0.92),
            zorder=13
        )
    
    # ------------------------------------------------------------
    # Commune labels
    # ------------------------------------------------------------
    
    if len(commune_labels) > 0:
        for _, row in commune_labels.iterrows():
            ax.text(
                row["x"],
                row["y"],
                row["nom_com"],
                fontsize=8.2,
                fontweight="bold",
                color="#222222",
                ha="center",
                va="center",
                zorder=14,
                path_effects=[
                    pe.withStroke(linewidth=3.0, foreground="white")
                ]
            )
    
    # ------------------------------------------------------------
    # Coloured emoji-style landmarks
    # ------------------------------------------------------------
    
    if len(landmarks) > 0:
        for focus_id, focus_start, _ in FOCUS_RANGES:
            ax.text(
                theta_to_xnew(focus_start),
                -222,
                f"Selected landmarks — {focus_id}",
                ha="left",
                va="bottom",
                fontsize=5,
                fontweight="bold",
                color="#333333",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.90, pad=2),
                zorder=25
            )
    
        for _, row in landmarks.iterrows():
    
            emoji = row["emoji"]
            icon_color = row["color"]
    
            ax.plot(
                [row["x"], row["label_x"]],
                [row["y"], row["label_y"] + 38],
                color="#666666",
                linewidth=0.45,
                linestyle=":",
                alpha=0.30,
                zorder=15
            )
    
            ax.scatter(
                row["x"],
                row["y"],
                s=380,
                marker="o",
                facecolor=icon_color,
                edgecolor="#222222",
                linewidth=0.75,
                alpha=0.95,
                zorder=17
            )
    
            ax.text(
                row["x"],
                row["y"],
                emoji,
                fontsize=14,
                ha="center",
                va="center",
                fontproperties=EMOJI_FONT,
                color="black",
                zorder=18
            )
    
            ax.text(
                row["x"] + 1.5,
                row["y"] + 38,
                str(int(row["number"])),
                ha="center",
                va="center",
                fontsize=7,
                fontweight="bold",
                color="#111111",
                bbox=dict(
                    facecolor="white",
                    edgecolor="#333333",
                    linewidth=0.4,
                    alpha=0.95,
                    boxstyle="circle,pad=0.18"
                ),
                zorder=19
            )
    
            ax.scatter(
                row["label_x"] - 4.2,
                row["label_y"] - 8,
                s=210,
                marker="o",
                facecolor=icon_color,
                edgecolor="#222222",
                linewidth=0.55,
                alpha=0.95,
                zorder=18
            )
    
            ax.text(
                row["label_x"] - 4.2,
                row["label_y"] - 8,
                emoji,
                fontsize=9.5,
                ha="center",
                va="center",
                fontproperties=EMOJI_FONT,
                color="black",
                zorder=19
            )
    
            ax.text(
                row["label_x"] - 1.8,
                row["label_y"],
                f"{int(row['number'])}. {row['name']}",
                ha="left",
                va="top",
                fontsize=6.8,
                color="#222222",
                bbox=dict(
                    facecolor="white",
                    edgecolor="#777777",
                    linewidth=0.35,
                    alpha=0.94,
                    pad=2.3
                ),
                zorder=18
            )
    
    # ------------------------------------------------------------
    # Axes and ticks
    # ------------------------------------------------------------
    
    theta_ticks = [
        0, 30, 60, 90, 120, 150, 180,
        210, 225, 240, 255, 270, 285, 300,
        315, 335, 360
    ]
    
    x_ticks = [theta_to_xnew(t) for t in theta_ticks]
    
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{t}°" for t in theta_ticks], fontsize=8.4)
    
    ax.set_xlim(theta_to_xnew(0), theta_to_xnew(360))
    ax_pop.set_xlim(theta_to_xnew(0), theta_to_xnew(360))
    
    ax.set_ylim(-LANDMARK_SPACE, FOCUS_HEIGHT_SCALE * STRIP_HEIGHT + 90)
    
    # keeps the strip wide
    ax.set_aspect("auto")
    
    ax.set_xlabel(
        "Original angular position θ shown on a modified x-scale",
        fontsize=10,
        labelpad=8
    )
    
    ax.set_ylabel(
        "Height-distorted normalized radial position",
        fontsize=10
    )
    
    ax.set_title("")
    
    ax.grid(True, linestyle="--", alpha=0.14)
    
    ax.text(
        0.01,
        0.055,
        "Sea is represented below the coastline. x- and y-scales are locally distorted for cartographic emphasis.",
        transform=ax.transAxes,
        fontsize=7.7,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.86),
        zorder=25
    )
    
    # ------------------------------------------------------------
    # Horizontal flood colorbar
    # ------------------------------------------------------------
    
    if flood_sm is not None:
        cbar = fig.colorbar(
            flood_sm,
            cax=cax,
            orientation="horizontal"
        )
    
        cbar.set_label("Flooded scenarios", fontsize=9)
        cax.tick_params(axis="x", labelsize=8)
    else:
        cax.set_axis_off()
    
    # ------------------------------------------------------------
    # Legend
    # ------------------------------------------------------------
    
    legend_handles = [
        Patch(facecolor=STYLE["focus_fill"], edgecolor=STYLE["focus_edge"], alpha=STYLE["focus_alpha"], label="Focus area"),
        Patch(facecolor=STYLE["transition_fill"], edgecolor="none", alpha=STYLE["transition_alpha"], label="Transition area"),
        Patch(facecolor=STYLE["sea_face"], edgecolor="none", alpha=STYLE["sea_alpha"], label="Sea background"),
        Patch(facecolor=STYLE["waterbody_face"], edgecolor=STYLE["waterbody_edge"], alpha=STYLE["waterbody_alpha"], label="Inland water bodies"),
        Patch(facecolor=STYLE["urban_face"], edgecolor="none", alpha=STYLE["urban_alpha"], label="Urbanized area"),
        Patch(facecolor=STYLE["pop_focus"], edgecolor="none", alpha=0.95, label="Population in focus areas"),
        Patch(facecolor="#FFB74D", edgecolor=STYLE["flood_edge"], alpha=STYLE["flood_alpha"], label="Flood occurrence"),
        Patch(facecolor=STYLE["building_face"], edgecolor=STYLE["building_edge"], alpha=STYLE["building_alpha"], label="Buildings in focus areas"),
        Line2D([0], [0], color=STYLE["road_context"], lw=1.5, label="Roads context"),
        Line2D([0], [0], color=STYLE["road_focus"], lw=1.8, label="Detailed roads in focus areas"),
        Line2D([0], [0], color=STYLE["railways"], lw=1.5, label="Railways"),
        Line2D([0], [0], color=STYLE["coastline"], lw=2.0, label="Coastline"),
        Line2D([0], [0], color=STYLE["min_boundary"], lw=1.8, label="Minimum boundary"),
        Line2D([0], [0], color=STYLE["max_boundary"], lw=1.7, label="Height-distorted maximum boundary"),
    ]
    
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=7,
        fontsize=6.7,
        frameon=True
    )
    
    fig.subplots_adjust(
        left=0.045,
        right=0.985,
        top=0.96,
        bottom=0.27
    )
    
    # ----------------------------------------------------------
    # 16. SAVE OUTPUTS - SAFE VERSION
    # -------------------------------------------------------------
    
    out_png = OUT_DIR / f"A3_full_context_{FOCUS_NAME}.png"
    out_pdf = OUT_DIR / f"A3_full_context_{FOCUS_NAME}.pdf"
    out_svg = OUT_DIR / f"A3_full_context_{FOCUS_NAME}.svg"
    
    print("\nTrying to save files in:")
    print(OUT_DIR)
    
    fig.suptitle(
        f"Arcachon Basin — {focus_label}: {focus_start}°–{focus_end}°",
        fontsize=14,
        fontweight="bold",
        y=0.988
    )
    
    
    fig.savefig(out_png, dpi=300, facecolor="white")
    fig.savefig(out_pdf, dpi=300, facecolor="white")
    fig.savefig(out_svg, facecolor="white")
    
    saved_paths = [out_png, out_pdf, out_svg]
    
    # -------------------------------------------------------------------
    # 17. SEPARATE A3 MAP FOR EACH FOCUS AREA
    # ---------------------------------------------------------------------
    
    overview_ticks = theta_ticks.copy()
    CREATE_ADDITIONAL_ZOOMED_CROPS = False
    
    for label, start, end in (FOCUS_RANGES if CREATE_ADDITIONAL_ZOOMED_CROPS else []):
        pad = 5
        view_start = max(0, start - pad)
        view_end = min(360, end + pad)
    
        ax.set_xlim(theta_to_xnew(view_start), theta_to_xnew(view_end))
        ax_pop.set_xlim(theta_to_xnew(view_start), theta_to_xnew(view_end))
        ax_spacer.set_xlim(theta_to_xnew(view_start), theta_to_xnew(view_end))
    
        local_ticks = sorted(set(
            [start, end] + list(np.arange(np.ceil(view_start / 5) * 5, view_end + 0.1, 5, dtype=int))
        ))
        ax.set_xticks([theta_to_xnew(t) for t in local_ticks])
        ax.set_xticklabels([f"{t}°" for t in local_ticks], fontsize=8.4)
    
        fig.suptitle(
            f"Arcachon Basin — {label}: {start}°–{end}°",
            fontsize=14,
            fontweight="bold",
            color=FOCUS_COLORS[label],
            y=0.988
        )
    
        focus_png = OUT_DIR / f"A3_{label}_{start}_{end}.png"
        focus_pdf = OUT_DIR / f"A3_{label}_{start}_{end}.pdf"
        focus_svg = OUT_DIR / f"A3_{label}_{start}_{end}.svg"
    
        fig.savefig(focus_png, dpi=300, facecolor="white")
        fig.savefig(focus_pdf, dpi=300, facecolor="white")
        fig.savefig(focus_svg, facecolor="white")
    
        saved_paths.extend([focus_png, focus_pdf, focus_svg])
    
    # Restore overview before displaying it.
    ax.set_xlim(theta_to_xnew(0), theta_to_xnew(360))
    ax_pop.set_xlim(theta_to_xnew(0), theta_to_xnew(360))
    ax_spacer.set_xlim(theta_to_xnew(0), theta_to_xnew(360))
    ax.set_xticks([theta_to_xnew(t) for t in overview_ticks])
    ax.set_xticklabels([f"{t}°" for t in overview_ticks], fontsize=8.4)
    fig.suptitle(
        f"Arcachon Basin — {focus_label}: {focus_start}°–{focus_end}°",
        fontsize=14,
        fontweight="bold",
        color="#222222",
        y=0.988
    )
    
    for path in saved_paths:
        if path.exists():
            print(f"SAVED: {path}")
            print(f"Size: {path.stat().st_size / 1024:.1f} KB")
        else:
            print(f"NOT SAVED: {path}")
    
    plt.close(fig)
    
    print("\nDONE.")

# All three maps are generated sequentially.
for active_focus_id, _, _ in ALL_FOCUS_CONFIGS:
    print("\n" + "=" * 70)
    print(f"CREATING {active_focus_id}")
    print("=" * 70)
    generate_full_context_focus_map(active_focus_id)

print("\nDONE: all three notebook-compatible focus maps were created.")
