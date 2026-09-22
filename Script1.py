# ----------------------------------------------------------------
# FULL RESHAPED POLAR-RADIAL DEFORMATION PIPELINE
# ALL LAYERS TRANSFORMED TOGETHER
#
# Deformation space:
# x = clockwise theta from angle-zero point, degrees
# y = normalized radial position between Mininum_Line and Maximum_Line
#
# Minimum boundary -> y = 0
# Maximum boundary -> y = 1000
# -------------------------------------------------------------------

from pathlib import Path
from datetime import datetime

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

from shapely.geometry import Point, LineString, Polygon, MultiLineString
from shapely.ops import unary_union, linemerge, transform as shapely_transform
from matplotlib.patches import Polygon as MplPolygon


# ============================================================
# 1. USER SETTINGS
# ============================================================

GDB_PATH = Path(
    r"C:\IGN_French_Visa\Arcachon_Project\Polar_Log_Python\Polar_Log_Corridor.gdb"
)

OUTPUT_ROOT = Path(
    r"C:\IGN_French_Visa\Arcachon_Project\Polar_Log_Python\Full_Reshape_Polar_Pipeline"
)

TARGET_CRS = "EPSG:2154"

# Control layers
RESHAPE_MIN_LAYER = "Reshape_Min_Line_v2"
RESHAPE_MAX_LAYER = "Reshape_Max_Line_v2"
COASTLINE_LAYER = "study_area_coastline"
ANGLE_ZERO_LAYER = "corridor_angle_zero_point"

CENTER_LAYER_CANDIDATES = [
    "center_point",
    "corridor_center_point",
    "polar_center_point"
]

# Internal layers
LAYER_FLOOD = "corridor_flood_occurrrence"
LAYER_AOI = "corridor_aio_polygon"
LAYER_BUILDINGS = "corridor_buildings"
LAYER_ROADS = "corridor_roads"
LAYER_RAILWAYS = "corridor_railways"
LAYER_POPULATION_GRID = "corridor_population_grid"
LAYER_POPULATION_POINTS = "corridor_population_point"
LAYER_TRANSPORT = "corridor_transport_facilities"
LAYER_TRAFFIC = "corridor_traffic"
LAYER_WATER = "corridor_water"
LAYER_WATERBODIES = "corridor_waterbodies"

# Transformation switches
TRANSFORM_FLOOD = True
TRANSFORM_AOI = True
TRANSFORM_BUILDINGS = True
TRANSFORM_ROADS = True
TRANSFORM_RAILWAYS = True
TRANSFORM_POPULATION_GRID = True
TRANSFORM_POPULATION_POINTS = True
TRANSFORM_TRANSPORT = True
TRANSFORM_TRAFFIC = True
TRANSFORM_WATER_LINES = True
TRANSFORM_WATERBODIES = True

# for building
SAMPLE_BUILDINGS = False
MAX_BUILDING_FEATURES = 5000

# Core deformation settings
STRIP_HEIGHT = 1000
ANGLE_STEP_DEG = 2
RAY_LENGTH_MULTIPLIER = 2.5
MIN_CORRIDOR_WIDTH_M = 100

# Keep features only inside corridor
V_MIN = 0.0
V_MAX = 1.0

# Densification spacing
DENSIFY_LINE_SPACING_M = 75
DENSIFY_POLYGON_SPACING_M = 50
COASTLINE_DENSIFY_SPACING_M = 50

# Seam handling
SEAM_JUMP_DEG = 180

# Width smoothing
SMOOTH_WIDTH_FOR_DEFORMATION = True
WIDTH_SMOOTH_WINDOW_DEG = 10
SMOOTH_ITERATIONS = 2

# Population
POPULATION_FIELD = "ind"
THETA_SLICE_DEG = 5

# Expected AOIs
EXPECTED_AOI_COUNT = 27

# Plot settings
PLOT_Y_MIN = 0
PLOT_Y_MAX = 1000

MAX_BUILDINGS_TO_DRAW = 4000

FLOOD_VALUE_CANDIDATES = [
    "nb_scn",
    "NB_SCN",
    "Flooded_Scenarios",
    "flooded_scenarios",
    "count",
    "COUNT"
]

AOI_LABEL_CANDIDATES = [
    "AOI_label",
    "AOI_ID",
    "aoi_id",
    "AOI",
    "aoi",
    "Name",
    "name",
    "NOM",
    "nom",
    "id",
    "ID",
    "OBJECTID"
]


# ------------------------------------
# 2. OUTPUT FOLDERS
# ------------------------------------

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

RUN_FOLDER = OUTPUT_ROOT / f"full_reshape_pipeline_{run_id}"
PROFILE_FOLDER = RUN_FOLDER / "01_profile"
LAYER_FOLDER = RUN_FOLDER / "02_transformed_layers"
POP_FOLDER = RUN_FOLDER / "03_population_slices"
PLOT_FOLDER = RUN_FOLDER / "04_plots"
SUMMARY_FOLDER = RUN_FOLDER / "05_summary"

for folder in [
    PROFILE_FOLDER,
    LAYER_FOLDER,
    POP_FOLDER,
    PLOT_FOLDER,
    SUMMARY_FOLDER
]:
    folder.mkdir(parents=True, exist_ok=True)

print("Main output folder:")
print(RUN_FOLDER)


# ------------------------------------------
# 3. HELPER FUNCTIONS: READING AND CLEANING
# ------------------------------------------

def read_gdb_layer_or_empty(gdb_path, layer_name, target_crs):
    try:
        print("Reading:", layer_name)

        try:
            gdf = gpd.read_file(
                gdb_path,
                layer=layer_name,
                engine="pyogrio"
            )
        except Exception:
            gdf = gpd.read_file(
                gdb_path,
                layer=layer_name
            )

        return clean_gdf(gdf, target_crs)

    except Exception as e:
        print("Could not read layer:", layer_name)
        print(e)

        return gpd.GeoDataFrame(
            columns=["geometry"],
            geometry="geometry",
            crs=target_crs
        )


def force_2d(geom):
    if geom is None or geom.is_empty:
        return geom

    try:
        return shapely_transform(
            lambda x, y, z=None: (x, y),
            geom
        )
    except Exception:
        return geom


def fix_geometry(geom):
    if geom is None or geom.is_empty:
        return geom

    try:
        if not geom.is_valid:
            geom = geom.buffer(0)
    except Exception:
        pass

    return geom


def clean_gdf(gdf, target_crs):
    if gdf is None:
        return gpd.GeoDataFrame(
            columns=["geometry"],
            geometry="geometry",
            crs=target_crs
        )

    gdf = gdf.copy()

    if "geometry" not in gdf.columns:
        return gpd.GeoDataFrame(
            columns=["geometry"],
            geometry="geometry",
            crs=target_crs
        )

    gdf = gdf[gdf.geometry.notnull()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()

    if len(gdf) == 0:
        return gpd.GeoDataFrame(
            columns=["geometry"],
            geometry="geometry",
            crs=target_crs
        )

    if gdf.crs is None:
        gdf = gdf.set_crs(target_crs)
    else:
        gdf = gdf.to_crs(target_crs)

    gdf["geometry"] = gdf.geometry.apply(force_2d)
    gdf["geometry"] = gdf.geometry.apply(fix_geometry)

    gdf = gdf[gdf.geometry.notnull()].copy()
    gdf = gdf[~gdf.geometry.is_empty].reset_index(drop=True)

    return gdf


def get_union_geometry(gdf):
    try:
        return gdf.geometry.union_all()
    except Exception:
        return unary_union(list(gdf.geometry))


def get_first_existing_field(gdf, candidates):
    for col in candidates:
        if col in gdf.columns:
            return col

    return None


# -------------------------------------------------
# 4. HELPER FUNCTIONS: GEOMETRY EXTRACTION
# ------------------------------------------------

def extract_lines(geom):
    lines = []

    if geom is None or geom.is_empty:
        return lines

    if geom.geom_type == "LineString":
        lines.append(geom)

    elif geom.geom_type == "MultiLineString":
        lines.extend(list(geom.geoms))

    elif geom.geom_type == "GeometryCollection":
        for part in geom.geoms:
            lines.extend(extract_lines(part))

    return lines


def extract_polygons(geom):
    polygons = []

    if geom is None or geom.is_empty:
        return polygons

    if geom.geom_type == "Polygon":
        polygons.append(geom)

    elif geom.geom_type == "MultiPolygon":
        polygons.extend(list(geom.geoms))

    elif geom.geom_type == "GeometryCollection":
        for part in geom.geoms:
            polygons.extend(extract_polygons(part))

    return polygons


def make_single_line(gdf):
    lines = []

    for geom in gdf.geometry:
        lines.extend(extract_lines(geom))

    if len(lines) == 0:
        raise ValueError("No line geometry found.")

    try:
        merged = linemerge(unary_union(lines))
    except Exception:
        if len(lines) == 1:
            merged = lines[0]
        else:
            merged = MultiLineString(lines)

    if merged.geom_type == "MultiLineString":
        parts = list(merged.geoms)
        parts = sorted(parts, key=lambda line: line.length, reverse=True)
        return parts[0]

    return merged


def densify_linestring(line, spacing_m):
    if line is None or line.is_empty:
        return line

    if line.length <= spacing_m:
        return line

    distances = list(np.arange(0, line.length, spacing_m))

    if len(distances) == 0 or distances[-1] < line.length:
        distances.append(line.length)

    coords = []

    for d in distances:
        p = line.interpolate(d)
        coords.append((p.x, p.y))

    if len(coords) < 2:
        return line

    return LineString(coords)


# ---------------------------------------------
# 5. HELPER FUNCTIONS: PROFILE CREATION
# ---------------------------------------------

def extract_points_from_intersection(geom):
    points = []

    if geom is None or geom.is_empty:
        return points

    if geom.geom_type == "Point":
        points.append(geom)

    elif geom.geom_type == "MultiPoint":
        points.extend(list(geom.geoms))

    elif geom.geom_type == "LineString":
        coords = list(geom.coords)

        if len(coords) > 0:
            points.append(Point(coords[0]))
            points.append(Point(coords[-1]))

    elif geom.geom_type == "MultiLineString":
        for line in geom.geoms:
            coords = list(line.coords)

            if len(coords) > 0:
                points.append(Point(coords[0]))
                points.append(Point(coords[-1]))

    elif geom.geom_type == "GeometryCollection":
        for part in geom.geoms:
            points.extend(extract_points_from_intersection(part))

    return points


def create_ray(center_point, theta_deg, zero_angle_rad, ray_length):
    angle_rad = zero_angle_rad - np.radians(theta_deg)

    x2 = center_point.x + ray_length * np.cos(angle_rad)
    y2 = center_point.y + ray_length * np.sin(angle_rad)

    return LineString([
        (center_point.x, center_point.y),
        (x2, y2)
    ])


def get_intersection_distances(ray, line_geom, center_point):
    inter = ray.intersection(line_geom)
    points = extract_points_from_intersection(inter)

    distances = []

    for p in points:
        d = center_point.distance(p)

        if np.isfinite(d):
            distances.append(d)

    distances = sorted(list(set([round(d, 6) for d in distances])))

    return distances, points


def smooth_open_array(values, window_points, iterations):
    arr = np.array(values, dtype=float)

    s = pd.Series(arr)

    if s.notna().any():
        s = s.interpolate(limit_direction="both")

    s = s.bfill()
    s = s.ffill()

    arr = s.to_numpy()

    if window_points < 3:
        return arr

    if window_points % 2 == 0:
        window_points += 1

    kernel = np.ones(window_points) / window_points

    for _ in range(iterations):
        pad = window_points // 2

        padded = np.pad(
            arr,
            pad_width=pad,
            mode="reflect"
        )

        arr = np.convolve(
            padded,
            kernel,
            mode="valid"
        )

    return arr


def smooth_width_by_valid_runs(profile_df):
    df = profile_df.copy()

    width_raw = df["width_m"].to_numpy(dtype=float)

    valid = (
        (df["valid"] == True).to_numpy()
        & np.isfinite(width_raw)
    )

    width_smooth = np.full_like(
        width_raw,
        np.nan,
        dtype=float
    )

    valid_indices = np.where(valid)[0]

    if len(valid_indices) == 0:
        df["width_used_m"] = np.nan
        df["r_max_used_m"] = np.nan
        return df

    runs = []
    current_run = [valid_indices[0]]

    for idx in valid_indices[1:]:
        if idx == current_run[-1] + 1:
            current_run.append(idx)
        else:
            runs.append(current_run)
            current_run = [idx]

    runs.append(current_run)

    window_points = int(WIDTH_SMOOTH_WINDOW_DEG / ANGLE_STEP_DEG)

    if window_points < 3:
        window_points = 3

    for run in runs:
        run_values = width_raw[run]

        if len(run_values) < 3:
            width_smooth[run] = run_values
        else:
            smooth_values = smooth_open_array(
                run_values,
                window_points,
                SMOOTH_ITERATIONS
            )

            width_smooth[run] = smooth_values

    df["width_used_m"] = width_smooth
    df["r_max_used_m"] = df["r_min_m"] + df["width_used_m"]

    return df


def build_radial_profile(
    reshape_min,
    reshape_max,
    coastline,
    center_point,
    zero_angle_rad
):
    min_geom = get_union_geometry(reshape_min)
    max_geom = get_union_geometry(reshape_max)

    all_bounds = pd.concat(
        [
            reshape_min[["geometry"]],
            reshape_max[["geometry"]],
            coastline[["geometry"]]
        ],
        ignore_index=True
    )

    all_bounds = gpd.GeoDataFrame(
        all_bounds,
        geometry="geometry",
        crs=TARGET_CRS
    )

    minx, miny, maxx, maxy = all_bounds.total_bounds

    bbox_diag = np.sqrt(
        (maxx - minx) ** 2
        + (maxy - miny) ** 2
    )

    ray_length = bbox_diag * RAY_LENGTH_MULTIPLIER

    print("Ray length, m:", round(ray_length, 2))

    profile_records = []
    ray_records = []
    min_point_records = []
    max_point_records = []

    for theta in np.arange(0, 360, ANGLE_STEP_DEG):

        ray = create_ray(
            center_point,
            theta,
            zero_angle_rad,
            ray_length
        )

        min_distances, min_points = get_intersection_distances(
            ray,
            min_geom,
            center_point
        )

        max_distances, max_points = get_intersection_distances(
            ray,
            max_geom,
            center_point
        )

        valid = False
        r_min = np.nan
        r_max = np.nan
        width = np.nan
        status = "no_valid_intersection"

        if len(min_distances) > 0 and len(max_distances) > 0:
            r_min_candidate = min(min_distances)

            max_candidates = [
                d for d in max_distances
                if d > r_min_candidate + MIN_CORRIDOR_WIDTH_M
            ]

            if len(max_candidates) > 0:
                r_min = r_min_candidate
                r_max = min(max_candidates)
                width = r_max - r_min
                valid = True
                status = "valid"
            else:
                status = "max_not_outside_min"

        profile_records.append({
            "theta_deg": float(theta),
            "r_min_m": r_min,
            "r_max_m": r_max,
            "width_m": width,
            "valid": valid,
            "status": status,
            "n_min_intersections": len(min_distances),
            "n_max_intersections": len(max_distances)
        })

        if theta % 10 == 0:
            ray_records.append({
                "theta_deg": float(theta),
                "valid": valid,
                "geometry": ray
            })

        if valid:
            angle_rad = zero_angle_rad - np.radians(theta)

            p_min = Point(
                center_point.x + r_min * np.cos(angle_rad),
                center_point.y + r_min * np.sin(angle_rad)
            )

            p_max = Point(
                center_point.x + r_max * np.cos(angle_rad),
                center_point.y + r_max * np.sin(angle_rad)
            )

            min_point_records.append({
                "theta_deg": float(theta),
                "r_min_m": r_min,
                "geometry": p_min
            })

            max_point_records.append({
                "theta_deg": float(theta),
                "r_max_m": r_max,
                "geometry": p_max
            })

    profile_df = pd.DataFrame(profile_records)

    if SMOOTH_WIDTH_FOR_DEFORMATION:
        profile_df = smooth_width_by_valid_runs(profile_df)
    else:
        profile_df["width_used_m"] = profile_df["width_m"]
        profile_df["r_max_used_m"] = profile_df["r_max_m"]

    rays_gdf = gpd.GeoDataFrame(
        ray_records,
        geometry="geometry",
        crs=TARGET_CRS
    )

    min_points_gdf = gpd.GeoDataFrame(
        min_point_records,
        geometry="geometry",
        crs=TARGET_CRS
    )

    max_points_gdf = gpd.GeoDataFrame(
        max_point_records,
        geometry="geometry",
        crs=TARGET_CRS
    )

    return profile_df, rays_gdf, min_points_gdf, max_points_gdf


# ----------------------------------------
# 6. HELPER FUNCTIONS: TRANSFORMATION
# ---------------------------------------

def circular_angle_from_point(center_point, zero_angle_rad, x, y):
    angle_rad = np.arctan2(
        y - center_point.y,
        x - center_point.x
    )

    theta_rad = zero_angle_rad - angle_rad
    theta_deg = np.degrees(theta_rad) % 360

    r_m = np.sqrt(
        (x - center_point.x) ** 2
        + (y - center_point.y) ** 2
    )

    return theta_deg, r_m


def interpolate_profile(theta_deg, profile_df):
    theta_deg = theta_deg % 360

    lower_theta = int(np.floor(theta_deg / ANGLE_STEP_DEG) * ANGLE_STEP_DEG)
    upper_theta = lower_theta + ANGLE_STEP_DEG

    if upper_theta >= 360:
        return None

    row0 = profile_df[profile_df["theta_deg"] == lower_theta]
    row1 = profile_df[profile_df["theta_deg"] == upper_theta]

    if len(row0) == 0 or len(row1) == 0:
        return None

    row0 = row0.iloc[0]
    row1 = row1.iloc[0]

    if not bool(row0["valid"]) or not bool(row1["valid"]):
        return None

    for col in ["r_min_m", "width_used_m"]:
        if not np.isfinite(row0[col]) or not np.isfinite(row1[col]):
            return None

    t = (theta_deg - lower_theta) / ANGLE_STEP_DEG

    r_min = (1 - t) * row0["r_min_m"] + t * row1["r_min_m"]
    width = (1 - t) * row0["width_used_m"] + t * row1["width_used_m"]

    if width <= 0:
        return None

    return r_min, width


def transform_xy_to_strip(
    x,
    y,
    center_point,
    zero_angle_rad,
    profile_df,
    v_min=V_MIN,
    v_max=V_MAX
):
    theta_deg, r_m = circular_angle_from_point(
        center_point,
        zero_angle_rad,
        x,
        y
    )

    profile_value = interpolate_profile(
        theta_deg,
        profile_df
    )

    if profile_value is None:
        return None

    r_min, width = profile_value

    v = (r_m - r_min) / width

    if v < v_min or v > v_max:
        return None

    x_new = theta_deg
    y_new = v * STRIP_HEIGHT

    return x_new, y_new, theta_deg, v


def transform_line_to_strip(
    line,
    center_point,
    zero_angle_rad,
    profile_df,
    spacing_m
):
    line = densify_linestring(
        line,
        spacing_m
    )

    coords = list(line.coords)

    segments = []
    current_coords = []
    previous_xy = None

    for coord in coords:
        x, y = coord[0], coord[1]

        result = transform_xy_to_strip(
            x,
            y,
            center_point,
            zero_angle_rad,
            profile_df
        )

        if result is None:
            if len(current_coords) >= 2:
                segments.append(LineString(current_coords))

            current_coords = []
            previous_xy = None
            continue

        x_new, y_new, theta_deg, v = result
        new_xy = (x_new, y_new)

        if previous_xy is not None:
            dx = abs(new_xy[0] - previous_xy[0])

            if dx > SEAM_JUMP_DEG:
                if len(current_coords) >= 2:
                    segments.append(LineString(current_coords))

                current_coords = []

        current_coords.append(new_xy)
        previous_xy = new_xy

    if len(current_coords) >= 2:
        segments.append(LineString(current_coords))

    return segments


def transform_line_layer_by_vertices(
    gdf,
    center_point,
    zero_angle_rad,
    profile_df,
    layer_name,
    spacing_m=DENSIFY_LINE_SPACING_M
):
    records = []
    skipped = 0

    for idx, row in gdf.iterrows():
        lines = extract_lines(row.geometry)

        if len(lines) == 0:
            skipped += 1
            continue

        for line_id, line in enumerate(lines):
            segments = transform_line_to_strip(
                line,
                center_point,
                zero_angle_rad,
                profile_df,
                spacing_m
            )

            if len(segments) == 0:
                skipped += 1

            for seg_id, seg in enumerate(segments):
                attrs = row.drop(labels="geometry").to_dict()
                attrs["source_id"] = idx
                attrs["line_id"] = line_id
                attrs["seg_id"] = seg_id
                attrs["layer_name"] = layer_name
                attrs["geometry"] = seg

                records.append(attrs)

    if len(records) == 0:
        out = gpd.GeoDataFrame(
            columns=["geometry"],
            geometry="geometry",
            crs=None
        )
    else:
        out = gpd.GeoDataFrame(
            records,
            geometry="geometry",
            crs=None
        )

    print(layer_name, "| transformed line segments:", len(out), "| skipped:", skipped)

    return out, skipped


def transform_ring_to_strip(
    ring_coords,
    center_point,
    zero_angle_rad,
    profile_df
):
    line = LineString(ring_coords)

    line = densify_linestring(
        line,
        DENSIFY_POLYGON_SPACING_M
    )

    coords = list(line.coords)

    transformed = []
    previous_xy = None

    for coord in coords:
        x, y = coord[0], coord[1]

        result = transform_xy_to_strip(
            x,
            y,
            center_point,
            zero_angle_rad,
            profile_df
        )

        if result is None:
            return None

        x_new, y_new, theta_deg, v = result
        new_xy = (x_new, y_new)

        if previous_xy is not None:
            dx = abs(new_xy[0] - previous_xy[0])

            if dx > SEAM_JUMP_DEG:
                return None

        transformed.append(new_xy)
        previous_xy = new_xy

    if len(transformed) < 4:
        return None

    if transformed[0] != transformed[-1]:
        transformed.append(transformed[0])

    return transformed


def transform_polygon_to_strip(
    polygon,
    center_point,
    zero_angle_rad,
    profile_df
):
    polygon = force_2d(polygon)
    polygon = fix_geometry(polygon)

    if polygon is None or polygon.is_empty:
        return None

    exterior_coords = list(polygon.exterior.coords)

    transformed_exterior = transform_ring_to_strip(
        exterior_coords,
        center_point,
        zero_angle_rad,
        profile_df
    )

    if transformed_exterior is None:
        return None

    transformed_interiors = []

    for interior in polygon.interiors:
        transformed_interior = transform_ring_to_strip(
            list(interior.coords),
            center_point,
            zero_angle_rad,
            profile_df
        )

        if transformed_interior is not None:
            transformed_interiors.append(transformed_interior)

    try:
        new_polygon = Polygon(
            transformed_exterior,
            transformed_interiors
        )

        new_polygon = fix_geometry(new_polygon)

        if new_polygon is None or new_polygon.is_empty:
            return None

        if new_polygon.area <= 0:
            return None

        return new_polygon

    except Exception:
        return None


def transform_polygon_layer_by_vertices(
    gdf,
    center_point,
    zero_angle_rad,
    profile_df,
    layer_name
):
    polygon_records = []
    fallback_records = []

    source_part_count = 0
    failed_count = 0

    for idx, row in gdf.iterrows():
        polygons = extract_polygons(row.geometry)

        if len(polygons) == 0:
            failed_count += 1
            continue

        for poly_id, poly in enumerate(polygons):
            source_part_count += 1

            attrs = row.drop(labels="geometry").to_dict()
            attrs["source_id"] = idx
            attrs["poly_id"] = poly_id
            attrs["layer_name"] = layer_name

            new_polygon = transform_polygon_to_strip(
                poly,
                center_point,
                zero_angle_rad,
                profile_df
            )

            if new_polygon is not None:
                poly_attrs = attrs.copy()
                poly_attrs["geometry"] = new_polygon
                polygon_records.append(poly_attrs)

            else:
                failed_count += 1

                # fallback representative point
                try:
                    p = poly.representative_point()

                    result = transform_xy_to_strip(
                        p.x,
                        p.y,
                        center_point,
                        zero_angle_rad,
                        profile_df
                    )

                    if result is not None:
                        x_new, y_new, theta_deg, v = result

                        point_attrs = attrs.copy()
                        point_attrs["theta_deg"] = theta_deg
                        point_attrs["v_norm"] = v
                        point_attrs["y_norm"] = y_new
                        point_attrs["geometry"] = Point(x_new, y_new)

                        fallback_records.append(point_attrs)

                except Exception:
                    pass

    if len(polygon_records) == 0:
        polygon_gdf = gpd.GeoDataFrame(
            columns=["geometry"],
            geometry="geometry",
            crs=None
        )
    else:
        polygon_gdf = gpd.GeoDataFrame(
            polygon_records,
            geometry="geometry",
            crs=None
        )

    if len(fallback_records) == 0:
        fallback_gdf = gpd.GeoDataFrame(
            columns=["geometry"],
            geometry="geometry",
            crs=None
        )
    else:
        fallback_gdf = gpd.GeoDataFrame(
            fallback_records,
            geometry="geometry",
            crs=None
        )

    print("\n" + layer_name)
    print("Source polygon parts:", source_part_count)
    print("Transformed polygons:", len(polygon_gdf))
    print("Failed polygon parts:", failed_count)
    print("Fallback points:", len(fallback_gdf))

    return polygon_gdf, fallback_gdf, failed_count, source_part_count


def transform_point_layer_direct(
    gdf,
    center_point,
    zero_angle_rad,
    profile_df,
    layer_name
):
    records = []
    skipped = 0

    for idx, row in gdf.iterrows():
        geom = row.geometry

        if geom is None or geom.is_empty:
            skipped += 1
            continue

        if geom.geom_type == "Point":
            p = geom
        else:
            p = geom.representative_point()

        result = transform_xy_to_strip(
            p.x,
            p.y,
            center_point,
            zero_angle_rad,
            profile_df
        )

        if result is None:
            skipped += 1
            continue

        x_new, y_new, theta_deg, v = result

        attrs = row.drop(labels="geometry").to_dict()
        attrs["source_id"] = idx
        attrs["theta_deg"] = theta_deg
        attrs["v_norm"] = v
        attrs["y_norm"] = y_new
        attrs["layer_name"] = layer_name
        attrs["geometry"] = Point(x_new, y_new)

        records.append(attrs)

    if len(records) == 0:
        out = gpd.GeoDataFrame(
            columns=["geometry"],
            geometry="geometry",
            crs=None
        )
    else:
        out = gpd.GeoDataFrame(
            records,
            geometry="geometry",
            crs=None
        )

    print(layer_name, "| transformed points:", len(out), "| skipped:", skipped)

    return out, skipped


def make_strip_boundary_lines(profile_df):
    valid = profile_df[
        (profile_df["valid"] == True)
        & (profile_df["width_used_m"].notnull())
    ].copy()

    valid = valid.sort_values("theta_deg").reset_index(drop=True)

    records_min = []
    records_max = []

    current_min = []
    current_max = []
    previous_theta = None
    part_id = 1

    for _, row in valid.iterrows():
        theta = row["theta_deg"]

        if previous_theta is not None:
            if theta - previous_theta > ANGLE_STEP_DEG * 1.5:
                if len(current_min) >= 2:
                    records_min.append({
                        "part_id": part_id,
                        "geometry": LineString(current_min)
                    })

                    records_max.append({
                        "part_id": part_id,
                        "geometry": LineString(current_max)
                    })

                    part_id += 1

                current_min = []
                current_max = []

        current_min.append((theta, 0))
        current_max.append((theta, STRIP_HEIGHT))

        previous_theta = theta

    if len(current_min) >= 2:
        records_min.append({
            "part_id": part_id,
            "geometry": LineString(current_min)
        })

        records_max.append({
            "part_id": part_id,
            "geometry": LineString(current_max)
        })

    min_gdf = gpd.GeoDataFrame(
        records_min,
        geometry="geometry",
        crs=None
    )

    max_gdf = gpd.GeoDataFrame(
        records_max,
        geometry="geometry",
        crs=None
    )

    return min_gdf, max_gdf


# -------------------------------
# 7. SAVE AND PLOT HELPERS
# -------------------------------

def save_vector(gdf, path):
    if gdf is None or len(gdf) == 0:
        print("Skipped empty:", path.name)
        return

    out = gdf.copy()

    for col in out.columns:
        if col == "geometry":
            continue

        if out[col].dtype == "object":
            out[col] = out[col].astype(str)

    try:
        if path.suffix.lower() == ".geojson":
            out.to_file(path, driver="GeoJSON")
        else:
            out.to_file(path)

        print("Saved:", path)

    except Exception as e:
        print("Could not save:", path)
        print(e)


def save_both(gdf, base_name):
    save_vector(
        gdf,
        LAYER_FOLDER / f"{base_name}.geojson"
    )

    save_vector(
        gdf,
        LAYER_FOLDER / f"{base_name}.shp"
    )


def draw_lines(ax, gdf, color, linewidth, label=None, alpha=1.0, linestyle="-"):
    if gdf is None or len(gdf) == 0:
        return

    first_label = label

    for geom in gdf.geometry:
        lines = extract_lines(geom)

        for line in lines:
            xs, ys = line.xy

            ax.plot(
                xs,
                ys,
                color=color,
                linewidth=linewidth,
                alpha=alpha,
                linestyle=linestyle,
                label=first_label
            )

            first_label = None


def draw_polygons(
    ax,
    gdf,
    facecolor="lightblue",
    edgecolor="none",
    linewidth=0.15,
    alpha=0.65,
    label=None,
    value_field=None,
    cmap_name="Blues"
):
    if gdf is None or len(gdf) == 0:
        return None

    first_label = label
    scalar_mappable = None

    if value_field is not None and value_field in gdf.columns:
        values = pd.to_numeric(
            gdf[value_field],
            errors="coerce"
        )

        finite_values = values[np.isfinite(values)]

        if len(finite_values) > 0:
            norm = colors.Normalize(
                vmin=float(finite_values.min()),
                vmax=float(finite_values.max())
            )

            cmap = cm.get_cmap(cmap_name)

            scalar_mappable = cm.ScalarMappable(
                norm=norm,
                cmap=cmap
            )
        else:
            value_field = None

    for _, row in gdf.iterrows():
        polygons = extract_polygons(row.geometry)

        for poly in polygons:
            coords = list(poly.exterior.coords)

            if len(coords) < 4:
                continue

            if value_field is not None and scalar_mappable is not None:
                value = pd.to_numeric(
                    row[value_field],
                    errors="coerce"
                )

                if np.isfinite(value):
                    fc = scalar_mappable.to_rgba(value)
                else:
                    fc = facecolor
            else:
                fc = facecolor

            patch = MplPolygon(
                coords,
                closed=True,
                facecolor=fc,
                edgecolor=edgecolor,
                linewidth=linewidth,
                alpha=alpha,
                label=first_label
            )

            ax.add_patch(patch)

            first_label = None

    return scalar_mappable


def draw_points(ax, gdf, color, size, label=None, alpha=0.7, marker="o"):
    if gdf is None or len(gdf) == 0:
        return

    point_gdf = gdf[gdf.geometry.geom_type == "Point"].copy()

    if len(point_gdf) == 0:
        return

    ax.scatter(
        point_gdf.geometry.x,
        point_gdf.geometry.y,
        color=color,
        s=size,
        alpha=alpha,
        marker=marker,
        label=label
    )


def clean_legend(ax, ncol=4, y=-0.34):
    handles, labels = ax.get_legend_handles_labels()

    unique = {}

    for handle, label in zip(handles, labels):
        if label not in unique and label != "":
            unique[label] = handle

    ax.legend(
        unique.values(),
        unique.keys(),
        loc="lower center",
        bbox_to_anchor=(0.5, y),
        ncol=ncol,
        fontsize=8,
        frameon=True
    )


def setup_strip_axis(ax, title):
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Clockwise theta from angle-zero point, degrees")
    ax.set_ylabel("Normalized radial position")
    ax.set_xlim(0, 360)
    ax.set_ylim(PLOT_Y_MIN, PLOT_Y_MAX)
    ax.set_aspect("auto")
    ax.grid(True, linestyle="--", alpha=0.20)


# -------------------------------
# 8. POPULATION SLICES
# ------------------------------

def calculate_population_by_theta_slice(pop_grid_gdf):
    if pop_grid_gdf is None or len(pop_grid_gdf) == 0:
        return pd.DataFrame(
            columns=["theta_start", "theta_end", "theta_mid", "population"]
        )

    if POPULATION_FIELD not in pop_grid_gdf.columns:
        print("Population field not found:", POPULATION_FIELD)
        return pd.DataFrame(
            columns=["theta_start", "theta_end", "theta_mid", "population"]
        )

    rows = []

    for theta_start in np.arange(0, 360, THETA_SLICE_DEG):
        theta_end = theta_start + THETA_SLICE_DEG

        slice_poly = Polygon([
            (theta_start, 0),
            (theta_end, 0),
            (theta_end, STRIP_HEIGHT),
            (theta_start, STRIP_HEIGHT),
            (theta_start, 0)
        ])

        pop_sum = 0.0

        for _, row in pop_grid_gdf.iterrows():
            geom = row.geometry

            if geom is None or geom.is_empty:
                continue

            pop_value = pd.to_numeric(
                row[POPULATION_FIELD],
                errors="coerce"
            )

            if not np.isfinite(pop_value):
                continue

            original_area = geom.area

            if original_area <= 0:
                continue

            inter = geom.intersection(slice_poly)

            if inter.is_empty:
                continue

            fraction = inter.area / original_area
            pop_sum += pop_value * fraction

        rows.append({
            "theta_start": theta_start,
            "theta_end": theta_end,
            "theta_mid": theta_start + THETA_SLICE_DEG / 2,
            "population": pop_sum
        })

    return pd.DataFrame(rows)


# --------------------------------
# 9. READ SOURCE LAYERS
# ------------------------------

reshape_min = read_gdb_layer_or_empty(GDB_PATH, RESHAPE_MIN_LAYER, TARGET_CRS)
reshape_max = read_gdb_layer_or_empty(GDB_PATH, RESHAPE_MAX_LAYER, TARGET_CRS)
coastline = read_gdb_layer_or_empty(GDB_PATH, COASTLINE_LAYER, TARGET_CRS)

flood = read_gdb_layer_or_empty(GDB_PATH, LAYER_FLOOD, TARGET_CRS)
aoi = read_gdb_layer_or_empty(GDB_PATH, LAYER_AOI, TARGET_CRS)
buildings = read_gdb_layer_or_empty(GDB_PATH, LAYER_BUILDINGS, TARGET_CRS)
roads = read_gdb_layer_or_empty(GDB_PATH, LAYER_ROADS, TARGET_CRS)
railways = read_gdb_layer_or_empty(GDB_PATH, LAYER_RAILWAYS, TARGET_CRS)
population_grid = read_gdb_layer_or_empty(GDB_PATH, LAYER_POPULATION_GRID, TARGET_CRS)
population_points = read_gdb_layer_or_empty(GDB_PATH, LAYER_POPULATION_POINTS, TARGET_CRS)
transport = read_gdb_layer_or_empty(GDB_PATH, LAYER_TRANSPORT, TARGET_CRS)
traffic = read_gdb_layer_or_empty(GDB_PATH, LAYER_TRAFFIC, TARGET_CRS)
water_lines = read_gdb_layer_or_empty(GDB_PATH, LAYER_WATER, TARGET_CRS)
waterbodies = read_gdb_layer_or_empty(GDB_PATH, LAYER_WATERBODIES, TARGET_CRS)

print("\nSOURCE COUNTS")
print("Reshape min:", len(reshape_min))
print("Reshape max:", len(reshape_max))
print("Coastline:", len(coastline))
print("Flood:", len(flood))
print("AOIs:", len(aoi))
print("Buildings:", len(buildings))
print("Roads:", len(roads))
print("Railways:", len(railways))
print("Population grid:", len(population_grid))
print("Population points:", len(population_points))
print("Transport:", len(transport))
print("Traffic:", len(traffic))
print("Water lines:", len(water_lines))
print("Waterbodies:", len(waterbodies))

if len(aoi) != EXPECTED_AOI_COUNT:
    print("WARNING: AOI count is not 27. AOIs found:", len(aoi))


# -------------------------------------
# 10. OPTIONAL BUILDING SAMPLE
# ------------------------------------

if SAMPLE_BUILDINGS and len(buildings) > MAX_BUILDING_FEATURES:
    buildings = buildings.sample(
        n=MAX_BUILDING_FEATURES,
        random_state=42
    ).reset_index(drop=True)

    print("Buildings sampled to:", len(buildings))


# ---------------------------
# 11. CENTRE AND ANGLE-ZERO
# -------------------------

center_gdf = None

for center_layer in CENTER_LAYER_CANDIDATES:
    test_gdf = read_gdb_layer_or_empty(
        GDB_PATH,
        center_layer,
        TARGET_CRS
    )

    if len(test_gdf) > 0:
        center_gdf = test_gdf
        print("Using centre layer:", center_layer)
        break

angle_zero_gdf = read_gdb_layer_or_empty(
    GDB_PATH,
    ANGLE_ZERO_LAYER,
    TARGET_CRS
)

if center_gdf is not None and len(center_gdf) > 0:
    center_point = center_gdf.geometry.iloc[0]

    if center_point.geom_type != "Point":
        center_point = center_point.centroid
else:
    combined = pd.concat(
        [
            reshape_min[["geometry"]],
            reshape_max[["geometry"]]
        ],
        ignore_index=True
    )

    combined = gpd.GeoDataFrame(
        combined,
        geometry="geometry",
        crs=TARGET_CRS
    )

    center_point = get_union_geometry(combined).centroid
    print("No centre layer found. Using reshaped line centroid.")

if len(angle_zero_gdf) > 0:
    angle_zero_point = angle_zero_gdf.geometry.iloc[0]

    if angle_zero_point.geom_type != "Point":
        angle_zero_point = angle_zero_point.centroid
else:
    minx, miny, maxx, maxy = reshape_max.total_bounds
    angle_zero_point = Point(minx, miny)

    print("No angle-zero point found. Using lower-left fallback.")

zero_angle_rad = np.arctan2(
    angle_zero_point.y - center_point.y,
    angle_zero_point.x - center_point.x
)

print("\nCentre point:", center_point)
print("Angle-zero point:", angle_zero_point)


# ----------------------------------
# 12. BUILD RADIAL PROFILE
# -------------------------------------

profile_df, rays_gdf, min_points_gdf, max_points_gdf = build_radial_profile(
    reshape_min,
    reshape_max,
    coastline,
    center_point,
    zero_angle_rad
)

profile_csv = PROFILE_FOLDER / "reshape_radial_profile.csv"

profile_df.to_csv(
    profile_csv,
    index=False
)

save_vector(rays_gdf, PROFILE_FOLDER / "diagnostic_radial_rays_every_10deg.shp")
save_vector(min_points_gdf, PROFILE_FOLDER / "diagnostic_rmin_points.shp")
save_vector(max_points_gdf, PROFILE_FOLDER / "diagnostic_rmax_points.shp")

print("\nPROFILE SUMMARY")
print(profile_df["valid"].value_counts(dropna=False))

if profile_df["valid"].sum() > 0:
    print(profile_df.loc[profile_df["valid"], "width_m"].describe())

print("Saved profile:", profile_csv)


# ------------------------------
# 13. TRANSFORM BASE LAYERS
# --------------------------------

deformed_min, deformed_max = make_strip_boundary_lines(profile_df)

deformed_coastline, coastline_skipped = transform_line_layer_by_vertices(
    coastline,
    center_point,
    zero_angle_rad,
    profile_df,
    "coastline",
    spacing_m=COASTLINE_DENSIFY_SPACING_M
)

save_both(deformed_min, "deformed_reshape_minimum_y0")
save_both(deformed_max, "deformed_reshape_maximum_y1000")
save_both(deformed_coastline, "deformed_coastline_by_vertices")


# -------------------------------------------
# 14. TRANSFORM POLYGON LAYERS BY VERTICES
# ------------------------------------------

summary_records = []

def add_summary(layer_name, source_count, transformed_count, fallback_count, failed_count):
    summary_records.append({
        "layer": layer_name,
        "source_features_or_parts": source_count,
        "transformed_features": transformed_count,
        "fallback_points": fallback_count,
        "failed_or_skipped": failed_count
    })


if TRANSFORM_FLOOD:
    deformed_flood, flood_fallback, flood_failed, flood_parts = transform_polygon_layer_by_vertices(
        flood,
        center_point,
        zero_angle_rad,
        profile_df,
        "flood_occurrence"
    )
else:
    deformed_flood = gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=None)
    flood_fallback = gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=None)
    flood_failed = 0
    flood_parts = 0

save_both(deformed_flood, "deformed_flood_polygons_by_vertices")
save_both(flood_fallback, "deformed_flood_fallback_points")
add_summary("flood_occurrence", flood_parts, len(deformed_flood), len(flood_fallback), flood_failed)


if TRANSFORM_AOI:
    deformed_aoi, aoi_fallback, aoi_failed, aoi_parts = transform_polygon_layer_by_vertices(
        aoi,
        center_point,
        zero_angle_rad,
        profile_df,
        "aoi_27_polygons"
    )
else:
    deformed_aoi = gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=None)
    aoi_fallback = gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=None)
    aoi_failed = 0
    aoi_parts = 0

# Force AOI labels 1 to 27 if no good field exists
aoi_label_field = get_first_existing_field(
    deformed_aoi,
    AOI_LABEL_CANDIDATES
)

if len(deformed_aoi) > 0:
    if aoi_label_field is None:
        deformed_aoi["AOI_label"] = [str(i + 1) for i in range(len(deformed_aoi))]
    else:
        deformed_aoi["AOI_label"] = deformed_aoi[aoi_label_field].astype(str)

if len(aoi_fallback) > 0:
    aoi_fallback["AOI_label"] = [str(i + 1) for i in range(len(aoi_fallback))]

save_both(deformed_aoi, "deformed_27_aoi_polygons_by_vertices")
save_both(aoi_fallback, "deformed_aoi_fallback_points")
add_summary("27_aoi_polygons", aoi_parts, len(deformed_aoi), len(aoi_fallback), aoi_failed)


if TRANSFORM_BUILDINGS:
    deformed_buildings, building_fallback, building_failed, building_parts = transform_polygon_layer_by_vertices(
        buildings,
        center_point,
        zero_angle_rad,
        profile_df,
        "buildings"
    )
else:
    deformed_buildings = gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=None)
    building_fallback = gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=None)
    building_failed = 0
    building_parts = 0

save_both(deformed_buildings, "deformed_building_polygons_by_vertices")
save_both(building_fallback, "deformed_building_fallback_points")
add_summary("buildings", building_parts, len(deformed_buildings), len(building_fallback), building_failed)


if TRANSFORM_POPULATION_GRID:
    deformed_population_grid, popgrid_fallback, popgrid_failed, popgrid_parts = transform_polygon_layer_by_vertices(
        population_grid,
        center_point,
        zero_angle_rad,
        profile_df,
        "population_grid"
    )
else:
    deformed_population_grid = gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=None)
    popgrid_fallback = gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=None)
    popgrid_failed = 0
    popgrid_parts = 0

save_both(deformed_population_grid, "deformed_population_grid_polygons_by_vertices")
save_both(popgrid_fallback, "deformed_population_grid_fallback_points")
add_summary("population_grid", popgrid_parts, len(deformed_population_grid), len(popgrid_fallback), popgrid_failed)


if TRANSFORM_WATERBODIES:
    deformed_waterbodies, waterbody_fallback, waterbody_failed, waterbody_parts = transform_polygon_layer_by_vertices(
        waterbodies,
        center_point,
        zero_angle_rad,
        profile_df,
        "waterbodies"
    )
else:
    deformed_waterbodies = gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=None)
    waterbody_fallback = gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=None)
    waterbody_failed = 0
    waterbody_parts = 0

save_both(deformed_waterbodies, "deformed_waterbody_polygons_by_vertices")
save_both(waterbody_fallback, "deformed_waterbody_fallback_points")
add_summary("waterbodies", waterbody_parts, len(deformed_waterbodies), len(waterbody_fallback), waterbody_failed)


# -------------------------------------------
# 15. TRANSFORM LINE LAYERS BY VERTICES
# ----------------------------------------

if TRANSFORM_ROADS:
    deformed_roads, roads_skipped = transform_line_layer_by_vertices(
        roads,
        center_point,
        zero_angle_rad,
        profile_df,
        "roads"
    )
else:
    deformed_roads = gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=None)
    roads_skipped = 0

save_both(deformed_roads, "deformed_roads_by_vertices")
add_summary("roads", len(roads), len(deformed_roads), 0, roads_skipped)


if TRANSFORM_RAILWAYS:
    deformed_railways, railways_skipped = transform_line_layer_by_vertices(
        railways,
        center_point,
        zero_angle_rad,
        profile_df,
        "railways"
    )
else:
    deformed_railways = gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=None)
    railways_skipped = 0

save_both(deformed_railways, "deformed_railways_by_vertices")
add_summary("railways", len(railways), len(deformed_railways), 0, railways_skipped)


if TRANSFORM_WATER_LINES:
    deformed_water_lines, water_lines_skipped = transform_line_layer_by_vertices(
        water_lines,
        center_point,
        zero_angle_rad,
        profile_df,
        "water_lines"
    )
else:
    deformed_water_lines = gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=None)
    water_lines_skipped = 0

save_both(deformed_water_lines, "deformed_water_lines_by_vertices")
add_summary("water_lines", len(water_lines), len(deformed_water_lines), 0, water_lines_skipped)


# -----------------------------------------
# 16. TRANSFORM POINT LAYERS DIRECTLY
# -----------------------------------------

if TRANSFORM_POPULATION_POINTS:
    deformed_population_points, poppoints_skipped = transform_point_layer_direct(
        population_points,
        center_point,
        zero_angle_rad,
        profile_df,
        "population_points"
    )
else:
    deformed_population_points = gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=None)
    poppoints_skipped = 0

save_both(deformed_population_points, "deformed_population_points")
add_summary("population_points", len(population_points), len(deformed_population_points), 0, poppoints_skipped)


if TRANSFORM_TRANSPORT:
    deformed_transport, transport_skipped = transform_point_layer_direct(
        transport,
        center_point,
        zero_angle_rad,
        profile_df,
        "transport_facilities"
    )
else:
    deformed_transport = gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=None)
    transport_skipped = 0

save_both(deformed_transport, "deformed_transport_facilities_points")
add_summary("transport_facilities", len(transport), len(deformed_transport), 0, transport_skipped)


if TRANSFORM_TRAFFIC:
    deformed_traffic, traffic_skipped = transform_point_layer_direct(
        traffic,
        center_point,
        zero_angle_rad,
        profile_df,
        "traffic_points"
    )
else:
    deformed_traffic = gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=None)
    traffic_skipped = 0

save_both(deformed_traffic, "deformed_traffic_points")
add_summary("traffic_points", len(traffic), len(deformed_traffic), 0, traffic_skipped)


# ------------------------------------------
# 17. POPULATION BY VERTICAL THETA SLICES
# ---------------------------------------------

population_slice_df = calculate_population_by_theta_slice(
    deformed_population_grid
)

population_slice_csv = POP_FOLDER / "population_by_theta_slice.csv"

population_slice_df.to_csv(
    population_slice_csv,
    index=False
)

print("Saved population slice CSV:")
print(population_slice_csv)


# ------------------------------------------------------------
# 18. SAVE SUMMARY
# --------------------------------------------------------------

summary_df = pd.DataFrame(summary_records)

summary_csv = SUMMARY_FOLDER / "layer_transformation_summary.csv"

summary_df.to_csv(
    summary_csv,
    index=False
)

print("Saved summary CSV:")
print(summary_csv)

print("\nTRANSFORMATION SUMMARY")
print(summary_df)


# ----------------------------------------------------
# 19. PLOT 1: WIDTH PROFILE
# ---------------------------------------------------

fig, ax = plt.subplots(figsize=(16, 4))

valid_profile = profile_df[profile_df["valid"] == True].copy()
invalid_profile = profile_df[profile_df["valid"] == False].copy()

if len(valid_profile) > 0:
    ax.plot(
        valid_profile["theta_deg"],
        valid_profile["width_m"],
        color="black",
        linewidth=1.2,
        label="Raw corridor width"
    )

    ax.plot(
        valid_profile["theta_deg"],
        valid_profile["width_used_m"],
        color="red",
        linewidth=1.2,
        alpha=0.8,
        label="Smoothed width used"
    )

if len(invalid_profile) > 0:
    ax.scatter(
        invalid_profile["theta_deg"],
        np.zeros(len(invalid_profile)),
        color="red",
        s=12,
        label="Invalid rays"
    )

ax.set_title("Reshaped corridor width profile")
ax.set_xlabel("Clockwise theta, degrees")
ax.set_ylabel("Width, metres")
ax.set_xlim(0, 360)
ax.grid(True, linestyle="--", alpha=0.25)
ax.legend(loc="best")

plt.tight_layout()

plot_width = PLOT_FOLDER / "plot_01_width_profile.png"

plt.savefig(plot_width, dpi=300, bbox_inches="tight")
plt.show()

print("Saved:", plot_width)


# ---------------------------------------------------
# 20. PLOT 2: FLOOD POLYGONS ONLY
# -------------------------------------------------

fig, ax = plt.subplots(figsize=(18, 4.5))

flood_field = get_first_existing_field(
    deformed_flood,
    FLOOD_VALUE_CANDIDATES
)

flood_sm = draw_polygons(
    ax,
    deformed_flood,
    facecolor="lightblue",
    edgecolor="none",
    linewidth=0.10,
    alpha=0.70,
    label="Flood polygons",
    value_field=flood_field,
    cmap_name="Blues"
)

draw_lines(
    ax,
    deformed_coastline,
    color="purple",
    linewidth=0.8,
    alpha=0.50,
    label="Transformed coastline"
)

draw_lines(
    ax,
    deformed_min,
    color="blue",
    linewidth=2.2,
    label="Minimum boundary, y=0"
)

draw_lines(
    ax,
    deformed_max,
    color="red",
    linewidth=2.2,
    label="Maximum boundary, y=1000"
)

if flood_sm is not None:
    cbar = plt.colorbar(flood_sm, ax=ax, shrink=0.85, pad=0.01)
    cbar.set_label("Flooded scenarios")

setup_strip_axis(
    ax,
    "Reshaped polar-radial deformation strip: flood polygons"
)

clean_legend(ax, ncol=4, y=-0.35)

plt.tight_layout()

plot_flood = PLOT_FOLDER / "plot_02_flood_polygons_only.png"

plt.savefig(plot_flood, dpi=300, bbox_inches="tight")
plt.savefig(PLOT_FOLDER / "plot_02_flood_polygons_only.pdf", bbox_inches="tight")
plt.show()

print("Saved:", plot_flood)


# ---------------------------------------------------------
# 21. PLOT 3: FLOOD + 27 AOIs
# ----------------------------------------------------------

fig, ax = plt.subplots(figsize=(18, 4.8))

flood_sm = draw_polygons(
    ax,
    deformed_flood,
    facecolor="lightblue",
    edgecolor="none",
    linewidth=0.10,
    alpha=0.65,
    label="Flood polygons",
    value_field=flood_field,
    cmap_name="Blues"
)

draw_polygons(
    ax,
    deformed_aoi,
    facecolor="none",
    edgecolor="black",
    linewidth=1.0,
    alpha=0.95,
    label="27 AOI polygons"
)

# AOI labels
if len(deformed_aoi) > 0:
    for idx, row in deformed_aoi.iterrows():
        try:
            p = row.geometry.representative_point()

            if "AOI_label" in deformed_aoi.columns:
                label = str(row["AOI_label"])
            else:
                label = str(idx + 1)

            ax.text(
                p.x,
                p.y + 20,
                label,
                fontsize=8,
                ha="center",
                va="bottom",
                color="black",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.60, pad=0.3)
            )
        except Exception:
            pass

draw_lines(
    ax,
    deformed_coastline,
    color="purple",
    linewidth=0.8,
    alpha=0.45,
    label="Transformed coastline"
)

draw_lines(
    ax,
    deformed_min,
    color="blue",
    linewidth=2.2,
    label="Minimum boundary, y=0"
)

draw_lines(
    ax,
    deformed_max,
    color="red",
    linewidth=2.2,
    label="Maximum boundary, y=1000"
)

if flood_sm is not None:
    cbar = plt.colorbar(flood_sm, ax=ax, shrink=0.85, pad=0.01)
    cbar.set_label("Flooded scenarios")

setup_strip_axis(
    ax,
    "Reshaped polar-radial deformation strip: flood polygons and 27 AOIs"
)

clean_legend(ax, ncol=4, y=-0.36)

plt.tight_layout()

plot_flood_aoi = PLOT_FOLDER / "plot_03_flood_plus_27_aoi.png"

plt.savefig(plot_flood_aoi, dpi=300, bbox_inches="tight")
plt.savefig(PLOT_FOLDER / "plot_03_flood_plus_27_aoi.pdf", bbox_inches="tight")
plt.show()

print("Saved:", plot_flood_aoi)


# ----------------------------------------------------------
# 22. PLOT 4: POPULATION BAR + FLOOD + AOI
# -----------------------------------------------------------

fig, (ax_pop, ax_map) = plt.subplots(
    2,
    1,
    figsize=(18, 7),
    gridspec_kw={"height_ratios": [1.2, 4]},
    sharex=True
)

# Population bars
if len(population_slice_df) > 0:
    ax_pop.bar(
        population_slice_df["theta_mid"],
        population_slice_df["population"],
        width=THETA_SLICE_DEG * 0.9,
        align="center",
        alpha=0.75,
        label="Population"
    )

ax_pop.set_ylabel("Population")
ax_pop.set_title("Population by vertical theta slice")
ax_pop.grid(True, linestyle="--", alpha=0.20)

# Map
flood_sm = draw_polygons(
    ax_map,
    deformed_flood,
    facecolor="lightblue",
    edgecolor="none",
    linewidth=0.10,
    alpha=0.65,
    label="Flood polygons",
    value_field=flood_field,
    cmap_name="Blues"
)

draw_polygons(
    ax_map,
    deformed_aoi,
    facecolor="none",
    edgecolor="black",
    linewidth=1.0,
    alpha=0.95,
    label="27 AOI polygons"
)

draw_lines(
    ax_map,
    deformed_coastline,
    color="purple",
    linewidth=0.8,
    alpha=0.45,
    label="Transformed coastline"
)

draw_lines(
    ax_map,
    deformed_min,
    color="blue",
    linewidth=2.2,
    label="Minimum boundary, y=0"
)

draw_lines(
    ax_map,
    deformed_max,
    color="red",
    linewidth=2.2,
    label="Maximum boundary, y=1000"
)

if flood_sm is not None:
    cbar = plt.colorbar(flood_sm, ax=ax_map, shrink=0.85, pad=0.01)
    cbar.set_label("Flooded scenarios")

setup_strip_axis(
    ax_map,
    "Flood polygons, 27 AOIs and coastline in reshaped polar-radial strip"
)

ax_map.set_xlim(0, 360)
clean_legend(ax_map, ncol=4, y=-0.36)

plt.tight_layout()

plot_pop = PLOT_FOLDER / "plot_04_population_bar_plus_flood_aoi.png"

plt.savefig(plot_pop, dpi=300, bbox_inches="tight")
plt.savefig(PLOT_FOLDER / "plot_04_population_bar_plus_flood_aoi.pdf", bbox_inches="tight")
plt.show()

print("Saved:", plot_pop)


# ============================================================
# 23. PLOT 5: INFRASTRUCTURE CONTEXT
# ============================================================

fig, ax = plt.subplots(figsize=(18, 4.8))

draw_polygons(
    ax,
    deformed_flood,
    facecolor="lightblue",
    edgecolor="none",
    linewidth=0.10,
    alpha=0.35,
    label="Flood polygons"
)

draw_lines(
    ax,
    deformed_roads,
    color="grey",
    linewidth=0.45,
    alpha=0.35,
    label="Roads"
)

draw_lines(
    ax,
    deformed_railways,
    color="black",
    linewidth=1.2,
    alpha=0.80,
    label="Railways"
)

draw_points(
    ax,
    deformed_transport,
    color="orange",
    size=28,
    label="Transport facilities",
    alpha=0.75,
    marker="^"
)

draw_points(
    ax,
    deformed_traffic,
    color="brown",
    size=18,
    label="Traffic points",
    alpha=0.65,
    marker="x"
)

draw_lines(
    ax,
    deformed_coastline,
    color="purple",
    linewidth=0.8,
    alpha=0.45,
    label="Transformed coastline"
)

draw_lines(
    ax,
    deformed_min,
    color="blue",
    linewidth=2.2,
    label="Minimum boundary, y=0"
)

draw_lines(
    ax,
    deformed_max,
    color="red",
    linewidth=2.2,
    label="Maximum boundary, y=1000"
)

setup_strip_axis(
    ax,
    "Reshaped polar-radial deformation strip: infrastructure context"
)

clean_legend(ax, ncol=5, y=-0.36)

plt.tight_layout()

plot_infra = PLOT_FOLDER / "plot_05_infrastructure_context.png"

plt.savefig(plot_infra, dpi=300, bbox_inches="tight")
plt.savefig(PLOT_FOLDER / "plot_05_infrastructure_context.pdf", bbox_inches="tight")
plt.show()

print("Saved:", plot_infra)


# ============================================================
# 24. PLOT 6: COMBINED OVERVIEW
# ============================================================

fig, ax = plt.subplots(figsize=(20, 5.5))

draw_polygons(
    ax,
    deformed_waterbodies,
    facecolor="lightcyan",
    edgecolor="none",
    linewidth=0.10,
    alpha=0.25,
    label="Waterbodies"
)

draw_polygons(
    ax,
    deformed_flood,
    facecolor="lightblue",
    edgecolor="none",
    linewidth=0.10,
    alpha=0.45,
    label="Flood polygons",
    value_field=flood_field,
    cmap_name="Blues"
)

building_plot = deformed_buildings

if len(building_plot) > MAX_BUILDINGS_TO_DRAW:
    building_plot = building_plot.sample(
        n=MAX_BUILDINGS_TO_DRAW,
        random_state=42
    ).reset_index(drop=True)

draw_polygons(
    ax,
    building_plot,
    facecolor="darkorange",
    edgecolor="none",
    linewidth=0.05,
    alpha=0.14,
    label="Buildings"
)

draw_lines(
    ax,
    deformed_roads,
    color="grey",
    linewidth=0.35,
    alpha=0.25,
    label="Roads"
)

draw_lines(
    ax,
    deformed_railways,
    color="black",
    linewidth=1.0,
    alpha=0.75,
    label="Railways"
)

draw_polygons(
    ax,
    deformed_aoi,
    facecolor="none",
    edgecolor="black",
    linewidth=0.9,
    alpha=0.95,
    label="27 AOI polygons"
)

draw_lines(
    ax,
    deformed_coastline,
    color="purple",
    linewidth=0.8,
    alpha=0.45,
    label="Transformed coastline"
)

draw_lines(
    ax,
    deformed_min,
    color="blue",
    linewidth=2.2,
    label="Minimum boundary, y=0"
)

draw_lines(
    ax,
    deformed_max,
    color="red",
    linewidth=2.2,
    label="Maximum boundary, y=1000"
)

setup_strip_axis(
    ax,
    "Combined overview: flood, AOIs, buildings, infrastructure and coastline"
)

clean_legend(ax, ncol=6, y=-0.36)

plt.tight_layout()

plot_combined = PLOT_FOLDER / "plot_06_combined_overview.png"

plt.savefig(plot_combined, dpi=300, bbox_inches="tight")
plt.savefig(PLOT_FOLDER / "plot_06_combined_overview.pdf", bbox_inches="tight")
plt.show()

print("Saved:", plot_combined)


# ============================================================
# 25. FINAL MESSAGE
# ============================================================

print("\nDONE.")
print("Full reshaped polar-radial deformation pipeline completed.")
print("\nMain output folder:")
print(RUN_FOLDER)

print("\nImportant plots:")
print(plot_flood)
print(plot_flood_aoi)
print(plot_pop)
print(plot_infra)
print(plot_combined)

print("\nImportant transformed layers folder:")
print(LAYER_FOLDER)