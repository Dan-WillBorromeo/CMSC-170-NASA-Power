#K-Means Clustering
#
# Author: Dayahan, Rey Russel M.
# Disclosures:
#   - AI Tools were used to helpwith plotting an interactive map, nevertheless most of the code were adapted from Dayahan_exer071.py
# Adapted from Dayahan_exer07a.py

import os
import subprocess

import folium
import webbrowser
import json
import math
import random
from pathlib import Path

#plotting with pandas
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx


REPO_ROOT = Path(__file__).resolve().parent.parent  # Resolve project root directory

FEATURES = [  # 12 climate profile features
    "ghi", "ghi_iav", "ghi_season", "ghi_low",
    "t", "trange", "rh", "ps",
    "ws", "ws99", "rain", "wet"
]

FEATURE_UNITS = {  # Physical measurement units for each climate profile feature
    "ghi": "kWh/m2/day",
    "ghi_iav": "kWh/m2/day",
    "ghi_season": "ratio",
    "ghi_low": "fraction",
    "t": "deg C",
    "trange": "deg C",
    "rh": "%",
    "ps": "kPa",
    "ws": "m/s",
    "ws99": "m/s",
    "rain": "mm/yr",
    "wet": "fraction"
}


#CLIMATE PROFILE FEATURE EXTRACTION & SCALING


def read_power_raw(filepath):  # Read NASA POWER CSV rows (YEAR < 2026)
    file_text = Path(filepath).read_text()
    lines = file_text.splitlines()

    header_row_index = 0
    for line_index in range(len(lines)):
        if lines[line_index].startswith("-END HEADER-"):
            header_row_index = line_index + 1  # Column names immediately follow -END HEADER-
            break

    column_names = lines[header_row_index].split(",")
    valid_rows = []

    for line_index in range(header_row_index + 1, len(lines)):
        stripped_line = lines[line_index].strip()
        if not stripped_line:
            continue  # Skip empty lines

        string_values = stripped_line.split(",")
        row_dict = {}
        has_missing_value = False

        for col_index in range(len(column_names)):
            numeric_value = float(string_values[col_index])
            if numeric_value == -999.0:
                has_missing_value = True  # Flag NASA POWER missing data marker
            row_dict[column_names[col_index]] = numeric_value

        if row_dict["YEAR"] < 2026 and not has_missing_value:
            valid_rows.append(row_dict)  # Retain complete records prior to 2026

    return valid_rows


def compute_mean(values_list):  # Compute arithmetic mean of a numeric list
    total_sum = 0.0
    for value in values_list:
        total_sum += value
    return total_sum / len(values_list)


def compute_std(values_list):  # Compute population standard deviation
    mean_value = compute_mean(values_list)
    squared_diff_sum = 0.0

    for value in values_list:
        diff = value - mean_value
        squared_diff_sum += diff * diff

    variance = squared_diff_sum / len(values_list)
    return math.sqrt(variance)


def compute_quantile(values_list, quantile_fraction):  # Compute empirical quantile
    sorted_values = sorted(values_list)
    raw_index = int(quantile_fraction * len(sorted_values))
    clamped_index = min(raw_index, len(sorted_values) - 1)
    return sorted_values[clamped_index]


def extract_climate_profile(filepath):  # Compute 12-feature climate summary vector for one grid cell CSV
    daily_rows = read_power_raw(filepath)
    total_days = len(daily_rows)

    irradiance_values = []
    temperature_values = []
    temp_range_values = []
    humidity_values = []
    pressure_values = []
    wind_speed_values = []
    low_irradiance_days = 0
    wet_days_count = 0

    yearly_irradiance_map = {}
    monthly_irradiance_map = {}
    yearly_rainfall_map = {}

    for row in daily_rows:
        ghi_val = row["ALLSKY_SFC_SW_DWN"]
        year_val = row["YEAR"]
        month_val = row["MO"]
        rain_val = row["PRECTOTCORR"]

        irradiance_values.append(ghi_val)
        temperature_values.append(row["T2M"])
        temp_range_values.append(row["T2M_RANGE"])
        humidity_values.append(row["RH2M"])
        pressure_values.append(row["PS"])
        wind_speed_values.append(row["WS10M"])

        if ghi_val < 2.0:
            low_irradiance_days += 1  # Count heavily overcast days
        if rain_val > 1.0:
            wet_days_count += 1  # Count rainy days exceeding 1 mm

        if year_val not in yearly_irradiance_map:
            yearly_irradiance_map[year_val] = []
        yearly_irradiance_map[year_val].append(ghi_val)

        if month_val not in monthly_irradiance_map:
            monthly_irradiance_map[month_val] = []
        monthly_irradiance_map[month_val].append(ghi_val)

        if year_val not in yearly_rainfall_map:
            yearly_rainfall_map[year_val] = 0.0
        yearly_rainfall_map[year_val] += rain_val

    yearly_ghi_means = []
    for year_key in yearly_irradiance_map:
        yearly_ghi_means.append(compute_mean(yearly_irradiance_map[year_key]))

    monthly_ghi_means = []
    for month_key in monthly_irradiance_map:
        monthly_ghi_means.append(compute_mean(monthly_irradiance_map[month_key]))

    yearly_rain_totals = []
    for year_key in yearly_rainfall_map:
        yearly_rain_totals.append(yearly_rainfall_map[year_key])

    ghi_mean = compute_mean(irradiance_values)
    ghi_iav = compute_std(yearly_ghi_means)
    ghi_season = (max(monthly_ghi_means) - min(monthly_ghi_means)) / compute_mean(monthly_ghi_means)
    ghi_low = low_irradiance_days / total_days
    temp_mean = compute_mean(temperature_values)
    temp_range_mean = compute_mean(temp_range_values)
    rh_mean = compute_mean(humidity_values)
    ps_mean = compute_mean(pressure_values)
    ws_mean = compute_mean(wind_speed_values)
    ws_99 = compute_quantile(wind_speed_values, 0.99)
    rain_mean = compute_mean(yearly_rain_totals)
    wet_fraction = wet_days_count / total_days

    return [
        ghi_mean, ghi_iav, ghi_season, ghi_low,
        temp_mean, temp_range_mean, rh_mean, ps_mean,
        ws_mean, ws_99, rain_mean, wet_fraction
    ]


def load_climate_folder(folder_path):  # Load all grid cell CSVs sorted by filename, returning (cell_key, features)
    csv_files = sorted(Path(folder_path).glob("*.csv"))
    loaded_samples = []

    for csv_path in csv_files:
        cell_key = csv_path.stem
        climate_vector = extract_climate_profile(csv_path)
        loaded_samples.append((cell_key, climate_vector))

    return loaded_samples


def compute_feature_bounds(feature_vectors):  # Compute column-wise min and max bounds across dataset
    number_of_dimensions = len(feature_vectors[0])
    lower_bounds = []
    upper_bounds = []

    for dimension_index in range(number_of_dimensions):
        min_val = float("inf")
        max_val = float("-inf")

        for vector in feature_vectors:
            val = vector[dimension_index]
            if val < min_val:
                min_val = val
            if val > max_val:
                max_val = val

        lower_bounds.append(min_val)
        upper_bounds.append(max_val)

    return lower_bounds, upper_bounds


def scale_feature_vector(feature_vector, lower_bounds, upper_bounds):  # Scale features to [0.1, 0.9]
    scaled_vector = []

    for dimension_index in range(len(feature_vector)):
        raw_val = feature_vector[dimension_index]
        low_val = lower_bounds[dimension_index]
        high_val = upper_bounds[dimension_index]

        if high_val > low_val:
            scaled_val = 0.1 + 0.8 * (raw_val - low_val) / (high_val - low_val)
            clamped_val = min(1.0, max(0.0, scaled_val))
            scaled_vector.append(clamped_val)
        else:
            scaled_vector.append(0.5)  # Default midpoint if feature has zero variance

    return scaled_vector



# DISTANCE & CENTROID INITIALIZATION


def euclidean_distance(point_a, point_b):  # Compute N-dimensional Euclidean distance
    summation_of_squared_diffs = 0.0
    number_of_dimensions = len(point_a)

    for dimension_index in range(number_of_dimensions):
        coordinate_difference = point_a[dimension_index] - point_b[dimension_index]
        squared_difference = coordinate_difference * coordinate_difference
        summation_of_squared_diffs += squared_difference

    distance_value = math.sqrt(summation_of_squared_diffs)
    return distance_value


def initialize_centroids(feature_vectors, number_of_clusters, random_generator):  # Select random initial centroids
    sampled_vectors = random_generator.sample(feature_vectors, number_of_clusters)
    initial_centroids_list = []

    for sampled_vector in sampled_vectors:
        copied_vector = []
        for feature_value in sampled_vector:
            copied_vector.append(feature_value)  # Copy element-by-element to prevent mutation
        initial_centroids_list.append(copied_vector)

    return initial_centroids_list



# N-DIMENSIONAL K-MEANS CLUSTERING LOOP


def run_kmeans(feature_vectors, number_of_clusters, max_iterations=100, random_seed=170):  # Execute K-Means clustering
    random_generator = random.Random(random_seed)  # Deterministic RNG
    total_points = len(feature_vectors)
    number_of_dimensions = len(feature_vectors[0])

    initial_centroids = initialize_centroids(feature_vectors, number_of_clusters, random_generator)
    initial_centroids_snapshot = []
    for centroid_vector in initial_centroids:
        snapshot_vector = []
        for feature_value in centroid_vector:
            snapshot_vector.append(feature_value)
        initial_centroids_snapshot.append(snapshot_vector)

    current_centroids = initial_centroids
    iteration_count = 0
    cluster_assignments = []

    for iteration_index in range(max_iterations):
        iteration_count += 1
        cluster_assignments = []

        for point_vector in feature_vectors:  # Step 1: Assign each cell to nearest centroid
            minimum_distance = float("inf")
            assigned_cluster_index = -1

            for centroid_index in range(number_of_clusters):
                centroid_vector = current_centroids[centroid_index]
                computed_distance = euclidean_distance(point_vector, centroid_vector)

                if computed_distance < minimum_distance:
                    minimum_distance = computed_distance
                    assigned_cluster_index = centroid_index

            cluster_assignments.append(assigned_cluster_index)

        updated_centroids = []
        for centroid_index in range(number_of_clusters):  # Step 2: Recompute mean vector per cluster
            dimension_sums = []
            for dimension_index in range(number_of_dimensions):
                dimension_sums.append(0.0)
            assigned_point_count = 0

            for point_index in range(total_points):
                if cluster_assignments[point_index] == centroid_index:
                    for dimension_index in range(number_of_dimensions):
                        dimension_sums[dimension_index] += feature_vectors[point_index][dimension_index]
                    assigned_point_count += 1

            if assigned_point_count > 0:
                mean_centroid_vector = []
                for dimension_index in range(number_of_dimensions):
                    dimension_mean = dimension_sums[dimension_index] / assigned_point_count
                    mean_centroid_vector.append(dimension_mean)
                updated_centroids.append(mean_centroid_vector)
            else:
                fallback_vector = random_generator.choice(feature_vectors)  # Reinitialize empty cluster
                copied_fallback = []
                for feature_value in fallback_vector:
                    copied_fallback.append(feature_value)
                updated_centroids.append(copied_fallback)

        centroids_converged = True  # Step 3: Check centroid convergence
        for centroid_index in range(number_of_clusters):
            for dimension_index in range(number_of_dimensions):
                coordinate_shift = abs(
                    current_centroids[centroid_index][dimension_index]
                    - updated_centroids[centroid_index][dimension_index]
                )
                if coordinate_shift > 1e-9:
                    centroids_converged = False
                    break
            if not centroids_converged:
                break

        current_centroids = updated_centroids
        if centroids_converged:
            break

    return initial_centroids_snapshot, current_centroids, iteration_count, cluster_assignments



# CLUSTER QUALITY METRICS (WCSS & Silhouette Score)  to determine best k


def compute_wcss(feature_vectors, centroids, cluster_assignments):  # Compute Within-Cluster Sum of Squares
    total_wcss = 0.0

    for point_index in range(len(feature_vectors)):
        assigned_cluster = cluster_assignments[point_index]
        distance_to_centroid = euclidean_distance(feature_vectors[point_index], centroids[assigned_cluster])
        total_wcss += distance_to_centroid * distance_to_centroid

    return total_wcss


def compute_silhouette_score(feature_vectors, cluster_assignments, number_of_clusters):  # Compute mean Silhouette Score
    total_points = len(feature_vectors)
    silhouette_scores_list = []

    for point_i in range(total_points):
        own_cluster_id = cluster_assignments[point_i]
        intra_cluster_distances = []
        inter_cluster_distances_map = {}

        for point_j in range(total_points):
            if point_i == point_j:
                continue
            pairwise_distance = euclidean_distance(feature_vectors[point_i], feature_vectors[point_j])
            other_cluster_id = cluster_assignments[point_j]

            if other_cluster_id == own_cluster_id:
                intra_cluster_distances.append(pairwise_distance)
            else:
                if other_cluster_id not in inter_cluster_distances_map:
                    inter_cluster_distances_map[other_cluster_id] = []
                inter_cluster_distances_map[other_cluster_id].append(pairwise_distance)

        if len(intra_cluster_distances) == 0 or len(inter_cluster_distances_map) == 0:
            silhouette_scores_list.append(0.0)
            continue

        intra_sum = 0.0
        for distance_value in intra_cluster_distances:
            intra_sum += distance_value
        mean_intra_distance = intra_sum / len(intra_cluster_distances)

        nearest_inter_distance = float("inf")
        for other_cluster_id, distance_list in inter_cluster_distances_map.items():
            inter_sum = 0.0
            for distance_value in distance_list:
                inter_sum += distance_value
            mean_inter_distance = inter_sum / len(distance_list)
            if mean_inter_distance < nearest_inter_distance:
                nearest_inter_distance = mean_inter_distance

        denominator = max(mean_intra_distance, nearest_inter_distance)
        point_silhouette = (nearest_inter_distance - mean_intra_distance) / denominator if denominator > 0 else 0.0
        silhouette_scores_list.append(point_silhouette)

    silhouette_total = 0.0
    for score_value in silhouette_scores_list:
        silhouette_total += score_value

    return silhouette_total / len(silhouette_scores_list)



# CENTROID FEATURE-PROFILE UNSCALING & PRINTOUT


def unscale_centroid_vector(scaled_centroid, lower_bounds, upper_bounds):  # Invert [0.1, 0.9] scaling to physical units
    physical_centroid = []

    for dimension_index in range(len(scaled_centroid)):
        scaled_value = scaled_centroid[dimension_index]
        lower_val = lower_bounds[dimension_index]
        upper_val = upper_bounds[dimension_index]
        feature_span = upper_val - lower_val

        if feature_span > 0.0:
            normalized_fraction = (scaled_value - 0.1) / 0.8
            physical_value = lower_val + normalized_fraction * feature_span
        else:
            physical_value = lower_val

        physical_centroid.append(physical_value)

    return physical_centroid


def print_centroid_feature_profiles(
    scaled_centroids,
    lower_bounds,
    upper_bounds,
    cluster_assignments,
    ground_truth_labels
):  # Print physical and scaled feature profiles per cluster
    number_of_clusters = len(scaled_centroids)
    physical_centroids = []

    for centroid_index in range(number_of_clusters):
        unscaled_vector = unscale_centroid_vector(
            scaled_centroids[centroid_index], lower_bounds, upper_bounds
        )
        physical_centroids.append(unscaled_vector)

    print("\n" + "=" * 96)
    print(f"LEARNED CLUSTER FEATURE PROFILES (k = {number_of_clusters}) — PHYSICAL UNITS [SCALED 0.1–0.9]")
    print("=" * 96)

    for cluster_id in range(number_of_clusters):  # Print SF vs. NSF composition header per cluster
        solar_farm_count = 0
        non_solar_farm_count = 0
        for sample_index in range(len(cluster_assignments)):
            if cluster_assignments[sample_index] == cluster_id:
                if ground_truth_labels[sample_index] == 1:
                    solar_farm_count += 1
                else:
                    non_solar_farm_count += 1
        cluster_total = solar_farm_count + non_solar_farm_count
        solar_farm_percentage = (solar_farm_count / cluster_total * 100.0) if cluster_total > 0 else 0.0
        print(
            f"Cluster {cluster_id} Summary : {cluster_total:>3} cells "
            f"(SF: {solar_farm_count:>2}, NSF: {non_solar_farm_count:>3} | {solar_farm_percentage:.1f}% SF)"
        )

    print("-" * 96)
    header_line = f"{'Feature':<12} | {'Unit':<12}"
    for cluster_id in range(number_of_clusters):
        header_line += f" | {'Cluster ' + str(cluster_id):<21}"
    print(header_line)
    print("-" * 96)

    for feature_index in range(len(FEATURES)):  # Print row-by-row physical value and scaled value
        feature_name = FEATURES[feature_index]
        unit_label = FEATURE_UNITS[feature_name]
        row_line = f"{feature_name:<12} | {unit_label:<12}"

        for cluster_id in range(number_of_clusters):
            phys_val = physical_centroids[cluster_id][feature_index]
            scaled_val = scaled_centroids[cluster_id][feature_index]
            formatted_cell = f"{phys_val:>8.3f} [{scaled_val:.2f}]"
            row_line += f" | {formatted_cell:<21}"

        print(row_line)

    print("=" * 96)
    return physical_centroids



# SPATIAL MAP VISUALIZATION (Adapted from geoplot in Dayahan_exer07a.py)


def geoplot(clustered_records, output_image_path, number_of_clusters):  # Plot spatial clusters over Philippine basemap
    dataframe_rows = []

    for record in clustered_records:
        cell_key = record[0]
        cluster_label = record[1]
        actual_label = record[2]

        coordinate_parts = cell_key.split("_")  # Parse 'lat_lon' filename key (e.g., '6.500_125.000')
        latitude_value = float(coordinate_parts[0])
        longitude_value = float(coordinate_parts[1])
        site_category = "Solar Farm (SF)" if actual_label == 1 else "Non-Solar Farm (NSF)"

        dataframe_rows.append({
            "cell_key": cell_key,
            "classification": cluster_label,
            "site_type": site_category,
            "longitude": longitude_value,
            "latitude": latitude_value
        })

    spatial_df = pd.DataFrame(dataframe_rows)

    geo_df = gpd.GeoDataFrame(  # Create WGS84 GeoDataFrame
        spatial_df,
        geometry=gpd.points_from_xy(spatial_df["longitude"], spatial_df["latitude"]),
        crs="EPSG:4326"
    )

    geo_df_mercator = geo_df.to_crs("EPSG:3857")  # Reproject to Web Mercator for contextily basemap

    figure, axis = plt.subplots(figsize=(9, 11))
    color_map = plt.get_cmap("tab10")

    for cluster_id in range(number_of_clusters):
        cluster_color = color_map(cluster_id % 10)
        cluster_subset = geo_df_mercator[geo_df_mercator["classification"] == cluster_id]

        nsf_subset = cluster_subset[cluster_subset["site_type"] == "Non-Solar Farm (NSF)"]
        if len(nsf_subset) > 0:
            nsf_subset.plot(
                ax=axis,
                color=cluster_color,
                marker="o",
                markersize=36,
                alpha=0.75,
                edgecolor="black",
                linewidth=0.4,
                label=f"Cluster {cluster_id} - NSF ({len(nsf_subset)})"
            )

        sf_subset = cluster_subset[cluster_subset["site_type"] == "Solar Farm (SF)"]
        if len(sf_subset) > 0:
            sf_subset.plot(
                ax=axis,
                color=cluster_color,
                marker="^",
                markersize=85,
                alpha=0.95,
                edgecolor="black",
                linewidth=0.9,
                label=f"Cluster {cluster_id} - SF ({len(sf_subset)})"
            )

    try:
        ctx.add_basemap(axis, source=ctx.providers.CartoDB.Positron)  # Add clean basemap tiles
    except Exception as basemap_error:
        print(f"Warning: Basemap tiles unavailable ({basemap_error}); rendering vector grid coordinates.")
        axis.grid(True, linestyle="--", alpha=0.4)

    axis.set_title(f"K-Means Climate Clustering (k = {number_of_clusters}): Philippine Grid Cells (SF vs. NSF)")
    axis.set_xlabel("Easting (EPSG:3857)")
    axis.set_ylabel("Northing (EPSG:3857)")
    axis.legend(loc="upper right", fontsize=8, framealpha=0.9)

    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_image_path, dpi=200)
    print(f"Saved spatial geoplot     : {output_image_path}")
    plt.show()

def interactive_geoplot(
    clustered_records,
    raw_feature_vectors,
    output_html_path,
    number_of_clusters,
    open_in_browser=True
):  # Build interactive Leaflet map of Philippine grid cells
    philippines_center_coords = [12.8797, 121.7740]  # Geographic center of the Philippines
    interactive_map = folium.Map(
        location=philippines_center_coords,
        zoom_start=6,
        tiles="OpenStreetMap"
    )

    folium.TileLayer(  # Add Esri World Imagery satellite layer option
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Satellite Imagery",
        overlay=False,
        control=True
    ).add_to(interactive_map)

    cluster_palette = [  # Distinct hex colors per cluster ID
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ]

    cluster_layer_groups = []  # Create toggleable map layers for each cluster
    for cluster_id in range(number_of_clusters):
        layer_group = folium.FeatureGroup(name=f"Cluster {cluster_id}", show=True)
        cluster_layer_groups.append(layer_group)

    for sample_index in range(len(clustered_records)):
        cell_key = clustered_records[sample_index][0]
        cluster_label = clustered_records[sample_index][1]
        actual_label = clustered_records[sample_index][2]
        raw_vector = raw_feature_vectors[sample_index]

        coordinate_parts = cell_key.split("_")  # Parse 'lat_lon' filename key
        latitude_value = float(coordinate_parts[0])
        longitude_value = float(coordinate_parts[1])

        marker_color = cluster_palette[cluster_label % len(cluster_palette)]
        site_status = "Existing Solar Farm (SF)" if actual_label == 1 else "Non-Solar Farm (NSF)"

        popup_lines = [  # Build HTML popup table with key climate metrics
            f"<b>Grid Cell:</b> {cell_key}<br>",
            f"<b>Assigned Cluster:</b> Cluster {cluster_label}<br>",
            f"<b>Status:</b> {site_status}<br><hr style='margin:4px 0;'>",
            f"<b>GHI Mean:</b> {raw_vector[0]:.3f} kWh/m²/day<br>",
            f"<b>Temp Mean:</b> {raw_vector[4]:.2f} °C (Range: {raw_vector[5]:.2f} °C)<br>",
            f"<b>Humidity / Pressure:</b> {raw_vector[6]:.1f}% / {raw_vector[7]:.2f} kPa<br>",
            f"<b>Wind (Mean / 99th):</b> {raw_vector[8]:.2f} / {raw_vector[9]:.2f} m/s<br>",
            f"<b>Annual Rain / Wet Days:</b> {raw_vector[10]:.1f} mm / {raw_vector[11] * 100:.1f}%"
        ]
        popup_html = "".join(popup_lines)

        if actual_label == 1:  # Highlight existing Solar Farms with larger pin markers
            folium.Marker(
                location=[latitude_value, longitude_value],
                popup=folium.Popup(popup_html, max_width=280),
                tooltip=f"[SF] {cell_key} -> Cluster {cluster_label}",
                icon=folium.Icon(color="darkred", icon="sun-o", prefix="fa")
            ).add_to(cluster_layer_groups[cluster_label])
        else:  # Render Non-Solar Farm grid cells as color-coded circles
            folium.CircleMarker(
                location=[latitude_value, longitude_value],
                radius=6,
                color="#222222",
                weight=1,
                fill=True,
                fill_color=marker_color,
                fill_opacity=0.85,
                popup=folium.Popup(popup_html, max_width=280),
                tooltip=f"[NSF] {cell_key} -> Cluster {cluster_label}"
            ).add_to(cluster_layer_groups[cluster_label])

    for layer_group in cluster_layer_groups:
        layer_group.add_to(interactive_map)

    folium.LayerControl(collapsed=False).add_to(interactive_map)  # Add live layer & satellite toggle

    output_html_path.parent.mkdir(parents=True, exist_ok=True)
    interactive_map.save(str(output_html_path))
    print(f"Saved interactive map     : {output_html_path}")

    if open_in_browser:
        resolved_html_path = str(output_html_path.resolve())  # Get full Linux path

        if "microsoft" in os.uname().release.lower():  # Detect WSL environment
            wslpath_result = subprocess.run(
                ["wslpath", "-w", resolved_html_path],
                capture_output=True,
                text=True,
                check=True
            )  # Convert /mnt/c/... to C:\...
            windows_file_path = wslpath_result.stdout.strip()
            subprocess.run(["explorer.exe", windows_file_path])  # Launch in default Windows browser
        else:
            webbrowser.open(output_html_path.resolve().as_uri())  # Standard Linux/macOS/Windows launch


# MAIN EXECUTION


def run_clustering_pipeline(pos_dir, neg_dir, outputs_dir, k_values=None, random_seed=170):  # Run full K-Means pipeline
    if k_values is None:
        k_values = [2, 3, 4, 5]

    positive_samples = load_climate_folder(pos_dir)  # Load Solar Farm (SF) grid cells
    negative_samples = load_climate_folder(neg_dir)  # Load Non-Solar Farm (NSF) grid cells

    cell_keys = []
    raw_feature_vectors = []
    ground_truth_labels = []

    for cell_key, feature_vector in positive_samples:
        cell_keys.append(cell_key)
        raw_feature_vectors.append(feature_vector)
        ground_truth_labels.append(1)

    for cell_key, feature_vector in negative_samples:
        cell_keys.append(cell_key)
        raw_feature_vectors.append(feature_vector)
        ground_truth_labels.append(0)

    lower_bounds, upper_bounds = compute_feature_bounds(raw_feature_vectors)  # Scale to [0.1, 0.9]
    scaled_feature_vectors = []
    for raw_vector in raw_feature_vectors:
        scaled_vector = scale_feature_vector(raw_vector, lower_bounds, upper_bounds)
        scaled_feature_vectors.append(scaled_vector)

    print("=" * 76)
    print("K-MEANS CLUSTERING: SOLAR FARM (1) vs. NON-SOLAR FARM (0) CLIMATE PROFILES")
    print("=" * 76)
    print(f"Total Grid Cells : {len(scaled_feature_vectors)} (SF: {len(positive_samples)}, NSF: {len(negative_samples)})")
    print(f"Features         : {len(FEATURES)} ({', '.join(FEATURES)})\n")
    print(f"{'k':<4} | {'Iterations':<11} | {'WCSS (Inertia)':<16} | {'Silhouette Score':<16}")
    print("-" * 56)

    best_k = -1
    best_silhouette = -2.0
    best_scaled_centroids = []
    best_cluster_assignments = []
    summary_rows = []

    for candidate_k in k_values:
        _, final_centroids, iteration_count, assignments = run_kmeans(
            scaled_feature_vectors,
            number_of_clusters=candidate_k,
            max_iterations=100,
            random_seed=random_seed
        )
        wcss_score = compute_wcss(scaled_feature_vectors, final_centroids, assignments)
        silhouette_score = compute_silhouette_score(scaled_feature_vectors, assignments, candidate_k)

        summary_rows.append((candidate_k, iteration_count, wcss_score, silhouette_score))
        print(f"{candidate_k:<4} | {iteration_count:<11} | {wcss_score:<16.4f} | {silhouette_score:<16.4f}")

        if silhouette_score > best_silhouette:
            best_silhouette = silhouette_score
            best_k = candidate_k
            best_scaled_centroids = final_centroids
            best_cluster_assignments = assignments

    print(f"\nOptimal k by Silhouette Score: k = {best_k} (Silhouette = {best_silhouette:.4f})")

    physical_centroids = print_centroid_feature_profiles(
        best_scaled_centroids,
        lower_bounds,
        upper_bounds,
        best_cluster_assignments,
        ground_truth_labels
    )

    metrics_dir = Path(outputs_dir) / "metrics"
    models_dir = Path(outputs_dir) / "models"
    plots_dir = Path(outputs_dir) / "plots"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    summary_path = metrics_dir / "kmeans_summary.csv"
    with summary_path.open("w") as summary_handle:
        summary_handle.write("k,iterations,wcss,silhouette_score\n")
        for k_val, iters, wcss_val, sil_val in summary_rows:
            summary_handle.write(f"{k_val},{iters},{wcss_val:.6f},{sil_val:.6f}\n")

    assignments_path = metrics_dir / "kmeans_assignments.csv"
    with assignments_path.open("w") as assignments_handle:
        assignments_handle.write("cell_key,actual_label,cluster_id\n")
        for sample_index in range(len(cell_keys)):
            assignments_handle.write(
                f"{cell_keys[sample_index]},{ground_truth_labels[sample_index]},{best_cluster_assignments[sample_index]}\n"
            )

    model_path = models_dir / "kmeans_model.json"
    with model_path.open("w") as model_handle:
        json.dump(
            {
                "model": "KMeans",
                "best_k": best_k,
                "silhouette_score": round(best_silhouette, 6),
                "seed": random_seed,
                "features": FEATURES,
                "lo": lower_bounds,
                "hi": upper_bounds,
                "centroids_scaled": best_scaled_centroids,
                "centroids_physical": physical_centroids
            },
            model_handle,
            indent=2
        )

    print(f"\nSaved cluster summary     : {summary_path}")
    print(f"Saved cell assignments    : {assignments_path}")
    print(f"Saved centroid artifact   : {model_path}")

    clustered_records = []
    for sample_index in range(len(cell_keys)):
        clustered_records.append((
            cell_keys[sample_index],
            best_cluster_assignments[sample_index],
            ground_truth_labels[sample_index]
        ))

    geoplot_path = plots_dir / "kmeans_geoplot.png"
    geoplot(clustered_records, geoplot_path, best_k)

    interactive_map_path = plots_dir / "kmeans_interactive_map.html"
    interactive_geoplot(
        clustered_records,
        raw_feature_vectors,
        interactive_map_path,
        best_k,
        open_in_browser=True
    )

if __name__ == "__main__":
    datasets_dir = REPO_ROOT / "Datasets"
    outputs_dir = REPO_ROOT / "Outputs"

    pos_folder = datasets_dir / "Weather (SF)"
    neg_folder = datasets_dir / "Weather (NSF)"

    if not pos_folder.exists() or not neg_folder.exists():
        print("Missing 'Weather (SF)' or 'Weather (NSF)' folder inside Datasets/.")
    else:
        run_clustering_pipeline(pos_folder, neg_folder, outputs_dir, k_values=[2, 3, 4, 5], random_seed=170)