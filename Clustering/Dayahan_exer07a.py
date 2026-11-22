# Responsible Use of AI
# Dayahan, Rey Russel M.
#References used: https://stackoverflow.com/questions/16503560/read-specific-columns-from-a-csv-file-with-csv-module
#Use of Ai: Understanding matpltlib script for plotting

import sys
import csv
import math
import random
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import contextily as ctx



# Constants
DATASET_FILE_NAME = "crime-data.csv"
MAX_ENTRIES_TO_PROCESS = 500

def read_csv_data(file_path):       # Reads csv file with valid coordinates
    dataset = [] 
    try:
        with open(file_path, 'r', encoding="ISO-8859-1") as csv_file_handler: 
            csv_reader = csv.reader(csv_file_handler)
            # header_row = next(csv_reader)  # Skip the header row from reading
            
            entries_processed_count = 0
            for row in csv_reader: 
                if entries_processed_count >= MAX_ENTRIES_TO_PROCESS:
                    break  # Limit to first 500 valid entries
                
                try:
                    crime_description = row[4]  # desc column
                    longitude = row[10]  #longitude column
                    latitude = row[11]   #lat column
                    
                    if longitude.strip() != "" and latitude.strip() != "":
                        longitude_value = float(longitude)
                        latitude_value = float(latitude)
                        
                        processed_row = [crime_description, longitude_value, latitude_value]
                        dataset.append(processed_row)
                        entries_processed_count += 1
                        
                except (IndexError, ValueError):
                    continue  # skip invalid rows
                    
    except FileNotFoundError:
        print(f"Error: Dataset file '{file_path}' not found.") 
        sys.exit(1)
        
    return dataset

def euclidean_distance(point_a, point_b):  # Computes Euclidean distance
    summation_of_squared_diffs = 0.0
    num_dimensions = len(point_a)
    
    for dimension_index in range(num_dimensions):
        diff = point_a[dimension_index] - point_b[dimension_index]
        diff_squared = diff * diff
        summation_of_squared_diffs += diff_squared
        
    distance_value = math.sqrt(summation_of_squared_diffs)

    return distance_value

def initialize_centroids(dataset_rows, number_of_clusters):
    # testing_centroids_strings = [
    #     ("-76.61961", "39.29164"),
    #     ("-76.66073", "39.31828"),
    #     ("-76.60463", "39.32736"),
    #     ("-76.66799", "39.27466"),
    #     ("-76.60308", "39.23302")
    # ]
    
    # initial_centroids_list = []
    
    # for index in range(number_of_clusters):
    #     if index < len(testing_centroids_strings):     
    #         longitude_value = float(testing_centroids_strings[index][0])   #convert coor to float
    #         latitude_value = float(testing_centroids_strings[index][1])    #convert to float
    #         initial_centroids_list.append([longitude_value, latitude_value])    #add to list
    #     else:
    #         # Fallback to a random data point if k is greater than the provided test centroids
    #         random_row = random.choice(dataset_rows)
    #         longitude_value = random_row[1]
    #         latitude_value = random_row[2]
    #         initial_centroids_list.append([longitude_value, latitude_value])
            
    # return initial_centroids_list

    # Initializes random centroids from dataset coordinates

    feature_points = []
    for row in dataset_rows:
        longitude = row[1]
        latitude = row[2]
        feature_points.append([longitude, latitude])
        
    initial_centroids_list = random.sample(feature_points, number_of_clusters)  #get random samples to test
    return initial_centroids_list

def run_kmeans(dataset_rows, number_of_clusters, max_iterations=100):   #k means fucntion
    feature_points = []
    for row in dataset_rows:
        feature_points.append([row[1], row[2]])
        
    initial_centroids = initialize_centroids(dataset_rows, number_of_clusters)
    
    # Store a snapshot of initial centroids for printing
    initial_centroids_copy = []
    for centroid in initial_centroids:
        initial_centroids_copy.append(list(centroid))
        
    current_centroids = initial_centroids
    iteration_count = 0
    cluster_assignments = []
    
    for iteration_index in range(max_iterations):
        iteration_count += 1
        cluster_assignments = []
        

        for point in feature_points:      #assign point to nearest centroid 
            min_distance = float('inf')
            assigned_cluster = -1
            
            for centroid_index in range(number_of_clusters):
                centroid = current_centroids[centroid_index]
                distance = euclidean_distance(point, centroid)   #apply euclidean distance between points
                
                if distance < min_distance:                  
                    min_distance = distance                     #update
                    assigned_cluster = centroid_index
                    
            cluster_assignments.append(assigned_cluster)
            
        new_centroids = []
        for centroid_index in range(number_of_clusters):
            sum_longitude = 0.0
            sum_latitude = 0.0
            point_count = 0
            
            for point_index in range(len(feature_points)):
                if cluster_assignments[point_index] == centroid_index:
                    sum_longitude += feature_points[point_index][0]
                    sum_latitude += feature_points[point_index][1]
                    point_count += 1
                    
            if point_count > 0:
                mean_longitude = sum_longitude / point_count
                mean_latitude = sum_latitude / point_count
                new_centroids.append([mean_longitude, mean_latitude])         #append to list
            else:
                fallback_point = random.choice(feature_points)  # Reinitialize empty cluster
                new_centroids.append(list(fallback_point))
 
        centroids_converged = True
        for centroid_index in range(number_of_clusters):
            if current_centroids[centroid_index] != new_centroids[centroid_index]:
                centroids_converged = False                                             #terminate if centroid does not change
                break
                
        current_centroids = new_centroids                   #update
        if centroids_converged:
            break
            
    return initial_centroids_copy, current_centroids, iteration_count, cluster_assignments


def geoplot(kmeans_output):
    
    # Convert to a DataFrame
    data = pd.DataFrame(kmeans_output, columns=["classification", "coordinates"])

    # Split the coordinates into longitude and latitude columns
    data["longitude"] = data["coordinates"].apply(lambda x: x[0])
    data["latitude"] = data["coordinates"].apply(lambda x: x[1])

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(
        data,
        geometry=gpd.points_from_xy(data.longitude, data.latitude),
        crs="EPSG:4326"  # WGS84 coordinate system
    )

    # Reproject to Web Mercator for compatibility with basemaps
    gdf = gdf.to_crs("EPSG:3857")

    # Plotting
    fig, ax = plt.subplots(figsize=(5, 5))

    # Plot points with classification colors
    for cluster in gdf["classification"].unique():
        cluster_gdf = gdf[gdf["classification"] == cluster]
        cluster_gdf.plot(ax=ax, markersize=10, label=f"Cluster {cluster}")

    # Add basemap
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
    # ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    ax.set_title("K-means Clustering on Baltimore Crime Data")
    ax.legend()
    plt.show()


def solver():
    dataset_rows = read_csv_data(DATASET_FILE_NAME)
    
    number_of_clusters = int(input("Enter number of clusters k: ")) 
    
    initial_centroids, final_centroids, iteration_count, cluster_assignments = run_kmeans(
        dataset_rows, number_of_clusters
    )
    
    # Print headers and results
    print(f"\nK-Means Clustering Output from {DATASET_FILE_NAME}")
    print(f"k = {number_of_clusters}\n")
    
    print("Initial Centroids")
    for centroid in initial_centroids:
        print(f"({centroid[0]:.5f}, {centroid[1]:.5f})")
        
    print("\nFinal Centroids")
    for centroid in final_centroids:
        print(f"({centroid[0]:.14f}, {centroid[1]:.14f})")
        
    print(f"\nIterations: {iteration_count}")
    
    print("\nLabeled Dataset:")
    formatted_output_for_plotting = []
    
    for index in range(len(dataset_rows)):
        row = dataset_rows[index]
        crime_description = row[0]
        longitude = row[1]
        latitude = row[2]
        cluster_label = cluster_assignments[index]
        
        formatted_output_for_plotting.append((cluster_label, [longitude, latitude]))
        
        if index < 25:  # Print sample preview of labeled dataset
            print(f"{cluster_label} : {crime_description:<25} [{longitude:.6f}, {latitude:.6f}]")
            
    print("...\n")
    
    #plotting
    geoplot(formatted_output_for_plotting)

solver()