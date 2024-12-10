from embrs.utilities.file_io import HistoricFireSelector
from embrs.map_generator import generate_map_from_file, DATA_RESOLUTION

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon, MultiPolygon, mapping
import numpy as np
import pickle
import json
import alphashape
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import utm
import pyproj
import openmeteo_requests
import requests_cache
from retry_requests import retry
from datetime import datetime, timedelta


# Function to convert (lon, lat) to relative (x, y)
def project_to_relative_xy(lon, lat, bounds):
    south, north, west, east = bounds
    x = (lon - west) / (east - west)
    y = (lat - south) / (north - south)
    return x, y

def extract_wind_data(params, bounds, time_step = 5):
    # Use center of the sim region as weather data location
    lat = params["Ignition Latitude"]
    lon = params["Ignition Longitude"]
    
    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    start_date = params["Scenario Start Date"]
    start_date_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_date_dt = start_date_dt + timedelta(days=7)
    end_date = end_date_dt.strftime("%Y-%m-%d")

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ["wind_speed_10m", "wind_direction_10m", "wind_gusts_10m"],
        "wind_speed_unit": "ms"
    }
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    hourly = response.Hourly()
    hourly_wind_speed_10m = hourly.Variables(0).ValuesAsNumpy()
    hourly_wind_direction_10m = hourly.Variables(1).ValuesAsNumpy()
    hourly_wind_gusts_10m = hourly.Variables(2).ValuesAsNumpy()

    hourly_data = {"date": pd.date_range(
        start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
        end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
        freq = pd.Timedelta(seconds = hourly.Interval()),
        inclusive = "left"
    )}
    hourly_data["wind_speed_10m"] = hourly_wind_speed_10m
    hourly_data["wind_direction_10m"] = hourly_wind_direction_10m
    hourly_data["wind_gusts_10m"] = hourly_wind_gusts_10m

    wind_df = pd.DataFrame(data = hourly_data)

    intervals = int(60 / time_step)

    forecast = []

    for index, row in wind_df.iterrows():
        speed_ms = row['wind_speed_10m']
        direction = row['wind_direction_10m']
        gusts_ms = row['wind_gusts_10m']

        direction = adjust_direction(direction)

        five_min_data = generate_data_for_hour(speed_ms, gusts_ms, direction, intervals)


        forecast.append((gusts_ms, direction))

        # forecast.extend(five_min_data)

    wind_data = [{'direction': dir, "speed_m_s": speed} for speed, dir in forecast]

    return {"time_step_min": 60, "data" : wind_data}


def generate_data_for_hour(speed_ms, gusts_ms, direction, intervals):
    # Returns list of (speed, dir)
    base_values = np.random.normal(loc = speed_ms, scale = 0.2 * speed_ms, size = intervals)

    gust_intervals = np.random.randint(0, intervals, size=3)
    base_values[gust_intervals] = gusts_ms

    # correction_factor  = speed_ms / np.mean(base_values)
    # corrected_values = base_values * correction_factor

    speeds = [(speed, direction) for speed in base_values]
    return speeds

def adjust_direction(direction):
    correct_dir = 270 - direction

    if correct_dir < 0:
        correct_dir += 360

    return correct_dir


def extract_scenario_data(params, bounds):
    scenario_data = {}

    ig_date = params['Ignition Date']
    start_date = params['Scenario Start Date']
    ig_lat = params['Ignition Latitude']
    ig_lon = params['Ignition Longitude']

	# Read shapefile # TODO: Get master dataset and change file path to its path
    shapefile_path = "/Users/rjdp3/Documents/Research/Code/firedpy/output/outputs/shapefiles/fired_california_2013_to_2023_daily.shp"
    gdf = gpd.read_file(shapefile_path)

    # Ensure the 'date' column is in datetime format
    gdf['date'] = pd.to_datetime(gdf['date'])
    gdf['ig_date'] = pd.to_datetime(gdf['ig_date'])

    # Reproject to WGS84 (EPSG:4326) if needed
    if gdf.crs != 'EPSG:4326':
        gdf = gdf.to_crs(epsg=4326)

    # Define the target point
    target_pt = Point(ig_lon, ig_lat)

    # Filter by dates
    gdf = gdf[(gdf['date'] >= ig_date) & (gdf['date'] <= start_date) & (gdf['ig_date'] == ig_date)]


    # Compute the distance from each geometry's centroid to the target point
    gdf['distance_to_target'] = gdf.geometry.centroid.distance(target_pt)

    
    # Filter based on a distance threshold (in kilometers)
    threshold_km = 100
    threshold_deg = threshold_km / 111  # Convert km to degrees approx.

    filtered_gdf = gdf[gdf['distance_to_target'] <= threshold_deg]

    print(filtered_gdf)
    # If there are no entries, print a message
    if filtered_gdf.empty:
        print("No entries found.")

    else:
        # Merge the geometries into a single shapely object
        merged_geom = filtered_gdf.unary_union

        # Extract the exterior coordinates of the merged polygon(s)
        if isinstance(merged_geom, Polygon):
            # For a single Polygon
            coords = list(merged_geom.exterior.coords)
        elif isinstance(merged_geom, MultiPolygon):
            # For a MultiPolygon, combine the exterior coordinates of all polygons
            coords = []
            for polygon in merged_geom.geoms:
                coords.extend(list(polygon.exterior.coords))

        # Convert coordinates to a list of (x, y) tuples
        points = np.array(coords)

        # Apply the function to each point
        # relative_points = [project_to_relative_xy(lon, lat, bounds) for lon, lat in points] # TODO: need to multiple each point by x and y size of sim

        # Calculate the concave hull (alpha shape)
        alpha = 30  # Adjust this parameter for more or less detailed shapes
        concave_hull = alphashape.alphashape(points, alpha)


        # If it's a Polygon, you can directly use it as a Shapely Polygon
        if isinstance(concave_hull, Polygon):
            # Get the exterior boundary of the Polygon
            polygon_boundary = concave_hull.exterior
            # Create a Shapely Polygon from the boundary
            boundary_polygon = Polygon(polygon_boundary)
        elif isinstance(concave_hull, MultiPolygon):
            # If it's a MultiPolygon, extract boundaries of each polygon
            boundary_polygon = MultiPolygon([Polygon(poly.exterior) for poly in concave_hull.geoms])
        else:
            raise ValueError("Unexpected geometry type returned from alphashape.")

        # Create a GeoDataFrame with the concave hull
        concave_hull_gdf = gpd.GeoDataFrame(geometry=[concave_hull], crs=filtered_gdf.crs)

        # Extract the boundary polygon (geometry) from the GeoDataFrame
        boundary = concave_hull_gdf.boundary.iloc[0]  # Get the first geometry (as there's only one)

        # Calculate the central point of the bounding box
        central_lat = (bounds[0] + bounds[1]) / 2
        central_lon = (bounds[2] + bounds[3]) / 2

        # Get the UTM zone of the central point
        _, _, utm_zone, _ = utm.from_latlon(central_lat, central_lon)

        # Get projection based on the utm_zone
        proj = pyproj.Proj(proj='utm', zone=utm_zone, ellps='WGS84')

        # Get the origin in x,y coordinates
        origin_x, origin_y = proj(bounds[2], bounds[0])
        max_x, max_y = proj(bounds[3], bounds[1])

        bbox_width = max_x - origin_x
        bbox_height = max_y - origin_y

        proj_points = []

        # First pass: get all node elements with their coordinates
        for point in boundary.coords:
            lon, lat = point    
            x, y = proj(lon, lat)

            # account for buffer on other data
            x -= 120    
            y -= 120

            proj_points.append((x - origin_x, y - origin_y))

        fig, ax = plt.subplots()
        ax.set_aspect('equal')

        # Draw bounding box
        bbox = patches.Rectangle((0, 0), bbox_width, bbox_height, linewidth=1, edgecolor='r', facecolor='none', label="Bounding Box")
        ax.add_patch(bbox)

        # Plot projected points
        x_coords, y_coords = zip(*proj_points)
        ax.scatter(x_coords, y_coords, color='blue', label="Projected Points")

        # Labels and legend
        plt.xlabel("UTM X Coordinate (meters)")
        plt.ylabel("UTM Y Coordinate (meters)")
        plt.legend()
        plt.title("Bounding Box and Projected Points in UTM Coordinates")
        plt.grid(True)

        plt.show()


        # Create a Shapely polygon from the boundary's exterior coordinates
        boundary_polygon = Polygon(proj_points)

        # Plot the concave hull
        concave_hull_gdf.boundary.plot(color='black', linewidth=2, label='Concave Hull')


        polygons = [mapping(boundary_polygon)]

        # Store in scenario_data
        scenario_data["initial_ignition"] = polygons
        scenario_data["fire_breaks"] = [] # TODO: Allow users to draw this on

        plt.show()

    return scenario_data

def save_to_file(params: dict, scenario_data: dict, wind_data: dict):
    save_path = params['save_path']
    fuel_data = params['fuel_data']
    topography_data = params['topography_data']
    aspect_data = params['aspect_data']
    slope_data = params['slope_data']
    roads = params['roads']
    bounds = params['bounds']


    # Save wind forecast
    wind_path = save_path + '/wind.json'

    with open(wind_path, 'w') as json_file:
        json.dump(wind_data, json_file, indent=4)

    data = {}

    # Save fuel data
    fuel_path = save_path + '/fuel.npy'

    np.save(fuel_path, fuel_data['map'])

    if bounds is None:
        data['geo_bounds'] = None

    else:
        data['geo_bounds'] = {
            'south': bounds[0],
            'north': bounds[1],
            'west': bounds[2],
            'east': bounds[3] 
        }

    data['fuel'] = {'file': fuel_path,
                        'width_m': fuel_data['width_m'],
                        'height_m': fuel_data['height_m'],
                        'rows': fuel_data['rows'],
                        'cols': fuel_data['cols'],
                        'resolution': fuel_data['resolution'],
                        'uniform': fuel_data['uniform']
                    }
    
    if fuel_data['uniform']:
        data['fuel']['fuel type'] = fuel_data['fuel type']


    # Save topography data
    topography_path = save_path + '/topography.npy'
    np.save(topography_path, topography_data['map'])

    data['topography'] = {'file': topography_path,
                        'width_m': topography_data['width_m'],
                        'height_m': topography_data['height_m'],
                        'rows': topography_data['rows'],
                        'cols': topography_data['cols'],
                        'resolution': topography_data['resolution'],
                        'uniform': topography_data['uniform']
                        }

    aspect_path = save_path + '/aspect.npy'
    np.save(aspect_path, aspect_data['map'])

    data['aspect'] = {'file': aspect_path,
                      'width_m': aspect_data['width_m'],
                      'height_m': aspect_data['height_m'],
                      'rows': aspect_data['rows'],
                      'cols': aspect_data['cols'],
                      'resolution': aspect_data['resolution'],
                      'uniform': aspect_data['uniform']}

    slope_path = save_path + '/aspect.npy'
    np.save(slope_path, slope_data['map'])

    data['slope'] = {'file': slope_path,
                      'width_m': slope_data['width_m'],
                      'height_m': slope_data['height_m'],
                      'rows': slope_data['rows'],
                      'cols': slope_data['cols'],
                      'resolution': slope_data['resolution'],
                      'uniform': slope_data['uniform']}

    # Save the roads data
    road_path = save_path + '/roads.pkl'
    with open(road_path, 'wb') as f:
        pickle.dump(roads, f)

    data['roads'] = {'file': road_path}

    data = data | scenario_data

    # Save data to JSON
    folder_name = os.path.basename(save_path)
    with open(save_path + "/" + folder_name + ".json", 'w') as f:
        json.dump(data, f, indent=4)


def main():
    file_selector = HistoricFireSelector()
    file_params = file_selector.run()

    params = generate_map_from_file(file_params, DATA_RESOLUTION, 1)

    wind_data = extract_wind_data(file_params, params['bounds'])
    scenario_data = extract_scenario_data(file_params, params['bounds'])

    save_to_file(params, scenario_data, wind_data)

if __name__ == "__main__":
	main()