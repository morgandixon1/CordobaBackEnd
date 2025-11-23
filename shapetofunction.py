import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from skimage.morphology import remove_small_holes, remove_small_objects, binary_opening, binary_closing, disk, skeletonize
from skimage.measure import label
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from shapely.geometry import LineString
from math import atan2, degrees
from rdp import rdp
import requests
from io import BytesIO
from scipy.spatial import distance
from scipy.ndimage import distance_transform_edt
import json
from scipy.interpolate import griddata
import pyproj
import time
import random
import traceback
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource
from matplotlib.patches import Polygon
import numpy as np
from scipy.spatial import distance
from skimage.morphology import skeletonize
from skimage.measure import label
from shapely.geometry import LineString
from rdp import rdp
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from shapely.geometry import LineString, Point
from skimage.draw import line
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.widgets import Button
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
#from flask import Flask, jsonify, request, abort
#from flask_restful import Api, Resource
#from pyngrok import ngrok
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from pyproj import Transformer
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# app = Flask(__name__)
# api = Api(app)

# Load API keys from environment variables
VALID_API_KEYS = set(os.getenv('VALID_API_KEYS', '').split(',')) if os.getenv('VALID_API_KEYS') else set()

METERS_TO_COLLECT = 500
GRID_SIZE = 100

class ElevationUtils:
    @staticmethod
    def collect_elevation(image_bounds, api_key_opentopo, zoom_factor=1.5):
        print("Starting elevation data collection...")
        west, south, east, north = image_bounds
        
        # Adjust bounds to zoom in
        center_lon = (west + east) / 2
        center_lat = (south + north) / 2
        half_width = (east - west) / (2 * zoom_factor)
        half_height = (north - south) / (2 * zoom_factor)
        
        west = center_lon - half_width
        east = center_lon + half_width
        south = center_lat - half_height
        north = center_lat + half_height
        
        print(f"Adjusted bounds: West: {west}, South: {south}, East: {east}, North: {north}")

        # Create a grid of points for API calls
        num_points = 10  # Adjust this for more or fewer API calls
        lats = np.linspace(south, north, num_points)
        lons = np.linspace(west, east, num_points)
        locations = [(lat, lon) for lat in lats for lon in lons]

        elevations = []
        for i in range(0, len(locations), 50):  # OpenTopoData allows up to 100 locations per request
            chunk = locations[i:i+50]
            locations_str = '|'.join([f"{lat},{lon}" for lat, lon in chunk])
            
            try:
                url = 'https://api.opentopodata.org/v1/ned10m'
                params = {'locations': locations_str, 'api_key': api_key_opentopo}
                print(f"Fetching elevation data for chunk {i//50 + 1}/{(len(locations)-1)//50 + 1}")
                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                if 'results' not in data:
                    print("Unexpected response from OpenTopo API.")
                    return None
                
                elevations.extend([result['elevation'] for result in data['results']])
                time.sleep(1)  # To avoid hitting rate limits
            
            except requests.exceptions.RequestException as e:
                print(f"Error fetching elevation data: {str(e)}")
                return None

        print("Elevation data collected. Interpolating to grid...")

        # Create grid for interpolation
        grid_x, grid_y = np.meshgrid(np.linspace(west, east, GRID_SIZE), np.linspace(north, south, GRID_SIZE))
        points = np.array([(lon, lat) for lat, lon in locations])
        grid_z = griddata(points, elevations, (grid_x, grid_y), method='cubic')

        # Handle any NaN values
        grid_z = np.nan_to_num(grid_z, nan=np.nanmean(grid_z))

        print("Elevation data processing completed.")
        
        return {
            'elevation_values': grid_z.tolist(),
            'x_values': np.linspace(0, GRID_SIZE-1, GRID_SIZE).tolist(),
            'y_values': np.linspace(0, GRID_SIZE-1, GRID_SIZE).tolist(),
            'lat_range': [south, north],
            'lon_range': [west, east],
            'grid_size': GRID_SIZE,
            'min_elevation': float(np.min(grid_z)),
            'max_elevation': float(np.max(grid_z))
        }
        
class MapUtils:
    EARTH_CIRCUMFERENCE = 40075016.686  # in meters
    TILE_SIZE = 512

    @staticmethod
    def calculate_bounds(lon, lat, zoom, width, height):
        meters_per_pixel = MapUtils.EARTH_CIRCUMFERENCE * np.cos(np.radians(lat)) / (2 ** (zoom + 8))
        span_x = width * meters_per_pixel
        span_y = height * meters_per_pixel
        lon_span = span_x / (MapUtils.EARTH_CIRCUMFERENCE * np.cos(np.radians(lat)) / 360)
        lat_span = span_y / (MapUtils.EARTH_CIRCUMFERENCE / 360)
        return (
            lon - lon_span / 2,  # west
            lat - lat_span / 2,  # south
            lon + lon_span / 2,  # east
            lat + lat_span / 2   # north
        )

    @classmethod
    def fetch_expanded_area(cls, style, access_token, center_lat, center_lon, zoom=18):
        # Calculate the size of the area to fetch
        meters_per_pixel = cls.EARTH_CIRCUMFERENCE * np.cos(np.radians(center_lat)) / (2 ** (zoom + 9))
        current_meters = meters_per_pixel * cls.TILE_SIZE
        
        while current_meters > METERS_TO_COLLECT:
            zoom += 1
            meters_per_pixel = cls.EARTH_CIRCUMFERENCE * np.cos(np.radians(center_lat)) / (2 ** (zoom + 9))
            current_meters = meters_per_pixel * cls.TILE_SIZE
        
        img = cls.fetch_mapbox_image(style, access_token, center_lon, center_lat, zoom, METERS_TO_COLLECT, METERS_TO_COLLECT)
        return img, zoom

    @staticmethod
    def fetch_mapbox_image(style, access_token, lon, lat, zoom, width, height):
        url = f"https://api.mapbox.com/styles/v1/{style}/static/{lon},{lat},{zoom}/{width}x{height}?access_token={access_token}"
        response = requests.get(url)
        response.raise_for_status()
        if 'image' not in response.headers['Content-Type']:
            raise Exception(f"Unexpected content type: {response.headers['Content-Type']}")
        return Image.open(BytesIO(response.content)).convert('RGB')

    @staticmethod
    def load_and_preprocess_image(image, target_colors, tolerance=30):
        img_array = np.array(image)
        if img_array.ndim == 2:
            img_array = np.stack((img_array,) * 3, axis=-1)
        masks = {}
        for color_name, target_color in target_colors.items():
            mask = np.sqrt(np.sum((img_array - target_color) ** 2, axis=2)) <= tolerance
            mask = binary_closing(binary_opening(remove_small_objects(remove_small_holes(mask, area_threshold=100), min_size=500), disk(2)), disk(2))
            skeleton = skeletonize(mask)
            masks[color_name] = (mask, skeleton)
        print("image preprocessed")
        return masks

class RoadProcessor:
    def __init__(self, real_bounds, grid_size):
        self.real_bounds = real_bounds
        self.grid_size = grid_size
        self.road_mask = None
        self.image_size = None

    def get_skeleton_sections(self, skeleton):
        labeled_mask, num_labels = label(skeleton, return_num=True)
        sections = [np.argwhere(labeled_mask == i) for i in range(1, num_labels + 1)]
        return sections

    def is_valid_road_point(self, point):
        return self.road_mask[tuple(point)] > 0

    def find_road_endpoints(self, points):
        valid_endpoints = [p for p in points if self.is_valid_road_point(p)]
        if len(valid_endpoints) < 2:
            return points[0], points[-1]
        distances = cdist(valid_endpoints, valid_endpoints)
        i, j = np.unravel_index(distances.argmax(), distances.shape)
        return valid_endpoints[i], valid_endpoints[j]

    def is_valid_path(self, start, end):
        rr, cc = line(int(start[0]), int(start[1]), int(end[0]), int(end[1]))
        return np.all(self.road_mask[rr, cc] > 0)

    def order_points(self, points):
        start, end = self.find_road_endpoints(points)
        ordered_sections = []
        current_section = [tuple(start)]
        remaining = set(map(tuple, points)) - set(current_section)
        
        while remaining:
            last = np.array(current_section[-1])
            nearest = min(remaining, key=lambda p: np.linalg.norm(np.array(p) - last))
            if self.is_valid_path(last, nearest):
                current_section.append(nearest)
                remaining.remove(nearest)
            else:
                # Start a new section
                ordered_sections.append(np.array(current_section))
                if remaining:
                    new_start = remaining.pop()
                    current_section = [new_start]
                    remaining.add(new_start)  # Add it back to process its neighbors
        
        # Add the last section
        if current_section:
            ordered_sections.append(np.array(current_section))
        
        return ordered_sections

    def simplify_section(self, section, epsilon=1.0):
        if len(section) < 3:
            return section
        line = LineString(section)
        simplified = line.simplify(epsilon, preserve_topology=False)
        return np.array(simplified.coords).astype(int)

    def reduce_points(self, points, max_distance=5):
        result = [points[0]]
        for point in points[1:]:
            if np.linalg.norm(point - result[-1]) > max_distance:
                result.append(point)
        if not np.array_equal(result[-1], points[-1]):
            result.append(points[-1])
        return np.array(result)

    def rotate_points_minus_90_degrees(self, points):
        rotated_points = []
        for point in points:
            rotated_x = self.grid_size - 1 - point[1]
            rotated_y = point[0]
            rotated_points.append([rotated_x, rotated_y])
        return np.array(rotated_points)

    def process_roads(self, road_masks):
        processed_roads = {}
        for road_type, mask in road_masks.items():
            if road_type != 'buildings':
                mask, skeleton = mask  # Unpack mask and skeleton
                self.road_mask = mask  # Set the road mask for validity checks
                self.image_size = (mask.shape[1], mask.shape[0])  # Set image size
                sections = self.get_skeleton_sections(skeleton)
                processed_sections = []
                
                for section in sections:
                    if len(section) > 1:  # Ensure we have at least two points
                        ordered_sections = self.order_points(section)
                        for ordered_section in ordered_sections:
                            simplified_section = self.simplify_section(ordered_section)
                            reduced_section = self.reduce_points(simplified_section)
                            grid_coordinates = CoordinateUtils.convert_to_grid_coordinates(
                                reduced_section, self.real_bounds, self.image_size, self.grid_size
                            )
                            processed_sections.append(grid_coordinates)
                
                processed_roads[road_type] = processed_sections
        
        return processed_roads

class BuildingProcessor:
    def __init__(self, real_bounds, grid_size):
        self.real_bounds = real_bounds
        self.grid_size = grid_size
        print(f"BuildingProcessor initialized with bounds: {real_bounds} and grid size: {grid_size}")

    def mirror_points(self, points):
        return [[self.grid_size - 1 - point[0], point[1]] for point in points]

    def rotate_90_degrees(self, points):
        return [[point[1], self.grid_size - 1 - point[0]] for point in points]

    def process_buildings(self, image, building_rgb):
        print(f"Processing buildings with RGB: {building_rgb}")
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Convert image to RGB color space
        rgb_image = image.copy()
        if rgb_image.shape[-1] == 4:  # RGBA
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGBA2RGB)
        elif rgb_image.shape[-1] == 3:  # RGB
            pass
        else:  # Grayscale or other format
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_GRAY2RGB)

        # Create mask based on the specific building color
        tolerance = 10  # Adjust this value to be more or less strict
        lower_bound = np.array([max(0, c - tolerance) for c in building_rgb])
        upper_bound = np.array([min(255, c + tolerance) for c in building_rgb])
        mask = cv2.inRange(rgb_image, lower_bound, upper_bound)

        # Apply morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"Number of contours found: {len(contours)}")
        buildings = []

        image_size = (image.shape[1], image.shape[0])  # width, height
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 50:  # Adjust this threshold based on your needs
                continue

            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            polygon_points = approx.reshape(-1, 2).tolist()

            # Get the centroid of the contour
            M = cv2.moments(contour)
            if M["m00"] != 0:
                centroid = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            else:
                centroid = (0, 0)

            # Convert to grid coordinates
            grid_polygon_points = CoordinateUtils.convert_to_grid_coordinates(
                polygon_points, self.real_bounds, image_size, self.grid_size
            )
            grid_centroid = CoordinateUtils.convert_single_point(
                centroid, self.real_bounds, image_size, self.grid_size
            )

            # Mirror the points
            mirrored_grid_polygon_points = self.mirror_points(grid_polygon_points)
            mirrored_grid_centroid = self.mirror_points([grid_centroid])[0]

            # Rotate the points 90 degrees clockwise
            rotated_grid_polygon_points = self.rotate_90_degrees(mirrored_grid_polygon_points)
            rotated_grid_centroid = self.rotate_90_degrees([mirrored_grid_centroid])[0]

            buildings.append({
                'corner_points': rotated_grid_polygon_points,
                'centroid': rotated_grid_centroid,
                'building_type': 'House'
            })

            # Debug print
            print(f"Original polygon points: {polygon_points}")
            print(f"Grid polygon points: {grid_polygon_points}")
            print(f"Mirrored grid polygon points: {mirrored_grid_polygon_points}")
            print(f"Rotated grid polygon points: {rotated_grid_polygon_points}")
            print(f"Grid centroid: {grid_centroid}")
            print(f"Mirrored grid centroid: {mirrored_grid_centroid}")
            print(f"Rotated grid centroid: {rotated_grid_centroid}")

        print(f"Number of buildings processed: {len(buildings)}")
        return buildings, mask

class DataProcessor:
    def __init__(self, real_bounds, grid_size):
        self.real_bounds = real_bounds
        self.grid_size = grid_size

    def flip_and_mirror(self, points):
        return [[self.grid_size - 1 - point[1], point[0]] for point in points]

    def process_building_and_road_data(self, buildings, processed_roads):
        house_count = 1
        commercial_count = 1
        building_data = []
        road_data = []
        section_labels = {
            'roads': 'Road',
            'freeway': 'Freeway',
            'highway': 'Highway'
        }

        for building in buildings:
            if building['building_type'] == 'House':
                building_title = f"House {house_count}"
                house_count += 1
            else:  # Commercial
                building_title = f"Commercial Building {commercial_count}"
                commercial_count += 1

            building_info = {
                'title': building_title,
                'center_point': self.flip_and_mirror([building['centroid']])[0],
                'corner_points': self.flip_and_mirror(building['corner_points'])
            }
            building_data.append(building_info)

        for road_type, sections in processed_roads.items():
            section_label = section_labels.get(road_type, 'Road')
            for idx, section in enumerate(sections, start=1):
                section_info = {
                    'title': f"{section_label} Section {idx}",
                    'points': self.flip_and_mirror(section)
                }
                road_data.append(section_info)

        return building_data, road_data

class CoordinateUtils:
    @staticmethod
    def convert_to_grid_coordinates(points, real_bounds, image_size, grid_size):
        west, south, east, north = real_bounds
        image_width, image_height = image_size
        
        converted_points = []
        for point in points:
            # Convert image coordinates to normalized coordinates
            normalized_x = point[0] / image_width
            normalized_y = point[1] / image_height  # Note: Not flipping Y-axis here
            
            # Convert normalized coordinates to real-world coordinates
            real_x = west + normalized_x * (east - west)
            real_y = north - normalized_y * (north - south)  # Flip Y-axis here
            
            # Convert real-world coordinates to grid coordinates
            grid_x = int((real_x - west) / (east - west) * (grid_size - 1))
            grid_y = int((north - real_y) / (north - south) * (grid_size - 1))
            
            # Ensure the point is within the grid bounds
            grid_x = max(0, min(grid_size - 1, grid_x))
            grid_y = max(0, min(grid_size - 1, grid_y))
            
            converted_points.append([grid_x, grid_y])
        
        return converted_points

    @staticmethod
    def convert_single_point(point, real_bounds, image_size, grid_size):
        return CoordinateUtils.convert_to_grid_coordinates([point], real_bounds, image_size, grid_size)[0]

class GeocodingUtils:
    @staticmethod
    def geocode_address(address, access_token):
        geocoding_url = f"https://api.mapbox.com/geocoding/v5/mapbox.places/{address}.json?access_token={access_token}"
        response = requests.get(geocoding_url)
        response.raise_for_status()
        data = response.json()
        if data['features']:
            lon, lat = data['features'][0]['center']
            return lat, lon
        else:
            raise Exception("Address not found")

class ImageUtils:
    @staticmethod
    def plot_stitched_image(stitched_image, expansion):
    
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(stitched_image)
        grid_size = 1 if expansion == 1 else (3 if expansion == 2 else 5)
        ax.set_title(f"Stitched Image (Expansion: {grid_size}x{grid_size})")
        
        for i in range(1, grid_size):
            ax.axvline(x=i * stitched_image.width / grid_size, color='r', linestyle='--')
            ax.axhline(y=i * stitched_image.height / grid_size, color='r', linestyle='--')
        
        ax.axis('off')
        try:
            plt.show()
        except:
            print("Could not display plot interactively. Please check the saved image file.")
        finally:
            plt.close(fig)

def main(api_key_opentopo, mapbox_token, style, address):
    target_colors = {
        'roads': (255, 255, 255),
        'freeway': (255, 179, 102),
        'highway': (247, 224, 110),
        'buildings': (172, 66, 13)
    }
    print("Starting main function...")

    # Geocode address
    geocoding_utils = GeocodingUtils()
    latitude, longitude = geocoding_utils.geocode_address(address, mapbox_token)
    print(f"Geocoded address to Latitude: {latitude}, Longitude: {longitude}")

    # Fetch map image
    print("Fetching map image...")
    map_utils = MapUtils()
    pil_image, zoom = map_utils.fetch_expanded_area(style, mapbox_token, latitude, longitude, zoom=17)

    bounds = map_utils.calculate_bounds(longitude, latitude, zoom, METERS_TO_COLLECT, METERS_TO_COLLECT)
    print(f"Image center coordinates: Lat {latitude}, Lon {longitude}")
    print(f"Image bounds: West: {bounds[0]}, South: {bounds[1]}, East: {bounds[2]}, North: {bounds[3]}")

    # Collect elevation data
    print("Collecting elevation data...")
    elevation_data = ElevationUtils.collect_elevation(bounds, api_key_opentopo)
    if elevation_data is None:
        print("Failed to retrieve elevation data.")
        return None

    # Process buildings and roads
    building_processor = BuildingProcessor(bounds, GRID_SIZE)
    buildings, building_mask = building_processor.process_buildings(pil_image, target_colors['buildings'])
    print(f"Number of buildings detected: {len(buildings)}")

    road_processor = RoadProcessor(bounds, GRID_SIZE)
    road_masks = map_utils.load_and_preprocess_image(pil_image, target_colors)
    processed_roads = road_processor.process_roads(road_masks)

    # Format building and road data
    data_processor = DataProcessor(bounds, GRID_SIZE)
    building_data, road_data = data_processor.process_building_and_road_data(buildings, processed_roads)

    output_data = {
        'elevation_data': elevation_data,
        'buildings': building_data,
        'roads': road_data
    }

    # Print some debug information
    print(f"Number of buildings in output data: {len(output_data['buildings'])}")
    print(f"Number of road sections in output data: {len(output_data['roads'])}")
    with open('area3.json', 'w') as f:
        json.dump(output_data, f, indent=2)

    print("Data processed and saved to area3.json")
    return output_data

if __name__ == '__main__':
    import sys
    import argparse

    # Load default values from environment variables
    DEFAULT_ADDRESS = os.getenv('DEFAULT_ADDRESS', '123 Main St, Your City, State ZIP')
    DEFAULT_STYLE = os.getenv('MAPBOX_STYLE', 'mapbox/streets-v11')
    DEFAULT_MAPBOX_TOKEN = os.getenv('MAPBOX_ACCESS_TOKEN', '')
    DEFAULT_API_KEY_OPENTOPO = os.getenv('OPENTOPO_API_KEY', '')

    parser = argparse.ArgumentParser(description='Fetch and process map data')
    parser.add_argument('--address', type=str, default=DEFAULT_ADDRESS, help='Address to geocode')
    parser.add_argument('--api_key_opentopo', type=str, default=DEFAULT_API_KEY_OPENTOPO, help='API key for OpenTopoData')
    parser.add_argument('--mapbox_token', type=str, default=DEFAULT_MAPBOX_TOKEN, help='Mapbox access token')
    parser.add_argument('--style', type=str, default=DEFAULT_STYLE, help='Mapbox style URL')
    args = parser.parse_args()

    # Validate API keys are provided
    if not args.mapbox_token:
        print("Error: MAPBOX_ACCESS_TOKEN not found. Please set it in .env file or pass via --mapbox_token")
        sys.exit(1)
    if not args.api_key_opentopo:
        print("Error: OPENTOPO_API_KEY not found. Please set it in .env file or pass via --api_key_opentopo")
        sys.exit(1)

    print(f"Using address: {args.address}")
    print(f"Using Mapbox style: {args.style}")
    print(f"Using Mapbox token: {args.mapbox_token[:10]}...{args.mapbox_token[-5:]}")
    print(f"Using OpenTopoData API key: {args.api_key_opentopo[:5]}...{args.api_key_opentopo[-5:]}")

    result = main(args.api_key_opentopo, args.mapbox_token, args.style, args.address)
    
    if result:
        print("Processing completed successfully.")
    else:
        print("Processing failed.")

class InteractiveVisualizer:
    def __init__(self, json_file_path):
        with open(json_file_path, 'r') as f:
            self.data = json.load(f)
        
        self.elevation_data = self.data['elevation_data']
        self.elevation_values = np.array(self.elevation_data['elevation_values'])
        self.x_values = np.array(self.elevation_data['x_values'])
        self.y_values = np.array(self.elevation_data['y_values'])
        self.buildings = self.data['buildings']
        self.roads = self.data['roads']

        self.grid_size = self.elevation_values.shape[0]
        self.flip_x = False
        self.flip_y = False
        self.rotation = 0  # Rotation in degrees

        self.fig = plt.figure(figsize=(12, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

    def rotate_coordinates(self, coords):
        center = (self.grid_size - 1) / 2
        x, y = coords[:, 0] - center, coords[:, 1] - center
        angle = np.radians(self.rotation)
        x_rot = x * np.cos(angle) - y * np.sin(angle)
        y_rot = x * np.sin(angle) + y * np.cos(angle)
        return np.column_stack((x_rot + center, y_rot + center))
    
    def flip_coordinates(self, coords):
        x, y = coords[:, 0], coords[:, 1]
        if self.flip_x:
            x = self.grid_size - 1 - x
        if self.flip_y:
            y = self.grid_size - 1 - y
        return np.column_stack((x, y))

    def rotate_coordinates(self, coords):
        center = (self.grid_size - 1) / 2
        x, y = coords[:, 0] - center, coords[:, 1] - center
        angle = np.radians(self.rotation)
        x_rot = x * np.cos(angle) - y * np.sin(angle)
        y_rot = x * np.sin(angle) + y * np.cos(angle)
        return np.column_stack((x_rot + center, y_rot + center))

    def clip_coordinates(self, coords):
        return np.clip(coords, 0, self.grid_size - 1).astype(int)

    def plot_data(self):
        self.ax.clear()
        X, Y = np.meshgrid(self.x_values, self.y_values)
        self.ax.plot_surface(X, Y, self.elevation_values, cmap='terrain', alpha=0.7)

        print(f"Number of buildings to plot: {len(self.buildings)}")
        for i, building in enumerate(self.buildings):
            corners = np.array(building['corner_points'])
            print(f"Building {i+1} corners: {corners}")
            corners_transformed = self.rotate_coordinates(self.flip_coordinates(corners))
            corners_clipped = self.clip_coordinates(corners_transformed)
            z = np.mean(self.elevation_values[corners_clipped[:, 1], corners_clipped[:, 0]])
            print(f"Building {i+1} z-value: {z}")

            # Create 3D representation of the building
            x = self.x_values[corners_clipped[:, 0]]
            y = self.y_values[corners_clipped[:, 1]]
            z = [z] * len(x)  # Use the same z-value for all corners
            
            # Create the vertices for the building (including roof)
            verts = [list(zip(x, y, z))]
            
            # Create a Poly3DCollection
            poly3d = Poly3DCollection(verts, alpha=0.5, facecolor='red', edgecolor='black')
            self.ax.add_collection3d(poly3d)

        for road in self.roads:
            points = np.array(road['points'])
            points_transformed = self.rotate_coordinates(self.flip_coordinates(points))
            points_clipped = self.clip_coordinates(points_transformed)
            z = self.elevation_values[points_clipped[:, 1], points_clipped[:, 0]]
            self.ax.plot(self.x_values[points_clipped[:, 0]], self.y_values[points_clipped[:, 1]], z, 'g-', linewidth=2)

        self.ax.set_xlabel('Longitude')
        self.ax.set_ylabel('Latitude')
        self.ax.set_zlabel('Elevation')
        self.ax.set_title(f'3D Terrain Visualization (Rotation: {self.rotation}°)')
        self.ax.set_box_aspect((np.ptp(self.x_values), np.ptp(self.y_values), np.ptp(self.elevation_values)*5))

        plt.draw()

    def on_key(self, event):
        if event.key == 'left' or event.key == 'right':
            self.flip_x = not self.flip_x
            print(f"Flip X: {self.flip_x}")
        elif event.key == 'up' or event.key == 'down':
            self.flip_y = not self.flip_y
            print(f"Flip Y: {self.flip_y}")
        elif event.key == 'r':
            self.rotation = (self.rotation + 45) % 360
            print(f"Rotation: {self.rotation}°")
        self.plot_data()

    def show(self):
        self.plot_data()
        plt.show()    

# Usage
if __name__ == '__main__':
    json_file_path = 'area3.json'
    visualizer = InteractiveVisualizer(json_file_path)
    visualizer.show()

# # Update the ElevationAPI class
# class ElevationAPI(Resource):
#     def get(self):
#         api_key = request.headers.get('x-api-key')
#         if not api_key or api_key not in VALID_API_KEYS:
#             abort(401, description="Invalid or missing API key")

#         address = request.args.get('address')
#         expansion = int(request.args.get('expansion', 1))  # Default to 1 if not specified
#         if expansion not in [1, 2, 3]:
#             abort(400, description="Expansion must be 1, 2, or 3")
#         result = main(address, expansion, use_flask=True)
#         if result:
#             return jsonify(result)
#         else:
#             return jsonify({"error": "Failed to process the request"}), 500

# if __name__ == '__main__':
#     import sys
#     import threading
#     import time

#     def run_with_timeout(func, args, timeout):
#         result = [None]
#         def worker():
#             result[0] = func(*args)
#         thread = threading.Thread(target=worker)
#         thread.start()
#         thread.join(timeout)
#         if thread.is_alive():
#             print(f"Function execution timed out after {timeout} seconds.")
#             return None
#         return result[0]

#     try:
#         import google.colab
#         in_colab = True
#     except ImportError:
#         in_colab = False

#     if in_colab:
#         address = input("Enter the address: ")
#         expansion = int(input("Enter expansion factor (1, 2, or 3): "))
#     else:
#         if len(sys.argv) < 2:
#             address = input("Enter the address: ")
#         else:
#             address = ' '.join(sys.argv[1:-1])  # Join all arguments except the last one as the address
        
#         if len(sys.argv) < 3:
#             expansion = int(input("Enter expansion factor (1, 2, or 3): "))
#         else:
#             expansion = int(sys.argv[-1])  # Use the last argument as the expansion factor

#     # Validate expansion input
#     while expansion not in [1, 2, 3]:
#         print("Invalid expansion factor. Please enter 1, 2, or 3.")
#         expansion = int(input("Enter expansion factor (1, 2, or 3): "))

#     print("Starting processing...")
#     start_time = time.time()
#     result = run_with_timeout(main, (address, expansion), timeout=300)  # 5 minutes timeout
#     end_time = time.time()

#     if result:
#         print(f"Processing completed successfully in {end_time - start_time:.2f} seconds.")
#         print("Terrain data has been saved.")
        
#         # Offer to display terrain plot
#         display_option = input("Do you want to display the terrain plot now? (y/n): ").lower()
#         if display_option == 'y':
#             display_terrain('/Users/morgandixon/Desktop/Cordoba/terrain_data.json')
#     else:
#         print("Processing failed or timed out.")