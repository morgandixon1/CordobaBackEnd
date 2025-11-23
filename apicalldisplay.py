from flask import Flask, render_template, request, jsonify
import requests
import json
import numpy as np
from sklearn.decomposition import PCA
import requests
from PIL import Image
from io import BytesIO
import rasterio
from scipy.interpolate import RegularGridInterpolator
import geocoder
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import itertools
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import label
from scipy.spatial.distance import cdist
import matplotlib.patches as mpatches
from scipy import interpolate  # Add this import statement
from scipy.spatial import ConvexHull
from skimage.morphology import remove_small_holes, remove_small_objects, binary_opening, binary_closing, disk, skeletonize
from shapely.geometry import LineString
from math import atan2, degrees
from rdp import rdp
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__, static_folder='static')
api_key_opentopo = os.getenv('OPENTOPO_API_KEY')
api_key_mapbox = os.getenv('MAPBOX_ACCESS_TOKEN')

buffer = 0.001
size = 0.001

class DataCollector:
    def __init__(self, address, api_key_mapbox, api_key_opentopo):
        self.address = address
        self.api_key_mapbox = api_key_mapbox
        self.api_key_opentopo = api_key_opentopo
        self.latitude, self.longitude = self.geocode_address()
        self.image_width = 800
        self.image_height = 800
        self.bounds = None
        self.elevation_data = None
        self.image = None  # Initialize self.image to None
        self.buildings = None
        self.roads = None
        self.freeways = None
        self.highways = None

    def geocode_address(self):
        print("geocode_address function called")
        g = geocoder.mapbox(self.address, key=self.api_key_mapbox)
        return g.latlng

    def get_satellite_image(self):
        print("get_satellite_image function called")
        url = f"https://api.mapbox.com/styles/v1/morgandixon/cltqpt1un01ij01ptc5b11pwv/static/{self.longitude},{self.latitude},18/800x800@2x?access_token={self.api_key_mapbox}&png=true"
        response = requests.get(url)

        if response.status_code == 200:
            try:
                image_np = np.array(Image.open(BytesIO(response.content)))
                if len(image_np.shape) == 3 and image_np.shape[2] >= 3:
                    # Assumes image has at least 3 channels, remove alpha if present
                    self.image = image_np[:, :, :3]  # Assign the image data to self.image
                else:
                    # Convert grayscale to RGB
                    self.image = np.stack((image_np,)*3, axis=-1)  # Assign the image data to self.image
                return self.image  # Return the image data
            except Exception as e:
                print(f"Error opening image: {str(e)}")
                print(f"Response content: {response.content}")
                return None
        else:
            print(f"Error retrieving satellite image. Status code: {response.status_code}")
            print(f"Response content: {response.content}")
            return None
    
    def load_and_preprocess_image(self, png_image, target_colors, tolerance=20):
        print("load_and_preprocess_image function called")
        
        # Convert the image to a 3-channel RGB array
        img_array = png_image[:, :, :3] if png_image.shape[2] >= 3 else np.dstack((png_image,) * 3)
        
        masks = {}
        for color_name, target_color in target_colors.items():
            color_diff = np.sqrt(np.sum((img_array - target_color) ** 2, axis=2))
            mask = color_diff <= tolerance
            filled_mask = remove_small_holes(mask, area_threshold=100, connectivity=2)
            cleaned_mask = remove_small_objects(filled_mask.astype(int), min_size=1000)
            closed_mask = binary_closing(cleaned_mask, disk(1))
            opened_mask = binary_opening(closed_mask, disk(1))
            skeleton = skeletonize(opened_mask)
            masks[color_name] = (opened_mask, skeleton)

        return masks
    
    def get_skeleton_sections(self, skeleton):
        print("get_skeleton_sections function called")
        cross_points, end_points = [], []
        for x, y in zip(*np.where(skeleton)):
            neighborhood = skeleton[x - 1:x + 2, y - 1:y + 2]
            if np.sum(neighborhood) > 3:
                cross_points.append((x, y))
            elif np.sum(neighborhood) == 2:
                end_points.append((x, y))
        
        cut_points = set(cross_points + end_points)
        cut_mask = np.zeros_like(skeleton, dtype=bool)
        cut_mask[tuple(zip(*cut_points))] = True
        cleaned_skeleton = skeleton & ~cut_mask
        labeled_mask, num_labels = label(cleaned_skeleton)
        skeleton_sections = [np.transpose(np.nonzero(labeled_mask == i)) for i in range(1, num_labels + 1)]
        return skeleton_sections, cross_points, end_points

    def calculate_angle(self, p1, p2, p3):
        print("calculate_angle function called")
        v1 = np.array(p1) - np.array(p2)
        v2 = np.array(p3) - np.array(p2)
        angle_rad = atan2(np.linalg.det([v1, v2]), np.dot(v1, v2))
        return abs(degrees(angle_rad))

    def calculate_slope(self, p1, p2):
        print("calculate_slope function called")
        if p1[0] == p2[0]:
            return float('inf')
        return (p2[1] - p1[1]) / (p2[0] - p1[0])

    def remove_similar_slopes(self, points, slope_threshold=0.1):
        print("remove_similar_slopes function called")
        if len(points) < 3:
            return points

        simplified_points = [points[0]]
        prev_slope = self.calculate_slope(points[0], points[1])
        for i in range(1, len(points) - 1):
            curr_slope = self.calculate_slope(points[i], points[i + 1])
            if abs(curr_slope - prev_slope) > slope_threshold:
                simplified_points.append(points[i])
                prev_slope = curr_slope

        simplified_points.append(points[-1])
        return simplified_points

    def simplify_sections(self, original_sections, epsilon=0.5, angle_threshold=5, min_distance=2.0, max_angle_diff=5, slope_threshold=0.1, rdp_tolerance=1.0):
        print("simplify_sections function called")
        simplified_sections = []
        for orig_section in original_sections:
            if len(orig_section) < 3:
                continue

            filtered_points = [orig_section[0]]
            for i in range(1, len(orig_section) - 1):
                angle = self.calculate_angle(orig_section[i - 1], orig_section[i], orig_section[i + 1])
                if angle > angle_threshold:
                    filtered_points.append(orig_section[i])
            filtered_points.append(orig_section[-1])
            line = LineString(filtered_points)
            simplified_line = line.simplify(epsilon, preserve_topology=False)
            simplified_points = np.array(simplified_line.coords).astype(int)
            final_points = [simplified_points[0]]
            prev_angle = None
            for i in range(1, len(simplified_points)):
                curr_point = simplified_points[i]
                prev_point = final_points[-1]
                if np.linalg.norm(curr_point - prev_point) >= min_distance:
                    if i < len(simplified_points) - 1:
                        next_point = simplified_points[i + 1]
                        curr_angle = self.calculate_angle(prev_point, curr_point, next_point)
                        
                        if prev_angle is None or abs(curr_angle - prev_angle) > max_angle_diff:
                            final_points.append(curr_point)
                            prev_angle = curr_angle
                    else:
                        final_points.append(curr_point)

            final_points = self.remove_similar_slopes(final_points, slope_threshold)
            rdp_points = rdp(final_points, epsilon=rdp_tolerance)
            simplified_sections.append(rdp_points)
        return simplified_sections

    def plot_skeleton_sections(self, road_mask, skeleton_sections, simplified_sections, cross_points, end_points, ax, color):
        print("plot_skeleton_sections function called")
        ax.imshow(road_mask, cmap='gray', alpha=0.5)
        for i, section in enumerate(skeleton_sections):
            ax.scatter(section[:, 1], section[:, 0], color=color, marker='.', s=1, label=f'Section {i+1}')
        for i, section in enumerate(simplified_sections):
            section_array = np.array(section)  # Convert section to a numpy array
            ax.scatter(section_array[:, 1], section_array[:, 0], color=color, marker='o', s=10, label=f'Simplified Section {i+1}')
        if cross_points:
            cross_points_arr = np.array(cross_points)
            ax.scatter(cross_points_arr[:, 1], cross_points_arr[:, 0], color='black', marker='o', s=10, label='Cross Points')
        if end_points:
            end_points_arr = np.array(end_points)
            ax.scatter(end_points_arr[:, 1], end_points_arr[:, 0], color='white', marker='o', s=10, label='End Points')

    def process_buildings(self, image, building_rgb, hue_tolerance=8, sat_tolerance=40, val_tolerance=40):
        print("process_buildings function called")
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        building_hsv = cv2.cvtColor(np.uint8([[building_rgb]]), cv2.COLOR_RGB2HSV)[0][0]

        lower_color = np.array([building_hsv[0] - hue_tolerance, max(building_hsv[1] - sat_tolerance, 0), max(building_hsv[2] - val_tolerance, 0)])
        upper_color = np.array([building_hsv[0] + hue_tolerance, min(building_hsv[1] + sat_tolerance, 255), min(building_hsv[2] + val_tolerance, 255)])

        mask = cv2.inRange(hsv_image, lower_color, upper_color)

        kernel_size = (5, 5) if mask.size < 50000 else (7, 7)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        buildings = []
        for contour in contours:
            if cv2.contourArea(contour) < 500:  # Filter out small noise contours
                continue

            factor = 0.003 if cv2.contourArea(contour) < 5000 else 0.007
            approx_polygon = cv2.approxPolyDP(contour, factor * cv2.arcLength(contour, True), True)
            polygon_points = approx_polygon.reshape(-1, 2)

            rect = cv2.minAreaRect(polygon_points)
            centroid = rect[0]
            rotation = rect[2]

            building = {
                'polygon_points': polygon_points,
                'centroid': centroid,
                'rotation': rotation
            }
            buildings.append(building)

        return buildings

    def collect_data(self):
        print("collect_data function called")

        # Retrieve the satellite image
        png_image = self.get_satellite_image()

        if png_image is None:
            print("Satellite image retrieval failed. Skipping data collection.")
            return {
                "buildings": [],
                "roads": [],
                "freeways": [],
                "highways": []
            }

        # Preprocess the image
        target_colors = {
            "roads": (255, 255, 255),
            "freeways": (255, 179, 102),
            "highways": (247, 224, 110)
        }
        road_masks = self.load_and_preprocess_image(png_image, target_colors)

        # Process buildings and roads
        building_rgb = [173, 66, 10]  # RGB color #AD420A
        self.buildings = self.process_buildings(png_image, building_rgb)
        self.roads = {}
        self.freeways = {}
        self.highways = {}

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(self.image)

        for road_type, (road_mask, skeleton) in road_masks.items():
            skeleton_sections, cross_points, end_points = self.get_skeleton_sections(skeleton)
            simplified_sections = self.simplify_sections(skeleton_sections, epsilon=0.5, angle_threshold=5)

            if road_type == "roads":
                self.roads = {f"road_{i+1}": {"points": section.tolist()} for i, section in enumerate(simplified_sections)}
                color = 'blue'
            elif road_type == "freeways":
                self.freeways = {f"freeway_{i+1}": {"points": section.tolist()} for i, section in enumerate(simplified_sections)}
                color = 'green'
            elif road_type == "highways":
                self.highways = {f"highway_{i+1}": {"points": section.tolist()} for i, section in enumerate(simplified_sections)}
                color = 'red'

            for section in skeleton_sections:
                section_array = np.array(section)
                ax.plot(section_array[:, 1], section_array[:, 0], color=color, linestyle='-', linewidth=1, alpha=0.7, label=f"{road_type} Skeleton")

            for section in simplified_sections:
                section_array = np.array(section)
                ax.plot(section_array[:, 1], section_array[:, 0], color=color, linestyle='-', linewidth=2, label=f"{road_type} Simplified")

            if cross_points:
                cross_points_arr = np.array(cross_points)
                ax.scatter(cross_points_arr[:, 1], cross_points_arr[:, 0], color='black', marker='o', s=30, label=f"{road_type} Cross Points")

            if end_points:
                end_points_arr = np.array(end_points)
                ax.scatter(end_points_arr[:, 1], end_points_arr[:, 0], color='white', marker='o', s=30, label=f"{road_type} End Points")

        for building in self.buildings:
            polygon_points = building['polygon_points']
            centroid = building['centroid']
            
            # Connect the first and last points to close the polygon
            polygon_points = np.vstack((polygon_points, polygon_points[0]))
            
            ax.plot(polygon_points[:, 0], polygon_points[:, 1], color='orange', linewidth=2, label='Buildings')
            ax.scatter(centroid[0], centroid[1], color='red', s=50, marker='o', label='Building Centroids')

        ax.set_title('Buildings, Roads, Skeletons, and Intersections')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = list(set(labels))
        unique_handles = [handles[labels.index(label)] for label in unique_labels]
        ax.legend(unique_handles, unique_labels)
        plt.tight_layout()
        plt.show()

        return {
            "buildings": self.buildings,
            "roads": self.roads,
            "freeways": self.freeways,
            "highways": self.highways
        }
            
    def collect_elevation(self):
        params = {
            'demtype': 'SRTMGL1',
            'south': self.latitude - buffer,
            'north': self.latitude + buffer,
            'west': self.longitude - buffer,
            'east': self.longitude + buffer,
            'outputFormat': 'gtiff',
            'API_Key': self.api_key_opentopo
        }
        api_endpoint = 'https://portal.opentopography.org/API/globaldem'
        response = requests.get(api_endpoint, params=params)
        if response.status_code == 200:
            dem_data = BytesIO(response.content)
            with rasterio.open(dem_data) as dataset:
                self.elevation_data = dataset.read(1)
                self.bounds = dataset.bounds
                print("Elevation data retrieved successfully.")
                print(f"Elevation data shape: {self.elevation_data.shape}")
                print(f"Bounds: {self.bounds}")
                return self.elevation_data, self.bounds
        else:
            print("Failed to retrieve elevation data.")
            print(f"Response status code: {response.status_code}")
            return None, None
        
class Assembly:
    def __init__(self, labeled_objects, elevation_data, bounds):
        self.image_width = 800
        self.image_height = 800
        self.labeled_objects = labeled_objects
        self.elevation_data = elevation_data
        self.bounds = bounds

    def interpolate_elevation(self, xs, ys):
        x_edges = np.linspace(self.bounds.left, self.bounds.right, self.elevation_data.shape[1])
        y_edges = np.linspace(self.bounds.bottom, self.bounds.top, self.elevation_data.shape[0])
        xs = np.clip(xs, x_edges[0], x_edges[-1])
        ys = np.clip(ys, y_edges[0], y_edges[-1])
        valid_mask = np.isfinite(self.elevation_data)
        masked_elevation_data = np.ma.masked_array(self.elevation_data, mask=~valid_mask)
        interpolating_function = RegularGridInterpolator((y_edges, x_edges), masked_elevation_data, bounds_error=False, fill_value=None)
        pts = np.array(list(zip(ys, xs)))
        z_coords = interpolating_function(pts)
        z_coords = np.where(np.isnan(z_coords), 0, z_coords)
        return z_coords.tolist()

    def combine_data(self):
        combined_data = {
            "buildings": [],
            "roads": [],
            "freeways": [],
            "highways": [],
            "elevation": []
        }

        # Add elevation data
        x_coords, y_coords = np.meshgrid(
            np.linspace(self.bounds.left, self.bounds.right, self.elevation_data.shape[1]),
            np.linspace(self.bounds.bottom, self.bounds.top, self.elevation_data.shape[0])
        )
        x_coords = x_coords.flatten()
        y_coords = y_coords.flatten()
        z_coords = self.elevation_data.flatten().astype(float)  # Convert elevation data to float
        combined_data["elevation"] = list(zip(x_coords, y_coords, z_coords))

        for building in self.labeled_objects["buildings"]:
            xs = building["polygon_points"][:, 0].tolist()
            ys = building["polygon_points"][:, 1].tolist()
            zs = self.interpolate_elevation(xs, ys)
            combined_data["buildings"].append({
                "points": list(zip(xs, ys, zs)),
                "centroid": building["centroid"],
                "rotation": building["rotation"]
            })

        for road_type in ["roads", "freeways", "highways"]:
            for road_id, road_data in self.labeled_objects[road_type].items():
                xs, ys = zip(*road_data["points"])
                zs = self.interpolate_elevation(xs, ys)
                combined_data[road_type].append({
                    "id": road_id,
                    "points": list(zip(xs, ys, zs))
                })

        return combined_data

@app.route('/', methods=['GET'])
def index():
    print("Index route called")
    return render_template('frontend.html')

@app.route('/layerassembly', methods=['POST'])
def layerassembly():
    print("Layerassembly route called")
    address = request.form.get('address', "1600 Pennsylvania Avenue NW, Washington, DC")
    data_collector = DataCollector(address, api_key_mapbox, api_key_opentopo)
    collected_data = data_collector.collect_data()
    elevation_data, bounds = data_collector.collect_elevation()

    assembly = Assembly(collected_data, elevation_data, bounds)
    combined_data = assembly.combine_data()

    print("Combined data sent to the frontend:")
    print(json.dumps(combined_data, indent=2))

    return jsonify(combined_data)

if __name__ == '__main__':
    # Only enable debug mode if explicitly set in environment
    debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(debug=debug_mode)


# @app.route('/generate-voxel', methods=['POST'])
# def generate_voxel():
#     text_input = request.form['text']
#     print("Received text input:", text_input)
#     if text_input == '1' or text_input.lower() == 'test':
#         print("Loading test voxel grid...")
#         with open('voxelgridtest.json') as f:
#             voxel_grid = np.array(json.load(f))
#             voxel_grid = np.where(voxel_grid, 1, 0)
#         print("Test voxel grid loaded. Shape:", voxel_grid.shape)
#         print("Test voxel grid data type:", type(voxel_grid))
#         print("Test voxel grid sample:")
#         print(voxel_grid[:5, :5, :5])
#     else:
#         url = 'http://18.226.164.139/generate_output'
#         payload = {"text": text_input, "output_format": "voxel_grid", "angle": 90}
#         print("Calling API...")
#         try:
#             response = requests.post(url, json=payload, timeout=5)
#         except requests.Timeout:
#             print("Request timed out.")
#             return jsonify({'error': 'Request timed out'}), 500

#         if response.status_code == 200:
#             print("Data received successfully.")
#             data = response.json()
#             print("API response data type:", type(data))
#             print("API response keys:", data.keys())
#             voxel_grid = np.array(data['voxel_grid'])
#             voxel_grid = np.where(voxel_grid, 1, 0)
#             print("API voxel grid loaded. Shape:", voxel_grid.shape)
#             print("API voxel grid data type:", type(voxel_grid))
#             print("API voxel grid sample:")
#             print(voxel_grid[:5, :5, :5])
#         else:
#             print("Failed to get voxel grid. Status code:", response.status_code)
#             return jsonify({'error': 'Failed to get voxel grid'}), 500

#     filled = np.argwhere(voxel_grid)
#     print("Filled voxels shape:", filled.shape)
#     print("Filled voxels sample:")
#     print(filled[:5])

#     pca = PCA(n_components=3)
#     filled_pca = pca.fit_transform(filled)
#     print("PCA transformed data shape:", filled_pca.shape)
#     print("PCA transformed data sample:")
#     print(filled_pca[:5])

#     filled_pca -= filled_pca.min(axis=0)
#     points = filled_pca.tolist()
#     print("Points data type:", type(points))
#     print("Points sample:")
#     print(points[:5])
#     return jsonify(voxel_grid=voxel_grid.tolist())