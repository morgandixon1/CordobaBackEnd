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
import os
import rasterio
from scipy.interpolate import griddata
import pyproj
from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

def collect_elevation(latitude, longitude, buffer, api_key_opentopo, num_points=100):
    import pyproj
    import numpy as np
    import requests

    # Setting up the projection
    proj = pyproj.Proj(proj='utm', zone=10, ellps='WGS84', preserve_units=False)
    latitudes = np.linspace(latitude - buffer, latitude + buffer, int(np.sqrt(num_points)))
    longitudes = np.linspace(longitude - buffer, longitude + buffer, int(np.sqrt(num_points)))
    locations = [(lat, lon) for lat in latitudes for lon in longitudes]
    location_str = '|'.join([f"{lat},{lon}" for lat, lon in locations])

    # API call
    url = 'https://api.opentopodata.org/v1/ned10m'
    params = {'locations': location_str, 'api_key': api_key_opentopo}
    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        if 'results' in data:
            elevation_points = [result['elevation'] for result in data['results'] if 'elevation' in result]
            utm_coords = [proj(lon, lat) for lat, lon in locations]
            x_vals, y_vals = zip(*utm_coords)
            min_x, max_x = min(x_vals), max(x_vals)
            min_y, max_y = min(y_vals), max(y_vals)
            
            # Convert elevation to meters
            elevation_points_meters = [point * 0.3048 for point in elevation_points]

            # Scale coordinates
            scale_x = 1000 / (max_x - min_x)
            scale_y = 1021 / (max_y - min_y)
            scaled_x = [scale_x * (x - min_x) for x in x_vals]
            scaled_y = [scale_y * (y - min_y) for y in y_vals]

            # Create (x, y, z) plot points
            plot_points = [(x, y, z) for x, y, z in zip(scaled_x, scaled_y, elevation_points_meters)]

            return elevation_points_meters, plot_points
        else:
            print("No elevation results found in the response.")
    else:
        print(f"Failed to retrieve elevation data. Response Code: {response.status_code}, Content: {response.text}")

    return None, None
    
def fetch_mapbox_image(style, access_token, lon, lat, zoom=17, width=1024, height=1024):
    url = f"https://api.mapbox.com/styles/v1/{style}/static/{lon},{lat},{zoom}/{width}x{height}?access_token={access_token}"
    response = requests.get(url)
    response.raise_for_status()
    if 'image' not in response.headers['Content-Type']:
        raise Exception(f"Unexpected content type: {response.headers['Content-Type']}")
    print("Image collected")
    return Image.open(BytesIO(response.content)).convert('RGB')

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

def get_skeleton_sections(skeleton):
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
    labeled_mask, num_labels = label(cleaned_skeleton, return_num=True)
    skeleton_sections = [np.transpose(np.nonzero(labeled_mask == i)) for i in range(1, num_labels + 1)]
    return skeleton_sections, cross_points, end_points

def calculate_angle(p1, p2, p3):
    v1, v2 = np.array(p1) - np.array(p2), np.array(p3) - np.array(p2)
    angle_rad = atan2(np.linalg.det([v1, v2]), np.dot(v1, v2))
    return abs(degrees(angle_rad))

def simplify_sections(original_sections, epsilon=0.5, angle_threshold=5, min_distance=2.0, slope_threshold=0.1, rdp_tolerance=1.0):
    simplified_sections = []
    for orig_section in original_sections:
        if len(orig_section) < 3:
            continue
        filtered_points = [orig_section[0]] + [
            orig_section[i] for i in range(1, len(orig_section) - 1)
            if calculate_angle(orig_section[i - 1], orig_section[i], orig_section[i + 1]) > angle_threshold
        ] + [orig_section[-1]]

        simplified_line = LineString(filtered_points).simplify(epsilon, preserve_topology=False)
        simplified_points = np.array(simplified_line.coords).astype(int)

        if len(simplified_points) > 1:
            distances = np.linalg.norm(np.diff(simplified_points, axis=0), axis=1)
            mask = np.concatenate(([True], distances >= min_distance))
            final_points = simplified_points[mask]
        else:
            final_points = simplified_points

        final_points = rdp(final_points, epsilon=rdp_tolerance)
        
        ordered_points = [final_points[0]]
        remaining_points = set(map(tuple, final_points[1:]))
        while remaining_points:
            current_point = ordered_points[-1]
            nearest_point = min(remaining_points, key=lambda x: distance.euclidean(current_point, x))
            ordered_points.append(nearest_point)
            remaining_points.remove(nearest_point)
        
        simplified_sections.append(ordered_points)
    return simplified_sections

# Example function to calculate bounds from road and building data
def calculate_bounds(buildings, roads):
    all_points = [point for building in buildings for point in building['polygon_points']]
    all_points += [point for road in roads for point in road['points']]
    if not all_points:
        return None

    min_x = min(point[0] for point in all_points)
    max_x = max(point[0] for point in all_points)
    min_y = min(point[1] for point in all_points)
    max_y = max(point[1] for point in all_points)

    return min_x, max_x, min_y, max_y

def process_buildings(image, building_rgb, hue_tolerance=10, sat_tolerance=50, val_tolerance=50, area_threshold=500):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    building_hsv = cv2.cvtColor(np.uint8([[building_rgb]]), cv2.COLOR_RGB2HSV)[0][0]
    lower_color = np.array([building_hsv[0] - hue_tolerance, max(building_hsv[1] - sat_tolerance, 0), max(building_hsv[2] - val_tolerance, 0)])
    upper_color = np.array([building_hsv[0] + hue_tolerance, min(building_hsv[1] + sat_tolerance, 255), min(building_hsv[2] + val_tolerance, 255)])
    mask = cv2.inRange(hsv_image, lower_color, upper_color)
    kernel_size = (5, 5) if mask.size < 50000 else (7, 7)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    dist_transform = distance_transform_edt(mask)
    local_max = peak_local_max(dist_transform, min_distance=20)

    markers = np.zeros_like(dist_transform, dtype=np.int32)
    for i, (y, x) in enumerate(local_max, start=1):
        markers[y, x] = i
    markers[mask == 0] = 0

    segmented_mask = watershed(dist_transform, markers, mask=mask)

    buildings = []
    for label in np.unique(segmented_mask):
        if label == 0:
            continue
        building_mask = (segmented_mask == label).astype(np.uint8) * 255
        contours, _ = cv2.findContours(building_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(contour)
            if area < area_threshold:
                continue
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            if len(approx) >= 4:
                polygon_points = approx.reshape(-1, 2)
                rect = cv2.minAreaRect(polygon_points)
                building_type = 'Commercial' if area >= area_threshold else 'House'
                buildings.append({
                    'polygon_points': polygon_points,
                    'centroid': rect[0],
                    'rotation': rect[2],
                    'area': area,
                    'building_type': building_type
                })
    return buildings

def plot_buildings(buildings, ax):
    for building in buildings:
        polygon_points = np.vstack((building['polygon_points'], building['polygon_points'][0]))
        ax.plot(polygon_points[:, 0], polygon_points[:, 1], marker='o', linestyle='-', linewidth=2, markersize=5)
        ax.scatter(building['centroid'][0], building['centroid'][1], color='green', s=100)

def print_building_and_road_info(buildings, road_masks, simplified_sections):
    house_count = 1
    commercial_count = 1
    town_house_count = 1
    building_data = []
    road_data = []
    section_labels = {
        'roads': 'Road',
        'freeway': 'Freeway',
        'highway': 'Highway'
    }

    print("Building Information:")
    for building in buildings:
        if building['building_type'] == 'House':
            building_title = f"House {house_count}"
            house_count += 1
        elif building['building_type'] == 'Commercial':
            building_title = f"Commercial Building {commercial_count}"
            commercial_count += 1
        else:  # Town House
            building_title = f"Town House {town_house_count}"
            town_house_count += 1

        # print(f"{building_title}:")
        # print(f"  Center Point: ({building['centroid'][0]:.2f}, {building['centroid'][1]:.2f})")
        # print(f"  Rotation: {building['rotation']:.2f} degrees")
        # print("  Corner Points:")
        # for corner in building['polygon_points']:
        #     print(f"    ({corner[0]:.2f}, {corner[1]:.2f})")
        # print()

        building_info = {
            'title': building_title,
            'center_point': [float(building['centroid'][0]), float(building['centroid'][1])],
            'rotation': float(building['rotation']),
            'corner_points': [[float(corner[0]), float(corner[1])] for corner in building['polygon_points']]
        }
        building_data.append(building_info)

    print("Road Information:")

    for road_type, (road_mask, skeleton) in road_masks.items():
        section_label = section_labels.get(road_type, 'Road')
        #print(f"Simplified Sections for {section_label}:")

        for idx, section in enumerate(simplified_sections[road_type], start=1):
            # print(f"{section_label} Section {idx}:")
            # for point in section:
            #     print(f"  ({point[0]}, {point[1]})")
            # print()

            section_info = {
                'title': f"{section_label} Section {idx}",
                'points': [[int(point[0]), int(point[1])] for point in section]
            }
            road_data.append(section_info)

    return building_data, road_data

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

def plot_3d_terrain(latitude, longitude, elevation_data, buildings, road_data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    num_points = int(np.sqrt(len(elevation_data)))
    latitudes = np.linspace(latitude - 0.0015, latitude + 0.0015, num_points)
    longitudes = np.linspace(longitude - 0.0015, longitude + 0.0015, num_points)
    lat_grid, lon_grid = np.meshgrid(latitudes, longitudes)
    elevation_grid = np.array(elevation_data).reshape(num_points, num_points)
    ax.plot_surface(lon_grid, lat_grid, elevation_grid, cmap='terrain', alpha=0.7)
    points = np.column_stack((lon_grid.ravel(), lat_grid.ravel()))
    values = elevation_grid.ravel()
    for building in buildings:
        poly_points = np.array(building['polygon_points'])
        z = griddata(points, values, (poly_points[:, 0], poly_points[:, 1]), method='linear')
        ax.plot(poly_points[:, 0], poly_points[:, 1], z, color='red')
    for road in road_data:
        road_points = np.array(road['points'])
        road_lats = road_points[:, 0]
        road_lons = road_points[:, 1]
        road_elevs = griddata(points, values, (road_lats, road_lons), method='linear')
        ax.plot(road_lats, road_lons, road_elevs, color='black')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_zlabel('Elevation')
    plt.show()

app = Flask(__name__)

@app.route('/process', methods=['GET'])
def process():
    address = request.args.get('address', default='1600 Pennsylvania Avenue NW, Washington, DC')
    buffer = 0.0015  # Buffer in degrees to cover approximately 300x300 meters
    api_key_opentopo = os.getenv('OPENTOPO_API_KEY')
    style = os.getenv('MAPBOX_STYLE', 'mapbox/streets-v11')
    mapbox_token = os.getenv('MAPBOX_ACCESS_TOKEN')

    try:
        latitude, longitude = geocode_address(address, mapbox_token)
        print(f"Geocoded address: {address} to Latitude: {latitude}, Longitude: {longitude}")
        
        image = fetch_mapbox_image(style, mapbox_token, longitude, latitude)
        target_colors = {
            'roads': (255, 255, 255),
            'freeway': (255, 179, 102),
            'highway': (247, 224, 110),
            'buildings': (255, 0, 0)
        }
        masks = load_and_preprocess_image(image, target_colors)
        print("Detected road types:", list(masks.keys()))

        img_array = np.array(image)
        building_rgb = [173, 66, 10]
        buildings = process_buildings(img_array, building_rgb)
        
        simplified_sections = {}
        for road_type, (mask, skeleton) in masks.items():
            skeleton_sections, cross_points, end_points = get_skeleton_sections(skeleton)
            simplified_sections[road_type] = simplify_sections(skeleton_sections)
        
        building_data, road_data = print_building_and_road_info(buildings, masks, simplified_sections)
        bounds = calculate_bounds(buildings, road_data)

        if bounds:
            print(f"Calculated bounds: X range ({bounds[0]}, {bounds[1]}), Y range ({bounds[2]}, {bounds[3]})")
        else:
            print("Failed to calculate bounds.")
        
        try:
            elevation_data_meters, plot_points = collect_elevation(latitude, longitude, buffer, api_key_opentopo, num_points=100)
            if elevation_data_meters and plot_points:
                print("Elevation data and plot points retrieved successfully.")
            else:
                print("Failed to retrieve or transform elevation data.")
                return jsonify({"error": "Failed to retrieve or transform elevation data"}), 500

        except Exception as e:
            print(f"Error occurred: {e}")
            return jsonify({"error": str(e)}), 500

        output_data = {
            'elevation_data': plot_points,
            'buildings_data': building_data,
            'road_data': road_data
        }

        return jsonify(output_data)

    except Exception as e:
        print(f"Error occurred: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Ensure the working directory is the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print(f"Current working directory: {os.getcwd()}")
    
    # Print the files in the current directory
    files = os.listdir(os.getcwd())
    print("Files in the current directory:")
    for file in files:
        print(file)

    # Use absolute paths for the SSL certificate files
    cert_file = os.path.join(script_dir, 'cert.pem')
    key_file = os.path.join(script_dir, 'key.pem')
    
    app.run(host='0.0.0.0', port=80, ssl_context=(cert_file, key_file))