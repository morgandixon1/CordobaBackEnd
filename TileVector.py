import requests
import matplotlib.pyplot as plt
from shapely.geometry import shape, Polygon, LineString, MultiPolygon
from mapbox_vector_tile import decode
import mercantile
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def fetch_mapbox_vector_tile(lon, lat, zoom, access_token):
    tileset_id = 'mapbox.mapbox-streets-v8'

    # Convert lon, lat to tile coordinates
    tile = mercantile.tile(lon, lat, zoom)
    tile_url = f"https://api.mapbox.com/v4/{tileset_id}/{tile.z}/{tile.x}/{tile.y}.mvt?access_token={access_token}"
    
    response = requests.get(tile_url)
    response.raise_for_status()
    tile_data = response.content

    # Decode the vector tile
    decoded_tile = decode(tile_data)
    return decoded_tile

def geocode_address(address, access_token):
    geocoding_url = f"https://api.mapbox.com/geocoding/v5/mapbox.places/{address}.json?access_token={access_token}"
    response = requests.get(geocoding_url)
    response.raise_for_status()
    data = response.json()
    if data['features']:
        return data['features'][0]['center']
    else:
        raise Exception("Address not found")

def main():
    access_token = os.getenv('MAPBOX_ACCESS_TOKEN')
    if not access_token:
        print("Error: MAPBOX_ACCESS_TOKEN not found. Please set it in .env file")
        return

    zoom = 17

    address = input("Enter an address: ")
    try:
        lon, lat = geocode_address(address, access_token)
    except Exception as e:
        print(f"Error: {e}")
        return

    # Fetch vector tile data
    vector_data = fetch_mapbox_vector_tile(lon, lat, zoom, access_token)
    
    buildings = []
    roads = []

    for layer_name, layer in vector_data.items():
        for feature in layer['features']:
            geom = shape(feature['geometry'])
            properties = feature.get('properties', {})
            layer_class = properties.get('class', None)
            if layer_class == 'building':
                buildings.append(geom)
            elif layer_class in ['road', 'street', 'highway']:
                roads.append(geom)

    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_aspect('equal')

    # Plot roads
    for road in roads:
        if isinstance(road, LineString):
            x, y = road.xy
            ax.plot(x, y, color='blue', linewidth=2)

    # Plot buildings
    for building in buildings:
        if isinstance(building, Polygon) or isinstance(building, MultiPolygon):
            if isinstance(building, Polygon):
                polygons = [building]
            else:
                polygons = building.geoms
            
            for polygon in polygons:
                x, y = polygon.exterior.xy
                ax.plot(x, y, color='red', linewidth=2)
                ax.fill(x, y, color='red', alpha=0.5)
                # Plot building center
                center = polygon.centroid
                ax.scatter(center.x, center.y, color='green', s=100)

    ax.set_title('Buildings and Roads from Mapbox Vector Tiles')
    ax.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
