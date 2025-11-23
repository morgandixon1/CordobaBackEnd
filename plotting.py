import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import json

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

        self.fig = plt.figure(figsize=(15, 15))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

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

        for building in self.buildings:
            corners = np.array(building['corner_points'])
            corners_transformed = self.rotate_coordinates(self.flip_coordinates(corners))
            corners_clipped = self.clip_coordinates(corners_transformed)
            z = np.mean(self.elevation_values[corners_clipped[:, 1], corners_clipped[:, 0]])
            self.ax.plot(self.x_values[corners_clipped[:, 0]], self.y_values[corners_clipped[:, 1]], [z]*len(corners_clipped), 'r-')

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

def visualize_terrain_data(json_file_path):
    visualizer = InteractiveVisualizer(json_file_path)
    visualizer.show()

# Usage
if __name__ == '__main__':
    visualize_terrain_data('area3.json')