import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point


class MapDiscretizer:
    def __init__(self, map_shape, obstacle_polygons):
        self.map_shape = map_shape
        self.obstacle_polygons = obstacle_polygons

    def discretize_map(self, grid_size):
        min_x, min_y, max_x, max_y = self.map_shape
        x_points = np.arange(min_x, max_x, grid_size)
        y_points = np.arange(min_y, max_y, grid_size)

        grid = np.zeros((len(y_points), len(x_points)), dtype=int)

        for i, x in enumerate(x_points):
            for j, y in enumerate(y_points):
                point = (x + grid_size / 2, y + grid_size / 2)  # Use the center of each grid cell
                for obstacle_polygon in self.obstacle_polygons:
                    if obstacle_polygon.contains(Point(point)):
                        grid[j, i] = 1  # Set to 1 if point is inside an obstacle

        return grid


# 示例使用
obstacle_polygons = [Polygon([(2, 2), (4, 6), (7, 8), (9, 5), (6, 1)])]
map_shape = (0, 0, 10, 10)
grid_size = 0.5

discretizer = MapDiscretizer(map_shape, obstacle_polygons)
grid_map = discretizer.discretize_map(grid_size)

print(grid_map)

# 绘制原始多边形地图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# 绘制原始多边形
for obstacle_polygon in obstacle_polygons:
    x, y = obstacle_polygon.exterior.xy
    ax1.plot(x, y, color='red')

ax1.set_title('Original Map with Obstacles')
ax1.set_xlabel('X-axis')
ax1.set_ylabel('Y-axis')

# 绘制离散化后的网格地图
ax2.imshow(grid_map, cmap='binary', origin='lower', extent=(map_shape[0], map_shape[2], map_shape[1], map_shape[3]))
ax2.set_title('Discretized Map with Obstacles')
ax2.set_xlabel('X-axis')
ax2.set_ylabel('Y-axis')

plt.show()

