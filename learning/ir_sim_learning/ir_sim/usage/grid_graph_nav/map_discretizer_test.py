import numpy as np
from shapely.geometry import Polygon, Point

from ir_sim.env import EnvBase


def gen_grid_points(env_range, obstacle_list, grid_size=1):
    """
    Discretize a map to grid representation and return grid points
    :param: env_range, obstacle_list, grid_size, grid_num
    :return: coordinates of grid points
    """
    env_x_range = env_range[0]
    env_y_range = env_range[1]
    grid_num_x = int((env_x_range[1] - env_x_range[0] - 2 * grid_size) / grid_size + 1)
    grid_num_y = int((env_y_range[1] - env_y_range[0] - 2 * grid_size) / grid_size + 1)

    x = np.linspace(env_x_range[0] + grid_size, env_x_range[1] - grid_size, grid_num_x)
    y = np.linspace(env_y_range[0] + grid_size, env_y_range[1] - grid_size, grid_num_y)
    t1, t2 = np.meshgrid(x, y)
    points = np.vstack([t1.T.ravel(), t2.T.ravel()]).T

    collision_points = []
    for i, point in enumerate(points):
        for obs in obstacle_list:
            obs_polygon = Polygon(obs)
            obs_inflated = obs_polygon.buffer(0.2)
            if obs_inflated.contains(Point(point)):
                collision_points.append(i)

    points = np.delete(points, collision_points, axis=0)

    return points


env = EnvBase('dense_map.yaml', control_mode='auto', init_args={'no_axis': False}, collision_mode='react',
              save_ani=False, full=False)
env_range = [env.world.x_range, env.world.y_range]
obs_list = [np.array(obs.vertex).T for obs in env.obstacle_list]
grid_points = gen_grid_points(env_range, obs_list, grid_size=1)

env.ax.scatter(grid_points[:, 0], grid_points[:, 1], color='lightgrey')

for i in range(200):
    env.render()