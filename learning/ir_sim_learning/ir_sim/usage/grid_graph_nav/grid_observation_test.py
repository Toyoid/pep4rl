import os
import sys
import copy
import numpy as np
from shapely.geometry import Polygon, Point

from ir_sim.env import EnvBase
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../rrt_random_walk")
from rrt_random_walk import DynamicRRW


class GridGraph:
    """
    Class for decision graph in grid representation built from robot's sensor inputs

    Main functions:

    """
    def __init__(self, env):
        self.env_range = [env.world.x_range, env.world.y_range]
        self.obstacle_list = [np.array(obs.vertex).T for obs in env.obstacle_list]

        self.global_unobserved_points = self.gen_grid_points(grid_size=2)
        self.observed_nodes = []

        self.occupancy_map = self.init_partial_map(resolution=0.4)
        self.frontiers = []

    def gen_grid_points(self, grid_size=1):
        """
        Discretize a map to grid representation and return grid points
        :param: env_range, obstacle_list, grid_size, grid_num
        :return: coordinates of grid points
        """
        env_x_range = self.env_range[0]
        env_y_range = self.env_range[1]
        grid_num_x = int((env_x_range[1] - env_x_range[0] - 2 * grid_size) / grid_size + 1)
        grid_num_y = int((env_y_range[1] - env_y_range[0] - 2 * grid_size) / grid_size + 1)

        x = np.linspace(env_x_range[0] + grid_size, env_x_range[1] - grid_size, grid_num_x)
        y = np.linspace(env_y_range[0] + grid_size, env_y_range[1] - grid_size, grid_num_y)
        t1, t2 = np.meshgrid(x, y)
        points_flat = np.vstack([t1.T.ravel(), t2.T.ravel()]).T

        collision_points_idx = []
        for i, point in enumerate(points_flat):
            for obs in self.obstacle_list:
                obs_polygon = Polygon(obs)
                obs_inflated = obs_polygon.buffer(0.2)
                if obs_inflated.contains(Point(point)):
                    collision_points_idx.append(i)

        points_flat = np.delete(points_flat, collision_points_idx, axis=0)

        return points_flat

    def update_observed_nodes(self, sensor_scan):
        """
        Update observed grid node list and unobserved grid node array from sensor scan
        :param sensor_scan: sensor scanning result
        """
        sensor_poly = Polygon(sensor_scan)

        observed_points_idx = []
        for i, point in enumerate(self.global_unobserved_points):
            if sensor_poly.contains(Point(point)):
                self.observed_nodes.append(point)
                observed_points_idx.append(i)

        self.global_unobserved_points = np.delete(self.global_unobserved_points, observed_points_idx, axis=0)

    def init_partial_map(self, resolution=0.2):
        env_x_range = self.env_range[0]
        env_y_range = self.env_range[1]
        grid_num_x = int((env_x_range[1] - env_x_range[0] - 2 * resolution) / resolution + 1)
        grid_num_y = int((env_y_range[1] - env_y_range[0] - 2 * resolution) / resolution + 1)

        x = np.linspace(env_x_range[0] + resolution, env_x_range[1] - resolution, grid_num_x)
        y = np.linspace(env_y_range[0] + resolution, env_y_range[1] - resolution, grid_num_y)
        t1, t2 = np.meshgrid(x, y)

        occup_values = np.ones_like(t1).astype(int)  # unknown pixel: 1
        occupancy_map = np.stack((t1, t2, occup_values), axis=-1)

        return occupancy_map

    def update_frontiers(self, sensor_scan):
        """
        Update partial map and frontiers
        :param sensor_scan: sensor scanning result
        """
        inflation_value = 0.2
        sensor_poly = Polygon(sensor_scan)

        # update obstacle observations
        obs_observation_inflated = []
        for obs in self.obstacle_list:
            poly = Polygon(obs).buffer(inflation_value)
            intersection = sensor_poly.intersection(poly)
            if not intersection.is_empty:
                obs_observation_inflated.append(intersection)

        # update patial map
        grid_num_x = self.occupancy_map.shape[1]
        grid_num_y = self.occupancy_map.shape[0]
        for i in range(grid_num_y):
            for j in range(grid_num_x):
                coordinate = self.occupancy_map[i, j, 0: 2]
                if sensor_poly.contains(Point(coordinate)):
                    self.occupancy_map[i, j, 2] = 0  # free pixel: 0
                    for obs in obs_observation_inflated:
                        if obs.contains(Point(coordinate)):
                            self.occupancy_map[i, j, 2] = 10  # occupied pixel: 10

        # update frontiers
        self.frontiers = []
        for i in range(1, grid_num_y - 1):
            for j in range(1, grid_num_x - 1):
                value = self.occupancy_map[i, j, 2]
                # possible sum values: all free: 0, all occupied: 30, all unknown: 3, unknown exists: sum % 10 != 0
                value_sum_x = np.sum(self.occupancy_map[i, j-1: j+2, 2])
                value_sum_y = np.sum(self.occupancy_map[i-1: i+2, j, 2])
                if value == 0 and (value_sum_x % 10 != 0 or value_sum_y % 10 != 0):
                    self.frontiers.append(self.occupancy_map[i, j, 0:2])

        return obs_observation_inflated  # just for test


if __name__ == "__main__":
    env = EnvBase('dense_map.yaml', control_mode='auto', init_args={'no_axis': False}, collision_mode='react',
                  save_ani=False, full=False)

    p_goal_init = np.squeeze(env.robot.goal[:2])
    p_robot_init = np.squeeze(env.robot.init_state[:2])
    rrw = DynamicRRW(env=env, s_start=p_robot_init, s_goal=p_goal_init, step_len=0.6, iter_max=30)

    grid_graph = GridGraph(env)

    lidar_scan = env.robot.lidar.global_scan_matrix.T

    grid_graph.update_observed_nodes(lidar_scan)
    grid_graph.update_frontiers(lidar_scan)

    rrw.update_sensor_observe(lidar_scan)
    # search initial path with a random temporary goal
    rrw.grow_rrt()

    # initialize s-ahead-robot and s-behind-robot
    rrw.s_ahead_robot = rrw.waypoints[-2]
    rrw.s_ahead_robot.parent = rrw.s_robot
    rrw.s_behind_robot = rrw.s_start
    rrw.s_behind_robot.flag = "INVALID"

    step = 0
    while step < 500:
        step += 1

        waypoint_now = rrw.waypoints[-2] if rrw.waypoints.__len__() > 1 else rrw.s_goal
        env.robot.goal[:2] = copy.deepcopy([[waypoint_now.x], [waypoint_now.y]])
        env.robot.init_goal_state[:2] = copy.deepcopy([[rrw.s_goal.x], [rrw.s_goal.y]])
        # move the robot one step to the current waypoint
        des_vel = env.cal_des_vel()
        env.step(des_vel)

        # update the robot location on the rrt
        rrw.s_robot.x = np.squeeze(env.robot.state)[0]
        rrw.s_robot.y = np.squeeze(env.robot.state)[1]
        # keep track of the robot location in the result path
        rrw.waypoints[-1] = rrw.s_robot
        rrw.path[-1] = (rrw.s_robot.x, rrw.s_robot.y)

        # update observed obstacle list in the rrt and the observed grid graph
        lidar_scan = env.robot.lidar.global_scan_matrix.T
        grid_graph.update_observed_nodes(lidar_scan)
        obs_observation = grid_graph.update_frontiers(lidar_scan)

        rrw.update_sensor_observe(lidar_scan)

        # adjust the rrt according to the updated observations
        rrw.plot_lidar(env.ax)
        rrw.rrt_random_walk()

        grid_points_unobv = env.ax.scatter(grid_graph.global_unobserved_points[:, 0],
                                           grid_graph.global_unobserved_points[:, 1], s=10, c='lightgrey')
        grid_points_unobv.set_zorder(0)
        env.dyna_patch_list.append(grid_points_unobv)
        grid_points_obv = env.ax.scatter([node[0] for node in grid_graph.observed_nodes],
                                         [node[1] for node in grid_graph.observed_nodes], s=15, c='mediumaquamarine')
        grid_points_obv.set_zorder(0)
        env.dyna_patch_list.append(grid_points_obv)

        # test: plot map
        # map = env.ax.scatter(grid_graph.occupancy_map[:, :, 0], grid_graph.occupancy_map[:, :, 1],
        #                      s=20, c=grid_graph.occupancy_map[:, :, 2], cmap='viridis')
        # map = env.ax.scatter(grid_graph.occupancy_map[:, :, 0], grid_graph.occupancy_map[:, :, 1],
        #                      s=20, c=[grid_graph.occupancy_map[:, :, 2] == 10])
        # map.set_zorder(10)
        # env.dyna_patch_list.append(map)

        fron = env.ax.scatter([f[0] for f in grid_graph.frontiers], [f[1] for f in grid_graph.frontiers], s=5, c='r')
        fron.set_zorder(20)
        env.dyna_patch_list.append(fron)

        # test obstacle observation
        # for obs in obs_observation:
        #     x, y = obs.exterior.xy
        #     obs_plot = env.ax.plot(x, y)
        #     env.dyna_line_list.append(obs_plot)

        env.render(show_traj=True)
        rrw.plot_clear()

        # update the figure without blocking the execution of the program
        env.fig.canvas.draw_idle()
        # exit with code 0 (successful exit) if key "esc" is pressed
        env.fig.canvas.mpl_connect('key_release_event',
                                   lambda event: [exit(0) if event.key == 'escape' else None])

    env.end(ani_name='dense_map_rrw', ani_kwargs={'subrectangles': True})