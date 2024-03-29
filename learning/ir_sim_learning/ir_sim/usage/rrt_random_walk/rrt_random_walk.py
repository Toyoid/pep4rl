"""
DYNAMIC 2D RRT for random walk in unknown environments
@author: Hanxiao Li
reference: https://github.com/zhm-real/PathPlanning
"""

import math
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import Point, Polygon, LineString, LinearRing

from ir_sim.env import EnvBase


class TreeNode:
    def __init__(self, n):
        self.x = n[0]
        self.y = n[1]
        self.parent = None
        self.flag = "VALID"


class TreeEdge:
    def __init__(self, n_p, n_c):
        self.parent = n_p
        self.child = n_c
        self.flag = "VALID"


class DynamicRRW:
    def __init__(self, env, s_start, s_goal, step_len, goal_sample_rate=0, waypoint_sample_rate=0, iter_max=100):
        self.s_start = TreeNode(s_start)
        # keep track of the robot position in the RRT
        # by dynamically specifying the child of s-robot and always keep s-robot as the root node
        self.s_robot = TreeNode(s_start)
        self.s_ahead_robot = TreeNode(s_start)
        self.s_behind_robot = TreeNode(s_start)

        self.s_goal = TreeNode(s_goal)
        self.step_len = step_len
        self.goal_sample_rate = goal_sample_rate
        self.waypoint_sample_rate = waypoint_sample_rate
        self.iter_max = iter_max

        self.vertex = [self.s_start]  # do not include s-robot
        # self.vertex = [self.s_robot]  # s-robot as dynamic root of the tree: but not working well with the trimming mechanism
        self.vertex_new = []
        self.vertex_old = []
        self.edges = []

        self.robot_radius = env.robot.radius
        self.lidar_radius = env.robot.lidar.range_max
        self.x_range = env.world.x_range
        self.y_range = env.world.y_range

        self.env_boundary = LinearRing([(self.x_range[0], self.y_range[0]),
                                        (self.x_range[0], self.y_range[1]),
                                        (self.x_range[1], self.y_range[1]),
                                        (self.x_range[1], self.y_range[0])])
        self.env_bound_inflated = self.env_boundary.buffer(self.robot_radius)
        self.sensor_bound = Polygon([])

        self.env_obstacles = [np.array(obs.vertex).T for obs in env.obstacle_list]
        self.obs_observation_inflated = Polygon([])

        # The node positions of resulting path
        self.path = []
        # The nodes of resulting path
        self.waypoints = []

        # threshold to determine if the robot has reached a waypoint and needs to switch to another one
        self.waypoint_thresh = self.robot_radius * 5  # note: must be larger than the env.robot.goal_threshold

        self.ax = env.ax
        self.plot_line_list = []
        self.plot_patch_list = []

    def grow_rrt(self):
        for i in range(self.iter_max):
            node_target = self.choose_target()
            node_near = self.nearest_neighbor(node_target)
            node_new = self.new_state(node_near, node_target)

            if node_new and not self.is_collision(node_near, node_new):
                self.vertex.append(node_new)
                self.edges.append(TreeEdge(node_near, node_new))

        farthest_node = self.vertex[int(np.argmax([math.hypot(nd.x - self.s_robot.x, nd.y - self.s_robot.y)
                                                   for nd in self.vertex]))]
        self.s_goal = farthest_node
        self.path = self.extract_path(farthest_node)
        self.waypoints = self.extract_waypoint(farthest_node)

        # plot
        self.plot_visited(self.ax, animation=True)
        self.plot_path(self.ax)

        return self.path

    def rrt_random_walk(self):
        # check if there are nodes becoming invalid
        self.invalidate_nodes()
        # trim the invalid subtree
        self.trim_rrt()
        if not self.vertex:  # if the tree becomes empty
            print("\n!!!!!!!!!!!!!!!!!\nEMPTY TREE\n!!!!!!!!!!!!!!!!!\n")
            self.vertex.append(copy.deepcopy(self.s_robot))

        # if the original path becomes invalid, regrow the tree to find a new one
        if self.is_path_invalid():
            self.waypoints, self.path = self.regrow_rrt()
            # after the new path is extracted,
            # the s-ahead-robot and the s-behind-robot will not change, so no update is needed

            self.vertex_new = []

        # when the goal state hasn't been achieved
        if self.waypoints.__len__() > 3:
            # check whether the last waypoint has been arrived
            dist, _ = self.get_distance_and_angle(self.s_robot, self.waypoints[-2])
            if dist < self.waypoint_thresh:
                # update the root of RRT as s-robot, note that update order is significant
                self.s_behind_robot = self.s_ahead_robot
                self.s_behind_robot.flag = "INVALID"
                # pop out old s-ahead-robot
                self.waypoints.pop(-2)
                self.path.pop(-2)
                # update s-ahead-robot and set its parent as s-robot
                self.s_ahead_robot = self.waypoints[-2]
                self.s_ahead_robot.parent = self.s_robot

            self.plot_visited(self.ax, animation=False)
            self.plot_path(self.ax)

        # self.waypoint = [self.s_robot], one final step left to achieve goal (waypoint threshold is larger than goal arrive threshold)
        else:
            self.waypoints, self.path = self.regrow_rrt()
            # after the new path is extracted,
            # the s-ahead-robot and the s-behind-robot will not change, so no update is needed

            self.vertex_new = []

            self.plot_visited(self.ax, animation=False)
            self.plot_path(self.ax)

    def regrow_rrt(self):
        for i in range(self.iter_max):
            # sample random target node biased with goal and waypoint cache
            node_target = self.choose_target()
            node_near = self.nearest_neighbor(node_target)
            node_new = self.new_state(node_near, node_target)

            if node_new and not self.is_collision(node_near, node_new):
                self.vertex.append(node_new)
                self.vertex_new.append(node_new)
                self.edges.append(TreeEdge(node_near, node_new))

        farthest_node = self.vertex[int(np.argmax([math.hypot(nd.x - self.s_robot.x, nd.y - self.s_robot.y)
                                        for nd in self.vertex]))]
        self.s_goal = farthest_node
        path = self.extract_path(farthest_node)
        waypoint = self.extract_waypoint(farthest_node)

        return waypoint, path

    def goal_bias_sampling(self):
        if np.random.random() < self.goal_sample_rate:
            return self.s_goal
        else:
            return TreeNode((np.random.uniform(self.x_range[0] + self.robot_radius, self.x_range[1] - self.robot_radius),
                             np.random.uniform(self.y_range[0] + self.robot_radius, self.y_range[1] - self.robot_radius)))

    def choose_target(self):
        p = np.random.random()

        if p < self.goal_sample_rate:
            return self.s_goal
        elif self.goal_sample_rate < p < self.goal_sample_rate + self.waypoint_sample_rate:
            return self.waypoints[np.random.randint(0, len(self.waypoints) - 1)]
        else:
            # width = self.x_range[1] - self.x_range[0]
            # height = self.y_range[1] - self.y_range[0]
            # return Node((np.random.uniform(self.x_range[0] - width, self.x_range[1] + width),
            #              np.random.uniform(self.y_range[0] - height, self.y_range[1] + height)))
            return TreeNode((np.random.uniform(self.x_range[0], self.x_range[1]),
                             np.random.uniform(self.y_range[0], self.y_range[1])))

    def nearest_neighbor(self, n):
        # kd-tree for speeding up the lookup?
        return self.vertex[int(np.argmin([math.hypot(nd.x - n.x, nd.y - n.y)
                                          for nd in self.vertex]))]

    def new_state(self, node_start, node_end):
        dist, theta = self.get_distance_and_angle(node_start, node_end)

        dist = min(self.step_len, dist)
        node_new = TreeNode((node_start.x + dist * math.cos(theta),
                             node_start.y + dist * math.sin(theta)))
        node_new.parent = node_start

        return node_new

    @staticmethod
    def get_distance_and_angle(node_start, node_end):
        dx = node_end.x - node_start.x
        dy = node_end.y - node_start.y
        return math.hypot(dx, dy), math.atan2(dy, dx)

    @staticmethod
    def extract_path(node_end):
        path = []
        node_now = node_end

        while node_now is not None:
            path.append((node_now.x, node_now.y))
            node_now = node_now.parent

        return path

    @staticmethod
    def extract_waypoint(node_end):
        waypoint = []
        node_now = node_end

        while node_now is not None:
            waypoint.append(node_now)
            node_now = node_now.parent

        return waypoint

    def invalidate_nodes(self):
        for edge in self.edges:
            if self.is_collision(edge.parent, edge.child):
                edge.child.flag = "INVALID"

    def trim_rrt(self):
        for i in range(1, len(self.vertex)):
            node = self.vertex[i]
            node_p = node.parent
            if node_p.flag == "INVALID":
                node.flag = "INVALID"
                # self.vertex_global.append(node)

        self.vertex = [node for node in self.vertex if node.flag == "VALID"]
        self.vertex_old = copy.deepcopy(self.vertex)
        self.edges = [TreeEdge(node.parent, node) for node in self.vertex[1:len(self.vertex)]]

    def is_path_invalid(self):
        for node in self.waypoints:
            if node.flag == "INVALID":
                return True

    def is_collision(self, start_node, end_node):
        # check if end_node is in the env boundary
        end_node_point = Point(end_node.x, end_node.y)
        if not self.sensor_bound.contains(end_node_point):
            return True

        line = LineString([(start_node.x, start_node.y), (end_node.x, end_node.y)])
        for obs_poly in self.obs_observation_inflated:
            intersection = line.intersection(obs_poly)
            if not intersection.is_empty:
                return True

        intersection = line.intersection(self.env_bound_inflated)
        if not intersection.is_empty:
            return True

        return False

    def update_sensor_observe(self, sensor_scan):
        sensor_poly = Polygon(sensor_scan)
        # update obstacle observations
        self.obs_observation_inflated = []
        for obs in self.env_obstacles:
            poly = Polygon(obs)
            intersection = sensor_poly.intersection(poly)
            intersection_inflate = intersection.buffer(self.robot_radius)
            self.obs_observation_inflated.append(intersection_inflate)

        # update sensed area
        self.sensor_bound = Polygon(sensor_scan)
        # update sample range for new nodes
        self.x_range[0], self.y_range[0], self.x_range[1], self.y_range[1] = self.sensor_bound.bounds

        # plot sensor ring
        x, y = self.sensor_bound.exterior.xy
        self.plot_line_list.append(self.ax.plot(x, y, color='yellow', zorder=0))

    def plot_visited(self, ax, animation=True):
        if animation:
            # plot local tree
            count = 0
            for node in self.vertex:
                count += 1
                if node.parent:
                    self.plot_line_list.append(
                        ax.plot([node.parent.x, node.x], [node.parent.y, node.y], "-y"))

                    if count % 10 == 0:
                        plt.pause(0.0005)
        else:
            # plot local vertex
            for node in self.vertex:
                if node.parent:
                    self.plot_line_list.append(
                        ax.plot([node.parent.x, node.x], [node.parent.y, node.y], "-c"))
            # plot global vertex
            # for node in self.vertex_global:
            #     if node.parent:
            #         self.plot_line_list.append(
            #             ax.plot([node.parent.x, node.x], [node.parent.y, node.y], "-y"))

    def plot_vertex_new(self, ax):
        count = 0
        for node in self.vertex_new:
            count += 1
            if node.parent:
                self.plot_line_list.append(
                    ax.plot([node.parent.x, node.x], [node.parent.y, node.y], color='darkorange'))

                # if count % 10 == 0:
                #     plt.pause(0.05)

    def plot_vertex_old(self, ax):
        for node in self.vertex_old:
            if node.parent:
                self.plot_line_list.append(
                    ax.plot([node.parent.x, node.x], [node.parent.y, node.y], "-g"))

    def plot_path(self, ax, color='darkcyan'):
        self.plot_line_list.append(
            ax.plot([x[0] for x in self.path], [x[1] for x in self.path], linewidth=2, color=color, zorder=2))

    def plot_lidar(self, ax):
        lidar_circle = patches.Circle(xy=(self.s_robot.x, self.s_robot.y), radius=self.lidar_radius, color='r', alpha=0.1)
        lidar_circle.set_zorder(0)

        ax.add_patch(lidar_circle)
        self.plot_patch_list.append(lidar_circle)

    def plot_clear(self):
        [line.pop(0).remove() for line in self.plot_line_list]
        [patch.remove() for patch in self.plot_patch_list]

        self.plot_line_list = []
        self.plot_patch_list = []



if __name__ == "__main__":
    env = EnvBase('dense_map_rrw.yaml', control_mode='auto', init_args={'no_axis': False}, collision_mode='react',
                  save_ani=False, full=True)
    p_goal_init = np.squeeze(env.robot.goal[:2])
    p_robot_init = np.squeeze(env.robot.init_state[:2])

    rrw = DynamicRRW(env=env, s_start=p_robot_init, s_goal=p_goal_init, step_len=0.7, iter_max=60)

    lidar_scan = env.robot.lidar.global_scan_matrix.T
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

        print(rrw.vertex.__len__())

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

        # update the observed obstacle list in the rrt
        lidar_scan = env.robot.lidar.global_scan_matrix.T
        rrw.update_sensor_observe(lidar_scan)

        # adjust the rrt according to the updated observations
        rrw.plot_lidar(env.ax)
        rrw.rrt_random_walk()
        env.render(show_traj=True)
        rrw.plot_clear()


        # update the figure without blocking the execution of the program
        env.fig.canvas.draw_idle()
        # exit with code 0 (successful exit) if key "esc" is pressed
        env.fig.canvas.mpl_connect('key_release_event',
                                   lambda event: [exit(0) if event.key == 'escape' else None])

    env.end(ani_name='dense_map_rrw', ani_kwargs={'subrectangles': True})

    '''
    Iteratively Grow RRT in the Sensed Area and walk randomly on the dynamic RRT
    1. grow RRT in the sensed area, and get into the loop
    2. loop:
        2.1 select an edge node to explore
        2.2 extract a path from the edge node to the robot
        2.3 move the robot one step along the path
        2.4 update the robot position on the RRT
        2.5 update sensor observation
        2.6 regrow RRT in the new sensed area
    
    The key point is to dynamically changing the root of the tree
    - s-robot cannot be the dynamic root (initialize self.vertex = [self.s_robot] and see the experiments)
    - set s-ahead-robot as the dynamic root
    - at each waypoint switching iteration, because the waypoint[-3] will become the next root, it must be on the tree 
      instead of being the goal, otherwise the tree will become empty
    '''