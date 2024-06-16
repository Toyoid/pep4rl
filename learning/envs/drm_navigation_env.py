import math
import pickle
from copy import deepcopy
import numpy as np
from scipy.spatial import KDTree
import torch
# import sys
# sys.path.append('/opt/ros/noetic/lib/python3/dist-packages')
import rospy

from geometry_msgs.msg import PointStamped, PoseStamped
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ModelState, ContactsState
from gazebo_msgs.srv import SetModelState
from std_srvs.srv import Empty
from global_planner.srv import GetRoadmap
from falco_planner.srv import SetRobotPose


def _t2n(x):
    return x.detach().cpu().numpy()


class DecisionRoadmapNavEnv:
    """
    RL Environment Class for Single-Robot Navigation based on Decision Roadmap
    """
    def __init__(self, args, device,
                 is_train=False,
                 goal_reward=20,
                 collision_reward=-20,
                 step_penalty_reward=-0.5,
                 goal_dis_amp=1/64.,
                 goal_near_th=0.4,
                 env_height_range=[0.2,2.5],
                 goal_dis_scale=1.0,
                 goal_dis_min_dis=0.3,
                 ):
        # # Observation space:
        # self.robot_obs_space = (5,)
        # # Action space:
        # self.action_space = (3,)

        # get initial robot and goal pose list in training_worlds.world
        self.init_robot_array, self.init_target_array = self._get_init_robot_goal("Rand_R1")

        # Robot messages
        self.odom = None
        self.robot_odom_init = False
        self.contact_sensor_init = False

        # Subscriber
        self.odom_sub = rospy.Subscriber("/CERLAB/quadcopter/odom", Odometry, self._odom_callback)
        self.contact_state_sub = rospy.Subscriber("/quadcopter/bumper_states", ContactsState, self._collision_callback)
        # Publisher
        # self.drone_pose_pub = rospy.Publisher("/CERLAB/quadcopter/setpoint_pose", PoseStamped, queue_size=10)  # subscribed by /gazebo
        self.current_goal_pub = rospy.Publisher("/agent/current_goal", PointStamped, queue_size=5)

        # Service
        self.pause_gazebo = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.unpause_gazebo = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.set_model_pose = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.get_roadmap = rospy.ServiceProxy('/dep/get_roadmap', GetRoadmap)
        self.reset_roadmap = rospy.ServiceProxy('/dep/reset_roadmap', SetRobotPose)
        self.set_robot_pose = rospy.ServiceProxy('/falco_planner/set_robot_pose', SetRobotPose)

        # Training environment setup
        self.step_time = args.step_time
        self.max_episode_steps = args.max_episode_steps
        self.target_position = None
        self.goal_reward = goal_reward
        self.collision_reward = collision_reward
        self.step_penalty = step_penalty_reward
        self.goal_near_th = goal_near_th
        self.goal_dis_amp = goal_dis_amp
        self.target_dis_dir_pre = None
        self.target_dis_dir_cur = None
        self.env_height_range = env_height_range
        # state scaling parameter for distance to goal
        self.goal_dis_scale = goal_dis_scale
        self.goal_dis_min_dis = goal_dis_min_dis

        # roadmap specific data
        self.edge_inputs = None
        self.node_coords = None
        self.route_node = []
        self.is_train = is_train
        self.device = device
        self.k_neighbor_size = args.k_neighbor_size
        self.coords_norm_coef_ = args.coords_norm_coef_
        self.utility_norm_coef_ = args.utility_norm_coef_
        self.node_padding_size = args.node_padding_size

        # information of interaction between agent and environment
        self.ita_in_episode = None
        self.robot_collision = False
        self.contact_info = None
        self.info = {
            "episodic_return": 0.0,
            "episodic_length": 0,
            "episodic_outcome": None,
            "outcome_statistic": {
                "success": 0,
                "collision": 0,
                "timeout": 0
            }
        }

        # Init Subscriber
        while not self.robot_odom_init:
            continue
        rospy.loginfo("Finish Odom Subscriber Init...")
        while not self.contact_sensor_init:
            continue
        rospy.loginfo("Finish Contact Sensor Subscriber Init...")
        rospy.loginfo("Subscriber Initialization Finished!")

    def reset(self, ita):
        """
        Reset funtion to start a new episode of the single robot navigation environment
        :param ita: iteration variable of ended episodes
        :return:
        next_img_obs: np-array (1, 10, 480, 640),
        next_robot_obs: np-array (1, 5)

        1. initialize target and robot position
        2. get the agent state: (relative target position, roadmap graph)
            roadmap graph: (potential to complete the task, utility for expanding the graph, maybe additionally whether the node has been accessed)
        """
        assert self.init_robot_array is not None
        assert self.init_target_array is not None
        assert ita < self.init_robot_array.shape[0]
        self.ita_in_episode = 0
        self.robot_collision = False  # initialize collision state at the start of an episode
        self.info["episodic_return"] = 0.0
        self.info["episodic_length"] = 0
        self.info["episodic_outcome"] = None
        '''
        unpause gazebo simulation and set robot initial pose
        '''
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause_gazebo()
        except rospy.ServiceException as e:
            print("Unpause Service Failed: %s" % e)
        # set goal position
        self.target_position = self.init_target_array[ita]
        target_msg = ModelState()
        target_msg.model_name = 'navigation_target'
        target_msg.pose.position.x = self.target_position[0]
        target_msg.pose.position.y = self.target_position[1]
        target_msg.pose.position.z = self.target_position[2]
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            resp = self.set_model_pose(target_msg)
        except rospy.ServiceException as e:
            print("Set Target Service Failed: %s" % e)
        # reset robot initial pose
        robot_init_pose = self.init_robot_array[ita]
        robot_init_quat = self._euler_2_quat(yaw=robot_init_pose[3])
        robot_msg = ModelState()
        robot_msg.model_name = 'quadcopter'
        robot_msg.pose.position.x = robot_init_pose[0]
        robot_msg.pose.position.y = robot_init_pose[1]
        robot_msg.pose.position.z = robot_init_pose[2]
        robot_msg.pose.orientation.x = robot_init_quat[1]
        robot_msg.pose.orientation.y = robot_init_quat[2]
        robot_msg.pose.orientation.z = robot_init_quat[3]
        robot_msg.pose.orientation.w = robot_init_quat[0]
        # reset robot pose
        rospy.wait_for_service('/falco_planner/set_robot_pose')
        try:
            resp = self.set_robot_pose(robot_msg)
        except rospy.ServiceException as e:
            print("Set Robot Pose Service Failed: %s" % e)
        print("[ROS Service Request]: set new quadcopter state...")
        # clear map and roadmap after reset
        rospy.wait_for_service('/dep/reset_roadmap')
        try:
            resp = self.reset_roadmap(robot_msg)
        except rospy.ServiceException as e:
            print("Reset Roadmap Service Failed: %s" % e)

        # IMPORTANT: sleep for enough time (1.0s) for the robot to scan and gain enough free range to build the roadmap
        rospy.sleep(1.2)
        robot_pose, roadmap_state = self._get_next_state()
        '''
        pause gazebo simulation and transform robot poses to robot observations
        '''
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause_gazebo()
        except rospy.ServiceException as e:
            print("Pause Service Failed: %s" % e)

        target_dis, target_dir = self._compute_dis_dir_2_goal(robot_pose)
        self.target_dis_dir_pre = [target_dis, target_dir]
        self.target_dis_dir_cur = [target_dis, target_dir]

        return roadmap_state

    def step(self, action_index):
        """
        Step funtion of single robot navigation environment
        :param action: tensor
        :return:
        next_img_obs: np-array (1, 10, 480, 640),
        next_robot_obs: np-array (1, 5),
        reward: list (1,),
        done: list (1,),
        info: dict
        """
        assert self.target_position is not None
        assert self.ita_in_episode is not None
        self.ita_in_episode += 1

        '''
        action decode
        '''
        selected_node_idx = self.edge_inputs[0, 0, action_index.item()]  # tensor(scalar)
        selected_node_pos = self.node_coords[selected_node_idx]  # np.array(2,)
        self.route_node.append(selected_node_pos)

        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause_gazebo()
        except rospy.ServiceException as e:
            print("Unpause Service Failed: %s" % e)
        '''
        Give action to robot and let robot execute, then get next observation
        '''
        goal = PointStamped()
        goal.header.frame_id = "map"
        goal.header.stamp = rospy.Time.now()
        goal.point.x = selected_node_pos[0]
        goal.point.y = selected_node_pos[1]
        goal.point.z = 1.0
        self.current_goal_pub.publish(goal)
        execute_start_time = rospy.get_time()
        goal_switch_flag = False
        while not (goal_switch_flag or rospy.is_shutdown()):
            dis = math.sqrt((self.odom.pose.pose.position.x - goal.point.x) ** 2 +
                            (self.odom.pose.pose.position.y - goal.point.y) ** 2 +
                            (self.odom.pose.pose.position.z - goal.point.z) ** 2)
            goal_switch_flag = (dis <= 0.9) or ((rospy.get_time() - execute_start_time) >= self.step_time)
                               # (self.odom.twist.twist.linear.x < 0.05 and self.odom.twist.twist.angular.z < 0.05)

        robot_pose, roadmap_state = self._get_next_state()
        '''
        Then pause the simulation
        1. Compute rewards of the actions
        2. Check if the episode is ended
        '''
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause_gazebo()
        except rospy.ServiceException as e:
            print("Pause Service Failed: %s" % e)

        goal_dis, goal_dir = self._compute_dis_dir_2_goal(robot_pose)
        self.target_dis_dir_cur = [goal_dis, goal_dir]
        reward, done = self._compute_reward(robot_pose)

        self.target_dis_dir_pre = [self.target_dis_dir_cur[0], self.target_dis_dir_cur[1]]

        self.info["episodic_return"] += reward
        self.info["episodic_length"] += 1

        reward = torch.tensor(reward).view(1, 1, 1).to(self.device)
        done = torch.tensor(done, dtype=torch.int32).view(1, 1, 1).to(self.device)

        return roadmap_state, reward, done, self.info

    def _get_next_state(self):
        # Get Collision State
        if self.contact_info:
            self.robot_collision = True

        # Get Robot Pose
        # compute yaw from quaternion
        quat = [self.odom.pose.pose.orientation.x,
                self.odom.pose.pose.orientation.y,
                self.odom.pose.pose.orientation.z,
                self.odom.pose.pose.orientation.w]
        siny_cosp = 2. * (quat[0] * quat[1] + quat[2] * quat[3])
        cosy_cosp = 1. - 2. * (quat[1] ** 2 + quat[2] ** 2)
        yaw = math.atan2(siny_cosp, cosy_cosp)  # range from -pi to pi
        robot_pose = np.array([self.odom.pose.pose.position.x,
                               self.odom.pose.pose.position.y,
                               self.odom.pose.pose.position.z, yaw])

        # Get Roadmap State
        rospy.wait_for_service('/dep/get_roadmap')
        try:
            roadmap_resp = self.get_roadmap()
        except rospy.ServiceException as e:
            print("Get Roadmap Service Failed: %s" % e)

        # 1. acquire node position, node utility, edges
        node_pos = []
        node_utility = []
        edges_adj_pos = []  # (num_nodes, adjacent_node_positions)
        edges_adj_idx = []  # (num_nodes, adjacent_node_indexes)
        for marker in roadmap_resp.roadmapMarkers.markers:
            if marker.ns == 'prm_point':
                # pos = [marker.pose.position.x, marker.pose.position.y, marker.pose.position.z]
                pos = [marker.pose.position.x, marker.pose.position.y]
                node_id = marker.id
                node_pos.append(pos)
            elif marker.ns == 'num_voxel_text':
                assert len(node_pos) > 0, "PRM node list is empty"
                utility_match = marker.id == node_id
                assert utility_match, "Utility does not match with PRM node"
                node_utility.append(float(marker.text))
            elif marker.ns == 'edge':
                edges_from_current_node = [marker.points[0]]
                for i, p in enumerate(marker.points):
                    if (i % 2) == 1:
                        edges_from_current_node.append(p)
                edges_adj_pos.append(edges_from_current_node)

        # 2. compute direction_vector
        self.node_coords = np.round(node_pos, 4)
        n_nodes = self.node_coords.shape[0]
        dis_vector = [self.target_position[0], self.target_position[1]] - self.node_coords
        dis = np.sqrt(np.sum(dis_vector**2, axis=1))
        # normalize direction vector
        non_zero_indices = dis != 0
        dis = np.expand_dims(dis, axis=1)
        dis_vector[non_zero_indices] = dis_vector[non_zero_indices] / dis[non_zero_indices]
        dis /= self.coords_norm_coef_
        direction_vector = np.concatenate((dis_vector, dis), axis=1)

        # 3. compute guidepost
        guidepost = np.zeros((n_nodes, 1))
        x = self.node_coords[:, 0] + self.node_coords[:, 1] * 1j
        for node in self.route_node:
            index = np.argwhere(x == node[0] + node[1] * 1j)
            guidepost[index] = 1

        # 4. formulate node_inputs tensor
        # normalize node observations
        node_coords_norm = self.node_coords / self.coords_norm_coef_
        node_utility_inputs = np.array(node_utility).reshape(n_nodes, 1)
        node_utility_inputs = node_utility_inputs / self.utility_norm_coef_
        # concatenate
        node_inputs = np.concatenate((node_coords_norm, node_utility_inputs, guidepost, direction_vector), axis=1)
        node_inputs = torch.FloatTensor(node_inputs).unsqueeze(0).to(self.device)

        # 5. calculate a mask for padded node
        if self.is_train:
            assert n_nodes < self.node_padding_size
            padding = torch.nn.ZeroPad2d((0, 0, 0, self.node_padding_size - n_nodes))
            node_inputs = padding(node_inputs)
            # calculate a mask to padded nodes
            node_padding_mask = torch.zeros((1, 1, n_nodes), dtype=torch.int64).to(self.device)
            node_padding = torch.ones((1, 1, self.node_padding_size - n_nodes), dtype=torch.int64).to(self.device)
            node_padding_mask = torch.cat((node_padding_mask, node_padding), dim=-1)
        else:
            node_padding_mask = None

        # 6. compute current node index
        current_node_idx = np.argmin(np.linalg.norm(self.node_coords - robot_pose[:2], axis=1))
        current_index = torch.tensor([current_node_idx]).unsqueeze(0).unsqueeze(0).to(self.device)

        # 7. compute edge_inputs
        for adj_nodes in edges_adj_pos:
            adj_node_idxs = []
            for adj_node_pos in adj_nodes:
                assert len(pos) == self.node_coords.shape[1], "Wrong dimension on node coordinates"
                # pos = [adj_node_pos.x, adj_node_pos.y, adj_node_pos.z]
                pos = np.round([adj_node_pos.x, adj_node_pos.y], 4)
                idx = np.argwhere((self.node_coords == pos).all(axis=1)).item()
                adj_node_idxs.append(idx)
            edges_adj_idx.append(adj_node_idxs)

        adjacent_matrix = self.calculate_edge_mask(edges_adj_idx)
        edge_mask = torch.from_numpy(adjacent_matrix).float().unsqueeze(0).to(self.device)
        # pad edge mask to node_padding_size while training
        if self.is_train:
            assert n_nodes < self.node_padding_size
            padding = torch.nn.ConstantPad2d(
                (0, self.node_padding_size - n_nodes, 0, self.node_padding_size - n_nodes), 1)
            edge_mask = padding(edge_mask)

        # pad edge_inputs with k-nearest neighbors
        current_node_pos = self.node_coords[current_node_idx]
        # build kd-tree to search k-nearest neighbors
        kdtree = KDTree(self.node_coords)
        k = min(self.k_neighbor_size, n_nodes)
        _, nearest_indices = kdtree.query(current_node_pos, k=k)

        # # padding option 1: pad edge_inputs to k_nearest_size with 0, this will filter node_0 as unconnected node
        # edge = np.pad(nearest_indices, (0, self.k_neighbor_size - k), mode='constant', constant_values=0)
        # self.edge_inputs = torch.tensor(edge).unsqueeze(0).unsqueeze(0).to(self.device)  # (1, 1, k_neighbor_size)
        #
        # edge_padding_mask = torch.zeros((1, 1, self.k_neighbor_size), dtype=torch.int64).to(self.device)
        # one = torch.ones_like(edge_padding_mask, dtype=torch.int64).to(self.device)
        # edge_padding_mask = torch.where(self.edge_inputs == 0, one, edge_padding_mask)

        # padding option 2: pad edge_inputs to k_nearest_size with -1 first, this keeps node_0 if it is connected
        edge = np.pad(nearest_indices, (0, self.k_neighbor_size - k), mode='constant', constant_values=-1)
        self.edge_inputs = torch.tensor(edge).unsqueeze(0).unsqueeze(0).to(self.device)  # (1, 1, k_neighbor_size)

        edge_padding_mask = torch.zeros((1, 1, self.k_neighbor_size), dtype=torch.int64).to(self.device)
        one = torch.ones_like(edge_padding_mask, dtype=torch.int64).to(self.device)
        edge_padding_mask = torch.where(self.edge_inputs == -1, one, edge_padding_mask)
        zero = torch.zeros_like(edge_padding_mask, dtype=torch.int64).to(self.device)
        self.edge_inputs = torch.where(self.edge_inputs == -1, zero, self.edge_inputs)  # change the unconnected node idxs from -1 back to 0 for attetion network

        roadmap_state = node_inputs, self.edge_inputs, current_index, node_padding_mask, edge_padding_mask, edge_mask
        '''
        Required Inputs:
        node_inputs, edge_inputs, current_index, node_padding_mask, edge_padding_mask, edge_mask
        
        Need to Provide:
            (x, y)
            utility
            guidepost
            (dx, dy, dis) -> node_inputs, node_padding_mask
            
            current_node_index (scalar: idx) -> current_index (tensor: (1,1,1))
            
            edges -> edges_inputs, edge_padding_mask, edge_mask
            
        NEXT:
        train SAC
        '''
        return robot_pose, roadmap_state

    def _compute_dis_dir_2_goal(self, robot_pose):
        """
        Compute relative distance and direction from robot pose to navigation goal
        """
        delta_x = self.target_position[0] - robot_pose[0]
        delta_y = self.target_position[1] - robot_pose[1]
        delta_z = self.target_position[2] - robot_pose[2]
        distance = math.sqrt(delta_x ** 2 + delta_y ** 2 + delta_z ** 2)

        ego_direction = math.atan2(delta_y, delta_x)
        robot_direction = robot_pose[3]
        # shift ego_direction and robot_direction to [0, 2*pi]
        while robot_direction < 0:
            robot_direction += 2 * math.pi
        while robot_direction > 2 * math.pi:
            robot_direction -= 2 * math.pi
        while ego_direction < 0:
            ego_direction += 2 * math.pi
        while ego_direction > 2 * math.pi:
            ego_direction -= 2 * math.pi
        '''
        Desired direction range is between -pi and pi, 
        [-pi, 0] for the goal on the right of the robot,
        [0, pi] for the goal on the left of the robot.
        Conditions are:
        if (ego_direction - robot_direction) is between 0 and pi, 
            then goal is on the left of the robot, direction = (ego_direction - robot_direction) ∈(0, pi)
        if (ego_direction - robot_direction) is between pi and 2*pi, 
            then goal is on the right of the robot, direction = -(2*pi - abs(ego_direction - robot_direction)) ∈(-pi, 0)
        if (ego_direction - robot_direction) is between -pi and 0, 
            then goal is on the right of the robot, direction = (ego_direction - robot_direction) ∈(-pi, 0)
        if (ego_direction - robot_direction) is between -2*pi and -pi, 
            then goal is on the left of the robot, direction = (2*pi - abs(ego_direction - robot_direction)) ∈(0, pi)
        '''
        pos_dir = abs(ego_direction - robot_direction)
        neg_dir = 2 * math.pi - abs(ego_direction - robot_direction)
        if pos_dir <= neg_dir:
            direction = math.copysign(pos_dir, ego_direction - robot_direction)
        else:
            direction = math.copysign(neg_dir, -(ego_direction - robot_direction))
        return distance, direction

    def _compute_reward(self, robot_pose):
        """
        Reward funtion:
        1. R_Arrive If Distance to Goal is smaller than D_goal
        2. R_Collision If Distance to Obstacle is smaller than D_obs
        3. a * (Last step distance to goal - current step distance to goal)
        """
        done = False
        if self.target_dis_dir_cur[0] < self.goal_near_th:
            reward = self.goal_reward
            done = True
            self.info["episodic_outcome"] = "success"
            self.info["outcome_statistic"]["success"] += 1
            print("[Episodic Outcome]: Goal achieved!")
        elif self.robot_collision:
            reward = self.collision_reward
            done = True
            self.info["episodic_outcome"] = "collision"
            self.info["outcome_statistic"]["collision"] += 1
            print("[Episodic Outcome]: Collides with obstacles!")
        elif (robot_pose[2] <= self.env_height_range[0]) or (robot_pose[2] >= self.env_height_range[1]):
            reward = self.collision_reward
            done = True
            self.info["episodic_outcome"] = "collision"
            self.info["outcome_statistic"]["collision"] += 1
            print("[Episodic Outcome]: Out of flying range!")
        elif self.ita_in_episode >= self.max_episode_steps:
            reward = self.goal_dis_amp * (self.target_dis_dir_pre[0] - self.target_dis_dir_cur[0])
            done = True
            self.info["episodic_outcome"] = "timeout"
            self.info["outcome_statistic"]["timeout"] += 1
            print("[Episodic Outcome]: Navigation timeout!")
        else:
            reward = self.goal_dis_amp * (self.target_dis_dir_pre[0] - self.target_dis_dir_cur[0])

        reward += self.step_penalty

        return reward, done

    @staticmethod
    def _get_init_robot_goal(rand_name):
        # Read Random Start Pose and Goal Position based on random name
        from os import path as os_path
        current_dir = os_path.dirname(os_path.abspath(__file__))
        overall_list_xy = pickle.load(open(current_dir + "/random_positions/" + rand_name + ".p", "rb"))
        overall_robot_list_xy = overall_list_xy[0]
        overall_goal_list_xy = overall_list_xy[1]
        print(f"Use Random Start and Goal Positions [{rand_name}] for training...")

        overall_init_z = np.array([1.0] * 1000)
        overall_robot_array = np.zeros((1000, 4))
        overall_goal_array = np.zeros((1000, 3))
        env_idx = 0
        for i in range(overall_goal_list_xy.__len__()):
            init_robot_xy = np.array(overall_robot_list_xy[i])
            init_goal_xy = np.array(overall_goal_list_xy[i])

            init_robot_pose = np.insert(init_robot_xy, 2, overall_init_z[env_idx: env_idx + init_goal_xy.shape[0]],
                                        axis=1)
            init_goal_pos = np.insert(init_goal_xy, 2, overall_init_z[env_idx: env_idx + init_goal_xy.shape[0]], axis=1)

            overall_robot_array[env_idx: env_idx + init_goal_xy.shape[0]] = init_robot_pose
            overall_goal_array[env_idx: env_idx + init_goal_xy.shape[0]] = init_goal_pos

            env_idx += init_goal_xy.shape[0]

        return overall_robot_array, overall_goal_array

    @staticmethod
    def calculate_edge_mask(edge_inputs):
        size = len(edge_inputs)
        bias_matrix = np.ones((size, size))
        for i in range(size):
            for j in range(size):
                if j in edge_inputs[i]:
                    bias_matrix[i][j] = 0
        return bias_matrix

    @staticmethod
    def _euler_2_quat(yaw=0, pitch=0, roll=0):
        """
        Transform euler angule to quaternion
        :param yaw: z
        :param pitch: y
        :param roll: x
        :return: quaternion
        """
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        w = cy * cp * cr + sy * sp * sr
        x = cy * cp * sr - sy * sp * cr
        y = sy * cp * sr + cy * sp * cr
        z = sy * cp * cr - cy * sp * sr
        return [w, x, y, z]

    def _odom_callback(self, odom):
        if self.robot_odom_init is False:
            self.robot_odom_init = True
        self.odom = odom

    def _collision_callback(self, contact_msg):
        if self.contact_sensor_init is False:
            self.contact_sensor_init = True
        if contact_msg.states:
            self.contact_info = contact_msg.states[-1].info
        else:
            self.contact_info = None




