import math
import pickle
import numpy as np
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


class PRMNode:
    def __init__(self, x=0, y=0, z=0, utility=0):
        self.x = x
        self.y = y
        self.z = z
        self.utility = utility


class PRMEdge:
    def __init__(self, p1, p2):
        # p1, p2: class p (p.x, p.y, p.z)
        self.point1 = p1
        self.point2 = p2


class DecisionRoadmapNavEnv:
    """
    RL Environment Class for Single-Robot Navigation based on Decision Roadmap
    """
    def __init__(self,
                 max_execute_time=10.0,
                 max_episode_steps=3,
                 goal_reward=30,
                 collision_reward=-20,
                 goal_dis_amp=1.5,
                 goal_near_th=0.4,
                 env_height_range=[0.2,2.5],
                 goal_dis_scale=1.0,
                 goal_dis_min_dis=0.3,
                 ):
        # Observation space:
        self.robot_obs_space = (5,)
        # Action space:
        self.action_space = (3,)

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
        self.max_execute_time = max_execute_time
        self.max_episode_steps = max_episode_steps
        self.target_position = None
        self.goal_reward = goal_reward
        self.collision_reward = collision_reward
        self.goal_near_th = goal_near_th
        self.goal_dis_amp = goal_dis_amp
        self.target_dis_dir_pre = None
        self.target_dis_dir_cur = None
        self.env_height_range = env_height_range
        # state scaling parameter for distance to goal
        self.goal_dis_scale = goal_dis_scale
        self.goal_dis_min_dis = goal_dis_min_dis

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

        rospy.sleep(2)
        robot_state, roadmap_state = self._get_next_state()
        '''
        pause gazebo simulation and transform robot poses to robot observations
        '''
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause_gazebo()
        except rospy.ServiceException as e:
            print("Pause Service Failed: %s" % e)

        target_dis, target_dir = self._compute_dis_dir_2_goal(robot_state[0])
        self.target_dis_dir_pre = [target_dis, target_dir]
        self.target_dis_dir_cur = [target_dis, target_dir]

        '''state pre-processing'''

        return roadmap_state

    def step(self, action):
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

        '''action decode'''
        # action = np.squeeze(_t2n(action))
        # action_clip = action.clip(self.action_range[0], self.action_range[1])

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
        goal.point.x = action.x
        goal.point.y = action.y
        goal.point.z = action.z
        self.current_goal_pub.publish(goal)
        execute_start_time = rospy.get_time()
        goal_switch_flag = False
        while not (goal_switch_flag or rospy.is_shutdown()):
            dis = math.sqrt((self.odom.pose.pose.position.x - action.x) ** 2 +
                            (self.odom.pose.pose.position.y - action.y) ** 2 +
                            (self.odom.pose.pose.position.z - action.z) ** 2)
            goal_switch_flag = (dis <= 0.9) or ((rospy.get_time() - execute_start_time) >= self.max_execute_time)
                               # (self.odom.twist.twist.linear.x < 0.05 and self.odom.twist.twist.angular.z < 0.05)

        robot_state, roadmap_state = self._get_next_state()
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

        goal_dis, goal_dir = self._compute_dis_dir_2_goal(robot_state[0])
        self.target_dis_dir_cur = [goal_dis, goal_dir]
        reward, done = self._compute_reward(robot_state[0])

        '''state pre-processing'''

        self.target_dis_dir_pre = [self.target_dis_dir_cur[0], self.target_dis_dir_cur[1]]
        '''need info that contains episodic information '''
        self.info["episodic_return"] += reward
        self.info["episodic_length"] += 1
        return roadmap_state, [reward], [done], self.info

    def _get_next_state(self):
        # get robot pose and speed
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
        robot_speed = np.array([self.odom.twist.twist.linear.x,
                                self.odom.twist.twist.linear.y,
                                # self.odom.twist.twist.linear.z,
                                self.odom.twist.twist.angular.z])
        robot_state = [robot_pose, robot_speed]

        # get roadmap state
        rospy.wait_for_service('/dep/get_roadmap')
        try:
            roadmap_resp = self.get_roadmap()
        except rospy.ServiceException as e:
            print("Get Roadmap Service Failed: %s" % e)

        # PRM data processing
        nodes = []
        edges = []
        for marker in roadmap_resp.roadmapMarkers.markers:
            if marker.ns == 'prm_point':
                node = PRMNode(marker.pose.position.x, marker.pose.position.y, marker.pose.position.z)
                nodes.append(node)
            elif marker.ns == 'num_voxel_text':
                assert len(nodes) > 0, "PRM node list is empty"
                utility_match = round(marker.pose.position.x, 4) == round(nodes[-1].x, 4) \
                                and round(marker.pose.position.y, 4) == round(nodes[-1].y, 4) \
                                and round(marker.pose.position.z - 0.1, 4) == round(nodes[-1].z, 4)
                assert utility_match, "Utility does not match with PRM node"
                nodes[-1].utility = int(marker.text)
            elif marker.ns == 'edge':
                p1 = marker.points[0]
                for i, p in enumerate(marker.points):
                    if (i % 2) == 1:
                        p2 = p
                        edge = PRMEdge(p1, p2)
                        edges.append(edge)

        prm = {'nodes': nodes, 'edges': edges}

        # get collision state
        if self.contact_info:
            self.robot_collision = True

        return robot_state, prm

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




