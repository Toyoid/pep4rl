import math
import pickle
import numpy as np
import rospy
import os

from geometry_msgs.msg import PointStamped, PoseStamped
from sensor_msgs.msg import Joy
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from std_srvs.srv import Empty
from std_msgs.msg import Float32
from falco_planner.srv import SetRobotPose


class NonLearningPlannerWrapper:
    """
    Wrapper for Navigation using Non-learning Global Planners
    """
    def __init__(self,
                 episodic_max_period=250,
                 goal_near_th=0.3,
                 env_height_range=[0.2,2.5]):
        #     # get initial robot and goal pose list in dense_worlds.world
        #     self.init_robot_array, self.init_target_array = self._get_init_robot_goal("eval_positions")
        self.init_robot_array, self.init_target_array = self._get_init_robot_goal("maze_medium")

        # Robot messages
        self.odom = None
        self.robot_odom_init = False
        self.time_duration = None
        self.time_duration_init = False
        self.run_time = None
        self.runtime_init = False
        self.explored_volume = None
        self.explored_volume_init = False
        self.traveling_distance = None
        self.traveling_dist_init = False

        # Subscriber
        # rospy.Subscriber("/CERLAB/quadcopter/odom", Odometry, self._odom_callback)
        rospy.Subscriber("/falco_planner/state_estimation", Odometry, self._odom_callback)
        rospy.Subscriber("/data_recording/time_duration", Float32, self._time_duration_callback)
        rospy.Subscriber("/data_recording/runtime", Float32, self._runtime_callback)
        rospy.Subscriber("/data_recording/explored_volume", Float32, self._explored_volume_callback)
        rospy.Subscriber("/data_recording/traveling_distance", Float32, self._traveling_dist_callback)
        # Publisher
        self.nav_target_pub = rospy.Publisher("env/nav_target", PointStamped, queue_size=5)
        # publish joy for activating falco planner
        self.pub_joy = rospy.Publisher('/falco_planner/joy', Joy, queue_size=5)

        # Service
        self.pause_gazebo = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.unpause_gazebo = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.set_model_pose = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.reset_roadmap = rospy.ServiceProxy('/dep/reset_roadmap', SetRobotPose)
        self.set_robot_pose = rospy.ServiceProxy('/falco_planner/set_robot_pose', SetRobotPose)
        self.init_rotate_scan = rospy.ServiceProxy('falco_planner/init_rotate_scan_service', Empty)

        self.target_position = None
        self.goal_near_th = goal_near_th
        self.episodic_max_period = episodic_max_period
        self.env_height_range = env_height_range
        self.episodic_start_time = None
        self.episodic_start_dist = None
        self.avg_runtime = 0
        self.runtime_count = 0
        self.info = {
            "episodic_time_cost": 0,
            "episodic_dist_cost": 0,
            "episodic_avg_runtime": 0,
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
        rospy.loginfo("Odom Subscriber Initialization Finished.")
        # rospy.loginfo("Subscriber Initialization Finished!")

    def reset(self, ita):
        assert self.init_robot_array is not None
        assert self.init_target_array is not None
        assert ita < self.init_robot_array.shape[0]
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
            print(f"[ROS Service Request]: Target position [{self.target_position[0]}, {self.target_position[1]}, {self.target_position[2]}]")
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
        # rotate the robot to get initial scan of the environment
        rospy.wait_for_service('/falco_planner/init_rotate_scan_service')
        try:
            resp = self.init_rotate_scan()
        except rospy.ServiceException as e:
            print("Initial Rotate Scan Service Failed: %s" % e)
        print("[ROS Service Request]: rotate to gain initial scan...")
        # IMPORTANT: sleep for enough time (>2.2s) for the robot to rotate, scan and gain enough free range to build the roadmap
        rospy.sleep(2.5)

        # publish joy for activating falco planner
        self._pub_falco_planner_joy()

        # Init Subscriber
        while not self.robot_odom_init:
            continue
        rospy.loginfo("Odom Subscriber Initialization Finished.")
        while not self.time_duration_init:
            continue
        rospy.loginfo("Time-Duration Subscriber Initialization Finished.")
        # while not self.runtime_init:
        #     continue
        # rospy.loginfo("Runtime Subscriber Initialization Finished.")
        while not self.explored_volume_init:
            continue
        rospy.loginfo("Explored-Volume Subscriber Initialization Finished.")
        while not self.traveling_dist_init:
            continue
        rospy.loginfo("Traveling-Distance Subscriber Initialization Finished.")
        rospy.loginfo("All Subscribers Initialization Finished.")

        # publish navigation target position to roadmap after resetting roadmap
        target = PointStamped()
        target.header.frame_id = "map"
        target.header.stamp = rospy.Time.now()
        target.point.x = self.target_position[0]
        target.point.y = self.target_position[1]
        target.point.z = self.target_position[2]
        self.nav_target_pub.publish(target)
        print("[Navigation]: Ultimate target position published.")

        self.episodic_start_time = self.time_duration
        self.episodic_start_dist = self.traveling_distance
        self.avg_runtime = 0
        self.runtime_count = 0

    def get_info(self):
        assert self.target_position is not None

        target_dis = math.sqrt((self.odom.pose.pose.position.x - self.target_position[0]) ** 2 +
                               (self.odom.pose.pose.position.y - self.target_position[1]) ** 2 +
                               (self.odom.pose.pose.position.z - self.target_position[2]) ** 2)
        target_reach_flag = (target_dis <= self.goal_near_th)
        out_of_range_flag = (self.odom.pose.pose.position.z <= self.env_height_range[0]) or \
                            (self.odom.pose.pose.position.z >= self.env_height_range[1])
        episodic_time_cost = self.time_duration - self.episodic_start_time
        episodic_dist_cost = self.traveling_distance - self.episodic_start_dist

        if target_reach_flag:
            self.info["episodic_time_cost"] = episodic_time_cost
            self.info["episodic_dist_cost"] = episodic_dist_cost
            self.info["episodic_avg_runtime"] = self.avg_runtime
            self.info["episodic_outcome"] = "success"
            self.info["outcome_statistic"]["success"] += 1
            print("[Episodic Outcome]: Goal achieved!")
        elif out_of_range_flag:
            self.info["episodic_time_cost"] = episodic_time_cost
            self.info["episodic_dist_cost"] = episodic_dist_cost
            self.info["episodic_avg_runtime"] = self.avg_runtime
            self.info["episodic_outcome"] = "collision"
            self.info["outcome_statistic"]["collision"] += 1
            print("[Episodic Outcome]: Out of flying range!")
        elif episodic_time_cost >= self.episodic_max_period:
            self.info["episodic_time_cost"] = episodic_time_cost
            self.info["episodic_dist_cost"] = episodic_dist_cost
            self.info["episodic_avg_runtime"] = self.avg_runtime
            self.info["episodic_outcome"] = "timeout"
            self.info["outcome_statistic"]["timeout"] += 1
            print("[Episodic Outcome]: Navigation timeout!")

        return self.info

    @staticmethod
    def close():
        print(" ")
        rospy.loginfo("Shutting down all nodes...")
        os.system("rosnode kill -a")

    @staticmethod
    def _get_init_robot_goal(map_name):
        # Read Random Start Pose and Goal Position based on random name
        from os import path as os_path
        current_dir = os_path.dirname(os_path.abspath(__file__))
        f = open(current_dir + "/random_positions/" + map_name + "/robot_target_poses" + ".p", "rb")
        overall_list = pickle.load(f)
        f.close()
        robot_poses_array = np.array(overall_list[0])
        target_pos_array = np.array(overall_list[1])

        print(f"Use Random Start and Goal Positions in [{map_name}] ...")

        return robot_poses_array, target_pos_array

    def _pub_falco_planner_joy(self):
        joy = Joy()
        joy.axes = [0., 0., -1.0, 0., 1.0, 1.0, 0., 0.]
        joy.buttons = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
        joy.header.stamp = rospy.Time.now()
        joy.header.frame_id = "waypoint_tool"
        self.pub_joy.publish(joy)

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

    def _time_duration_callback(self, msg):
        if self.time_duration_init is False:
            self.time_duration_init = True
        self.time_duration = msg.data

    def _runtime_callback(self, msg):
        if self.runtime_init is False:
            self.runtime_init = True
        self.run_time = msg.data
        self.runtime_count += 1
        self.avg_runtime += (1 / self.runtime_count) * (self.run_time - self.avg_runtime)

    def _explored_volume_callback(self, msg):
        if self.explored_volume_init is False:
            self.explored_volume_init = True
        self.explored_volume = msg.data

    def _traveling_dist_callback(self, msg):
        if self.traveling_dist_init is False:
            self.traveling_dist_init = True
        self.traveling_distance = msg.data
