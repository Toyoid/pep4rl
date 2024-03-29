import numpy as np
import rospy
from std_srvs.srv import Empty


class PEEnv:
    """
    Class for Two-Robot Pursuit-Evasion Navigation in Gazebo Environment

    Main Function:
        1. Reset: Rest environment at the end of each episode
        and generate new goal position for next episode

        2. Step: Execute new action and return state
     """
    def __init__(self,

                 step_time=0.1):
        # Service
        self.pause_gazebo = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        self.unpause_gazebo = rospy.ServiceProxy('gazebo/unpause_physics', Empty)

    def reset(self, ita):
        assert self.env_range is not None
        assert self.robots_init_pose_list is not None
        assert ita < len(self.robots_init_pos_list)

        '''
        unpause gazebo simulation and set the initial poses of two robots
        '''
        # unpause gazebo simulation
        rospy.wait_for_service('gazebo/unpause_physics')
        try:
            self.unpause_gazebo()
        except rospy.ServiceException as e:
            print("Unpause Service Failed: %s" % e)
        # reset robot states
        '''code on setting init poses of the two robots'''

        '''
        pause gazebo simulation and transform robot poses to robot observations
        '''
        rospy.sleep(0.5)
        rospy.wait_for_service('gazebo/pause_physics')
        try:
            self.pause_gazebo()
        except rospy.ServiceException as e:
            print("Pause Service Failed: %s" % e)

        '''code on transforming robot poses to robot observations'''

        return robots_states


    def step(self, actions, ita_in_episode):
        rospy.wait_for_service('gazebo/unpause_physics')
        try:
            self.unpause_gazebo()
        except rospy.ServiceException as e:
            print("Unpause Service Failed: %s" % e)

        '''
        First give actions to robots and let robots execute, then get next observations
        '''

        rospy.wait_for_service('gazebo/pause_physics')
        try:
            self.pause_gazebo()
        except rospy.ServiceException as e:
            print("Pause Service Failed: %s" % e)
        '''
        Then pause the simulation
        1. Compute rewards of the actions
        2. Compute if the episode is ended
        '''

        return next_robots_states, rewards, done, info

    def _compute_reward(self):
        pass

    def _robots_states_cb(self, msg):
        pass

    def _robots_scans_cb(self, msg):
        pass



