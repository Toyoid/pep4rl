import os
import sys
import copy
import numpy as np

from ir_sim.env import EnvBase
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../rrt_random_walk")
from rrt_random_walk import DynamicRRW


if __name__ == "__main__":
    env = EnvBase('dense_map.yaml', control_mode='auto', init_args={'no_axis': False}, collision_mode='react',
                  save_ani=False, full=False)
    p_goal_init = np.squeeze(env.robot.goal[:2])
    p_robot_init = np.squeeze(env.robot.init_state[:2])

    rrw = DynamicRRW(env=env, s_start=p_robot_init, s_goal=p_goal_init, step_len=0.7, iter_max=80)

    lidar_scan = env.robot.lidar.global_scan_matrix.T
    rrw.update_sensor_observe(lidar_scan)
    print("Searching initial path ... ")
    for i in range(10):
        print(f"Attempt {i + 1}: ")
        init_path = rrw.grow_rrt()
        if init_path is None:
            print("Failed")
        else:
            print("Success!\n")
            break

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