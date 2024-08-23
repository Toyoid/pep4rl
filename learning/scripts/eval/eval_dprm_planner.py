import os
import time
from datetime import datetime
import random
import numpy as np
import rospy
from envs.dprm_planner_wrapper import DPRMPlannerWrapper


def main():
    eval_num_episodes = 200
    episode_max_period = 250
    goal_near_th = 0.3

    # env setup
    rospy.init_node("dprm_planner_eval_env")
    envs = DPRMPlannerWrapper(episodic_max_period=episode_max_period, goal_near_th=goal_near_th)

    # run the experiment
    episode_ita = 0
    total_start_time = time.time()
    while episode_ita < eval_num_episodes:
        envs.reset(episode_ita)
        while not rospy.is_shutdown():
            info = envs.get_info()
            # check the episode end points and log the relevant episodic return (not considering parallel envs)
            if info["episodic_outcome"] is not None:
                print(f"[Evaluation Info]: episode={episode_ita}, "
                      f"outcome={info['episodic_outcome']}, "
                      f"episodic time cost={info['episodic_time_cost']}, "
                      f"success: {info['outcome_statistic']['success']}, "
                      f"collision: {info['outcome_statistic']['collision']}, "
                      f"timeout: {info['outcome_statistic']['timeout']}, "
                      f"success rate: {(100 * info['outcome_statistic']['success'] / (episode_ita + 1)):.1f}% \n")
                episode_ita += 1
                if episode_ita < eval_num_episodes:
                    print(
                        "*******************************************************************************************")
                    print(f"  Episode {episode_ita}: ")
                    print(
                        "*******************************************************************************************")
                break

    eval_period = time.time() - total_start_time
    hours = int(eval_period // 3600)
    minutes = int((eval_period % 3600) // 60)
    seconds = int(eval_period % 60)
    print(f"Total evaluation time: {hours} h, {minutes} mins, {seconds} s")
    envs.close()


if __name__ == "__main__":
    main()
