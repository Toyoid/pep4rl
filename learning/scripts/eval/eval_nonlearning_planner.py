import time
import rospy
from envs.nonlearning_planner_wrapper import NonLearningPlannerWrapper


def main():
    eval_num_episodes = 3
    episode_max_period = 150
    goal_near_th = 0.3
    ros_node_name = "dprm_planner_eval_env"

    # env setup
    rospy.init_node(ros_node_name)
    envs = NonLearningPlannerWrapper(episodic_max_period=episode_max_period, goal_near_th=goal_near_th)

    # eval results
    time_cost_list = []
    dist_cost_list = []
    runtime_list = []

    # run the experiment
    episode_ita = 0
    total_start_time = time.time()
    while episode_ita < eval_num_episodes:
        envs.reset(episode_ita)
        while not rospy.is_shutdown():
            info = envs.get_info()
            # check the episode end points and log the relevant episodic return (not considering parallel envs)
            if info["episodic_outcome"] is not None:
                runtime_list.append(info['episodic_avg_runtime'])
                if info["episodic_outcome"] == "success":
                    time_cost_list.append(info['episodic_time_cost'])
                    dist_cost_list.append(info['episodic_dist_cost'])

                print(f"\n[Evaluation Info]: episode={episode_ita}, "
                      f"outcome={info['episodic_outcome']}, \n"
                      f"episodic time cost={info['episodic_time_cost']} s, \n"
                      f"episodic distance cost={info['episodic_dist_cost']} m, \n"
                      f"average algorithm runtime={info['episodic_avg_runtime']} s, \n"
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
    print(f"Average episodic time cost: {sum(time_cost_list) / len(time_cost_list)}")
    print(f"Average episodic traveling distance: {sum(dist_cost_list) / len(dist_cost_list)}")
    print(f"Average algorithm runtime: {sum(runtime_list) / len(runtime_list)}")
    envs.close()


if __name__ == "__main__":
    main()
