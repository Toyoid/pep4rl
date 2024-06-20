import pickle
import numpy as np

def _get_init_robot_goal(rand_name):
    # Read Random Start Pose and Goal Position based on random name
    from os import path as os_path
    current_dir = os_path.dirname(os_path.abspath(__file__))
    f = open(current_dir + "/random_positions/" + rand_name + ".p", "rb")
    overall_list_xy = pickle.load(f)
    f.close()
    overall_robot_list_xy = overall_list_xy[0]
    overall_goal_list_xy = overall_list_xy[1]
    print(f"Use Random Start and Goal Positions [{rand_name}] ...")

    init_drone_height = 1.0
    is_train = False
    if is_train:
        num_episodes = 1000
    else:
        num_episodes = 200

    overall_init_z = np.array([init_drone_height] * num_episodes)

    if is_train:
        overall_robot_array = np.zeros((num_episodes, 4))
        overall_goal_array = np.zeros((num_episodes, 3))
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
    else:
        init_robot_xy = np.array(overall_robot_list_xy)
        init_goal_xy = np.array(overall_goal_list_xy)

        overall_robot_array = np.insert(init_robot_xy, 2, overall_init_z[:], axis=1)
        overall_goal_array = np.insert(init_goal_xy, 2, overall_init_z[:], axis=1)

    return overall_robot_array, overall_goal_array

init_robot_array, init_target_array = _get_init_robot_goal("eval_positions")


# from os import path as os_path
# current_dir = os_path.dirname(os_path.abspath(__file__))
# f = open(current_dir + "/random_positions/" + "eval_positions" + ".p", "rb")
# overall_list_xy = pickle.load(f)
# f.close()
# overall_robot_list_xy = overall_list_xy[0]  #36[8.2, 5.3, 2.05]  *82[6.800000000000001, -8.6, 4.4322441792358385, 2.942719855386857]    *151[6.800000000000001, -8.6, 4.4322441792358385, 2.942719855386857]
# overall_goal_list_xy = overall_list_xy[1]  #*36[-7.9, 1.8000000000000007, 6.06721949545603]  82[-5.6, -0.9]  151[-1.5, 7.2]
#
# print(overall_robot_list_xy[82])
# print(overall_robot_list_xy[151])
# print(overall_goal_list_xy[36])
#
# overall_robot_list_xy[82] = [6.800000000000001, -8.6, 2.942719855386857]
# overall_robot_list_xy[151] = [6.800000000000001, -8.6, 2.942719855386857]
# overall_goal_list_xy[36] = [-7.9, 1.8000000000000007]
#
# list = [overall_robot_list_xy, overall_goal_list_xy]
#
# f = open(current_dir + "/random_positions/" + "eval_positions" + ".p", "wb")
# pickle.dump(list, f)
# f.close()

print("finished")