import pickle
import random
import numpy as np


def _get_init_pose_list(rand_name):
    # Read Random Start Pose and Goal Position based on random name
    overall_list_xy = pickle.load(open("random_positions/" + rand_name + ".p", "rb"))
    overall_robot_list_xy = overall_list_xy[0]
    overall_goal_list_xy = overall_list_xy[1]
    print(f"Use Random Start and Goal Positions [{rand_name}] for training...")

    overall_init_z = np.array([1.5] * 1000)
    overall_robot_list = []
    overall_goal_list = []
    env_idx = 0
    for i in range(overall_goal_list_xy.__len__()):
        init_robot_xy = np.array(overall_robot_list_xy[i])
        init_goal_xy = np.array(overall_goal_list_xy[i])

        init_robot_pose = np.insert(init_robot_xy, 2, overall_init_z[env_idx: env_idx + init_goal_xy.shape[0]], axis=1)
        init_goal_pos = np.insert(init_goal_xy, 2, overall_init_z[env_idx: env_idx + init_goal_xy.shape[0]], axis=1)
        env_idx += init_goal_xy.shape[0]

        overall_robot_list.append(init_robot_pose)
        overall_goal_list.append(init_goal_pos)

    return overall_robot_list, overall_goal_list


def _get_init_robot_goal(rand_name):
    # Read Random Start Pose and Goal Position based on random name
    overall_list_xy = pickle.load(open("random_positions/" + rand_name + ".p", "rb"))
    overall_robot_list_xy = overall_list_xy[0]
    overall_goal_list_xy = overall_list_xy[1]
    print(f"Use Random Start and Goal Positions [{rand_name}] for training...")

    overall_init_z = np.array([1.5] * 1000)
    overall_robot_array = np.zeros((1000, 4))
    overall_goal_array = np.zeros((1000, 3))
    env_idx = 0
    for i in range(overall_goal_list_xy.__len__()):
        init_robot_xy = np.array(overall_robot_list_xy[i])
        init_goal_xy = np.array(overall_goal_list_xy[i])

        init_robot_pose = np.insert(init_robot_xy, 2, overall_init_z[env_idx: env_idx + init_goal_xy.shape[0]], axis=1)
        init_goal_pos = np.insert(init_goal_xy, 2, overall_init_z[env_idx: env_idx + init_goal_xy.shape[0]], axis=1)

        overall_robot_array[env_idx: env_idx + init_goal_xy.shape[0]] = init_robot_pose
        overall_goal_array[env_idx: env_idx + init_goal_xy.shape[0]] = init_goal_pos

        env_idx += init_goal_xy.shape[0]

    return overall_robot_array, overall_goal_array


if __name__ == "__main__":
    # overall_robot_list, overall_goal_list = _get_init_pose_list("Rand_R1")
    overall_robot_array, overall_goal_array = _get_init_robot_goal("Rand_R1")
    print(overall_robot_array.shape[0])
    print(overall_robot_array[0])

    print("Hello")
