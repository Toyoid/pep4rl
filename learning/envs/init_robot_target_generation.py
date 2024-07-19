import pickle
import random
import numpy as np
import rospy
from roadmap_service import RoadmapService
from pynput import keyboard
from matplotlib import pyplot as plt
from copy import deepcopy

terminate = False


def on_press(key):
    global terminate
    global load_roadmap

    if key == keyboard.Key.esc:
        terminate = True
        return False


def construct_roadmap():
    """
    Manually construct roadmap of a certain environment for robot-target pair generation
    :return: roadmap vertices
    """
    global terminate
    roadmap_srv = RoadmapService()
    # Set up key listener
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    print("Press 'ESC' key to terminate robot exploration when roadmap construction finished.")
    while not rospy.is_shutdown():
        vertex_pos_list, _, _ = roadmap_srv.update_roadmap_data()
        roadmap_srv.plot_roadmap()
        if terminate:
            print("ESC key pressed. Exiting...")
            break

    listener.join()  # Ensure listener thread has finished
    roadmap_srv.close()
    return vertex_pos_list


def sample_one_robot_target_pair(all_positions, robot_target_diff):
    """
    Randomly generate one pair of robot pose and target
    """
    target = random.choice(all_positions)
    pose = random.choice(all_positions)
    dist = np.hypot(target[0] - pose[0], target[1] - pose[1])
    while dist < robot_target_diff:
        pose = random.choice(all_positions)
        dist = np.hypot(target[0] - pose[0], target[1] - pose[1])
    pose.append(random.random() * 2 * np.pi)
    return pose, target


def gen_robot_target_pairs(candidate_vertex_list, num_pairs, robot_target_diff=9):
    """
    Generate initial robot-target pairs for RL training and testing
    :return: robot-target pairs
    """
    robot_pose_list = []
    target_pos_list = []
    for i in range(num_pairs):
        init_robot_pose, init_target_pos = sample_one_robot_target_pair(deepcopy(candidate_vertex_list), robot_target_diff)  # deepcopy is critical
        robot_pose_list.append(init_robot_pose)
        target_pos_list.append(init_target_pos)
    robot_target_pair_list = [robot_pose_list, target_pos_list]
    return robot_target_pair_list


def get_user_choice():
    while True:
        print("Do you want to load the pre-built roadmap? If not, you will need to start the robot system to build a global roadmap first.\n"
              "choice (y / n): ")
        choice = input().strip().lower()
        if choice in ['y', 'n']:
            return choice
        else:
            print("Invalid input. Please enter 'y' or 'n'.")


if __name__ == "__main__":
    from os import path as os_path

    world_name = "floorplan_world2"
    current_dir = os_path.dirname(os_path.abspath(__file__))
    file_path = current_dir + "/random_positions/" + world_name + "/roadmap_vertices" + ".p"

    choice = get_user_choice()
    if choice == 'y':
        print("Loading pre-built roadmap...")
        f = open(file_path, 'rb')
        roadmap_vertice_list = pickle.load(f)
        f.close()
        print("\n\nRoadmap vertices loaded.")
    else:
        print("Waiting for robot system to start. Please press 'Enter' after robot system get ready...")
        input()  # Enter pressed
        # construct roadmap and get vertices as robot-target pair candidates
        roadmap_vertice_list = construct_roadmap()
        # save candidate vertices
        f = open(file_path, 'wb')
        pickle.dump(roadmap_vertice_list, f)
        f.close()
        print("\n\nRoadmap vertices saved.")
    print(f"Number of Vertices: {len(roadmap_vertice_list)}\n"
          f"Vertice Dimension: {len(roadmap_vertice_list[0])}")


    # sample robot-target pairs
    robot_target_pairs = gen_robot_target_pairs(roadmap_vertice_list, num_pairs=1000)
    # save robot-target pairs
    file_path = current_dir + "/random_positions/" + world_name + "/robot_target_poses" + ".p"
    f = open(file_path, 'wb')
    pickle.dump(robot_target_pairs, f)
    f.close()
    print("Robot-target pairs saved.")

    # plot for test
    f = open(file_path, 'rb')
    read_robot_target_pairs = pickle.load(f)
    f.close()
    robot_poses, target_positions = read_robot_target_pairs

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')  # make width = height in each subplot
    ax.scatter([p[0] for p in robot_poses], [p[1] for p in robot_poses], c='darkcyan', s=15, alpha=0.5, zorder=2)
    ax.scatter([p[0] for p in target_positions], [p[1] for p in target_positions], c='r', alpha=0.5, zorder=1)
    for i in range(len(target_positions)):
        ax.plot([robot_poses[i][0], target_positions[i][0]], [robot_poses[i][1], target_positions[i][1]], linewidth=1.5,
                c='lightgrey', linestyle="--", zorder=0)
    plt.show()

