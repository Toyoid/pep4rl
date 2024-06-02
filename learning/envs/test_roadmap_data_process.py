# import sys
# sys.path.append('/opt/ros/noetic/lib/python3/dist-packages')

import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PointStamped
from global_planner.srv import GetRoadmap
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from scipy.spatial import KDTree
import torch
from copy import deepcopy
from os import path as os_path
from algorithm.attention_networks import PolicyNet


odom_ = None
node_coords_ = None
edge_inputs_ = None
route_node_ = []
target_position_ = [-7., -7.]
greedy_ = True
input_dim_ = 7
embedding_dim_ = 128
coords_norm_coef_ = 30.
utility_norm_coef_ = 4000.
use_node_padding_ = True
node_padding_size_ = 20
k_neighbor_size = 20
current_dir = os_path.dirname(os_path.abspath(__file__))
model_path_ = current_dir + "/../model_weights/context_aware_nav/"


def odom_callback(msg):
    global odom_
    odom_ = msg


def calculate_edge_mask(edge_inputs):
    size = len(edge_inputs)
    bias_matrix = np.ones((size, size))
    for i in range(size):
        for j in range(size):
            if j in edge_inputs[i]:
                bias_matrix[i][j] = 0
    return bias_matrix


if __name__ == "__main__":
    rospy.init_node('test_roadmap_service_node')
    odom_sub = rospy.Subscriber("/CERLAB/quadcopter/odom", Odometry, odom_callback)
    get_roadmap = rospy.ServiceProxy('/dep/get_roadmap', GetRoadmap)
    # current_goal_pub = rospy.Publisher("/falco_planner/way_point", PointStamped, queue_size=5)
    current_goal_pub = rospy.Publisher("/agent/current_goal", PointStamped, queue_size=5)

    # roadmap data processing specific variables
    if torch.cuda.is_available():
        device = torch.device("cuda:0")

    # load agent policy network
    agent_policy = PolicyNet(input_dim_, embedding_dim_).to(device)
    if device == 'cuda':
        checkpoint = torch.load(f'{model_path_}/checkpoint.pth')
    else:
        checkpoint = torch.load(f'{model_path_}/checkpoint.pth', map_location=torch.device('cpu'))
    agent_policy.load_state_dict(checkpoint['policy_model'])

    # plot
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))
    ax1.set_xlim(-15, 15)
    ax1.set_ylim(-15, 15)
    ax2.set_xlim(-15, 15)
    ax2.set_ylim(-15, 15)
    # make width = height in each subplot
    ax1.set_aspect('equal')
    ax2.set_aspect('equal')
    plot_line_list1 = []
    plot_patch_list1 = []
    plot_line_list2 = []
    plot_patch_list2 = []

    goal = PointStamped()
    goal.header.frame_id = "map"

    global_step = 0
    while not rospy.is_shutdown():
        global_step += 1

        robot_pose = np.array([odom_.pose.pose.position.x,
                               odom_.pose.pose.position.y,
                               odom_.pose.pose.position.z])
        '''
        Roadmap data processing
        '''
        # Get Roadmap State
        rospy.wait_for_service('/dep/get_roadmap')
        try:
            roadmap_resp = get_roadmap()
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
        node_coords_ = np.round(node_pos, 4)
        n_nodes = node_coords_.shape[0]
        dis_vector = [target_position_[0], target_position_[1]] - node_coords_
        dis = np.sqrt(np.sum(dis_vector ** 2, axis=1))
        # normalize direction vector
        non_zero_indices = dis != 0
        dis = np.expand_dims(dis, axis=1)
        dis_vector[non_zero_indices] = dis_vector[non_zero_indices] / dis[non_zero_indices]
        dis /= coords_norm_coef_
        direction_vector = np.concatenate((dis_vector, dis), axis=1)

        # 3. compute guidepost
        guidepost = np.zeros((n_nodes, 1))
        x = node_coords_[:, 0] + node_coords_[:, 1] * 1j
        for node in route_node_:
            index = np.argwhere(x == node[0] + node[1] * 1j)
            guidepost[index] = 1

        # 4. formulate node_inputs tensor
        # normalize node observations
        node_coords_norm = node_coords_ / coords_norm_coef_
        node_utility_inputs = np.array(node_utility).reshape(n_nodes, 1)
        node_utility_inputs = node_utility_inputs / utility_norm_coef_
        # concatenate
        node_inputs = np.concatenate((node_coords_norm, node_utility_inputs, guidepost, direction_vector), axis=1)
        node_inputs = torch.FloatTensor(node_inputs).unsqueeze(0).to(device)

        # 5. calculate a mask for padded node
        if use_node_padding_:
            assert n_nodes < node_padding_size_
            padding = torch.nn.ZeroPad2d((0, 0, 0, node_padding_size_ - n_nodes))
            node_inputs = padding(node_inputs)
            # calculate a mask to padded nodes
            node_padding_mask = torch.zeros((1, 1, n_nodes), dtype=torch.int64).to(device)
            node_padding = torch.ones((1, 1, node_padding_size_ - n_nodes), dtype=torch.int64).to(device)
            node_padding_mask = torch.cat((node_padding_mask, node_padding), dim=-1)
        else:
            node_padding_mask = None

        # 6. compute current node index
        current_node_idx = np.argmin(np.linalg.norm(node_coords_ - robot_pose[:2], axis=1))
        current_index = torch.tensor([current_node_idx]).unsqueeze(0).unsqueeze(0).to(device)

        # 7. compute edge_inputs
        for adj_nodes in edges_adj_pos:
            adj_node_idxs = []
            for adj_node_pos in adj_nodes:
                assert len(pos) == node_coords_.shape[1], "Wrong dimension on node coordinates"
                # pos = [adj_node_pos.x, adj_node_pos.y, adj_node_pos.z]
                pos = np.round([adj_node_pos.x, adj_node_pos.y], 4)
                idx = np.argwhere((node_coords_ == pos).all(axis=1)).item()
                adj_node_idxs.append(idx)
            edges_adj_idx.append(adj_node_idxs)

        adjacent_matrix = calculate_edge_mask(edges_adj_idx)
        edge_mask = torch.from_numpy(adjacent_matrix).float().unsqueeze(0).to(device)
        if use_node_padding_:
            assert n_nodes < node_padding_size_
            padding = torch.nn.ConstantPad2d(
                (0, node_padding_size_ - n_nodes, 0, node_padding_size_ - n_nodes), 1)
            edge_mask = padding(edge_mask)

        # # pad edge_inputs_ with 0
        # edge = deepcopy(edges_adj_idx[current_index])
        # while len(edge) < k_neighbor_size:
        #     edge.append(0)  # won't this be conflict with 'connection to node 0'? I think it will
        #
        # edge_inputs_ = torch.tensor(edge).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, k_neighbor_size)
        #
        # edge_padding_mask = torch.zeros((1, 1, k_neighbor_size), dtype=torch.int64).to(device)
        # one = torch.ones_like(edge_padding_mask, dtype=torch.int64).to(device)
        # edge_padding_mask = torch.where(edge_inputs_ == 0, one, edge_padding_mask)

        # pad edge_inputs_ with k-nearest neighbors
        current_node_pos = node_coords_[current_node_idx]
        # build kd-tree to search k-nearest neighbors
        kdtree = KDTree(node_coords_)
        k = min(k_neighbor_size, n_nodes)
        _, nearest_indices = kdtree.query(current_node_pos, k=k)

        # padding option 1: pad edge_inputs to k_nearest_size with 0, this will filter node_0 as unconnected node
        edge = np.pad(nearest_indices, (0, k_neighbor_size - k), mode='constant', constant_values=0)
        edge_inputs_ = torch.tensor(edge).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, k_neighbor_size)

        edge_padding_mask = torch.zeros((1, 1, k_neighbor_size), dtype=torch.int64).to(device)
        one = torch.ones_like(edge_padding_mask, dtype=torch.int64).to(device)
        edge_padding_mask = torch.where(edge_inputs_ == 0, one, edge_padding_mask)

        # padding option 2: pad edge_inputs to k_nearest_size with -1 first, this keeps node_0 if it is connected
        edge = np.pad(nearest_indices, (0, k_neighbor_size - k), mode='constant', constant_values=-1)
        edge_inputs_ = torch.tensor(edge).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, k_neighbor_size)

        edge_padding_mask = torch.zeros((1, 1, k_neighbor_size), dtype=torch.int64).to(device)
        one = torch.ones_like(edge_padding_mask, dtype=torch.int64).to(device)
        edge_padding_mask = torch.where(edge_inputs_ == -1, one, edge_padding_mask)
        zero = torch.zeros_like(edge_padding_mask, dtype=torch.int64).to(device)
        edge_inputs_ = torch.where(edge_inputs_ == -1, zero, edge_inputs_)

        roadmap_state = node_inputs, edge_inputs_, current_index, node_padding_mask, edge_padding_mask, edge_mask

        '''
        Node Selection and Publish
        '''
        # 1. randomly selection with mask (way one)
        # action_space = torch.ones_like(edge_inputs_, dtype=torch.float32).to(device)
        # action_space = action_space.masked_fill(edge_padding_mask == 1, -1e8)
        # logp_list = torch.log_softmax(action_space, dim=-1).squeeze(1)
        # print("----------------------------------------")
        # print(logp_list.exp())
        # print("----------------------------------------")
        # action_index = torch.multinomial(logp_list.exp(), 1).long().squeeze(1)

        # # 2. randomly selection with mask (way two)
        # flat_mask = edge_padding_mask.view(-1)
        # action_space = torch.nonzero(flat_mask != 1).squeeze()
        # action_index = torch.randint(0, len(action_space), (1,))

        # 3. select by attention network
        with torch.no_grad():
            logp_list = agent_policy(node_inputs, edge_inputs_, current_index, node_padding_mask, edge_padding_mask, edge_mask)

        if greedy_:
            action_index = torch.argmax(logp_list, dim=1).long()
        else:
            action_index = torch.multinomial(logp_list.exp(), 1).long().squeeze(1)

        selected_node_idx = edge_inputs_[0, 0, action_index.item()]  # tensor(scalar)

        print(f"Step: {global_step}")
        print(f"Current Node Position: {node_coords_[current_node_idx, 0], node_coords_[current_node_idx, 1]}")
        print(f"Selected Node Index: {selected_node_idx}")
        print(" ")

        selected_node_pos = node_coords_[selected_node_idx]  # np.array(2,)
        route_node_.append(selected_node_pos)

        goal.header.stamp = rospy.Time.now()
        goal.point.x = selected_node_pos[0]
        goal.point.y = selected_node_pos[1]
        goal.point.z = 1.0
        
        current_goal_pub.publish(goal)

        '''
        Plot Roadmap 
        '''
        # Figure 1: the global roadmap
        # edges
        for node_idx, adj_nodes_idx in enumerate(edges_adj_idx):
            for adj_node_idx in adj_nodes_idx:
                plot_line_list1.append(
                    ax1.plot([node_coords_[node_idx, 0], node_coords_[adj_node_idx, 0]],
                            [node_coords_[node_idx, 1], node_coords_[adj_node_idx, 1]], linewidth=1.5, color='darkcyan', zorder=0))
        # nodes
        nodes = ax1.scatter(node_coords_[:, 0], node_coords_[:, 1],
                             c=node_utility_inputs[:], cmap='viridis', alpha=1, zorder=1)
        plot_patch_list1.append(nodes)
        # guidepost
        post = ax1.scatter(node_coords_[:, 0], node_coords_[:, 1],
                             c=guidepost[:], cmap='Greys', s=30, alpha=0.25, zorder=2)
        plot_patch_list1.append(post)

        robot = patches.Circle(xy=(odom_.pose.pose.position.x, odom_.pose.pose.position.y), radius=0.2, color='r',
                               alpha=0.8)
        robot.set_zorder(3)
        ax1.add_patch(robot)
        plot_patch_list1.append(robot)

        # Figure 2: current node and edges
        current_node_edges = edges_adj_pos[current_node_idx]
        current_node_pos = current_node_edges[0]
        # edges
        for adj_node in current_node_edges:
            plot_line_list2.append(
                ax2.plot([current_node_pos.x, adj_node.x],
                         [current_node_pos.y, adj_node.y], linewidth=1.5, color='darkcyan', zorder=0))
        # nodes
        # nodes = ax2.scatter([n.x for n in current_node_edges], [n.y for n in current_node_edges],
        #                     c='orange', alpha=1, zorder=1)
        nodes = ax2.scatter(node_coords_[nearest_indices, 0], node_coords_[nearest_indices, 1],
                            c='orange', alpha=1, zorder=1)
        plot_patch_list2.append(nodes)
        # text
        # current_node_adjs = edges_adj_idx[current_node_idx]
        # for cnt, node_idx in enumerate(current_node_adjs):
        #     text = ax2.text(current_node_edges[cnt].x, current_node_edges[cnt].y, f'{node_idx}', fontsize=10, ha='right')
        #     plot_patch_list2.append(text)

        for node_idx in nearest_indices:
            text = ax2.text(node_coords_[node_idx, 0], node_coords_[node_idx, 1], f'{node_idx}', fontsize=10, ha='right')
            plot_patch_list2.append(text)
        # current node
        current = patches.Circle(xy=(current_node_pos.x, current_node_pos.y), radius=0.15, color='r', alpha=0.8)
        current.set_zorder(2)
        ax2.add_patch(current)
        plot_patch_list2.append(current)
        # selected node
        selection = patches.Circle(xy=(goal.point.x, goal.point.y), radius=0.15, color='green', alpha=0.8)
        selection.set_zorder(2)
        ax2.add_patch(selection)
        plot_patch_list2.append(selection)

        plt.pause(8)

        # clear figure
        [line.pop(0).remove() for line in plot_line_list1]
        [patch.remove() for patch in plot_patch_list1]
        [line.pop(0).remove() for line in plot_line_list2]
        [patch.remove() for patch in plot_patch_list2]
        plot_line_list1 = []
        plot_patch_list1 = []
        plot_line_list2 = []
        plot_patch_list2 = []
