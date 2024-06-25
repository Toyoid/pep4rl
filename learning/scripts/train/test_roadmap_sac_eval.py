import os
import time
from datetime import datetime
import random
import numpy as np
import rospy

import torch

from config_sac import get_config
from envs.drm_navigation_env import DecisionRoadmapNavEnv
from algorithm.attention_networks import PolicyNet


# def get_time_str():
#     date_time = datetime.fromtimestamp(time.time())
#     formatted_date_time = date_time.strftime('%Y%m%d%H%M%S')
#     return formatted_date_time


def main():
    args = get_config()

    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # cuda
    if args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.backends.cudnn.deterministic = args.cuda_deterministic
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
    torch.set_num_threads(args.n_training_threads)

    # env setup
    rospy.init_node("decision_roadmap_agent")
    envs = DecisionRoadmapNavEnv(args=args, device=device, is_train=False)

    # load attention policy network
    policy = PolicyNet(args.input_dim, args.embedding_dim).to(device)
    if device == 'cpu':
        checkpoint = torch.load(f'{args.model_path}/drm_nav/checkpoint_976.pth', map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(f'{args.model_path}/drm_nav/checkpoint_976.pth')
    policy.load_state_dict(checkpoint['actor_network'])

    np.set_printoptions(precision=3)

    # run the experiment
    global_step = 0
    episode_ita = 900
    start_time = time.time()
    while episode_ita < args.num_episodes:
        roadmap_state = envs.reset(episode_ita)
        while not rospy.is_shutdown():
            global_step += 1
            node_inputs, edge_inputs, current_index, node_padding_mask, edge_padding_mask, edge_mask = roadmap_state
            with torch.no_grad():
                log_pi_list = policy(node_inputs, edge_inputs, current_index, node_padding_mask, edge_padding_mask, edge_mask)

            if args.greedy:
                action_index = torch.argmax(log_pi_list, dim=1).long()
            else:
                action_index = torch.multinomial(log_pi_list.exp(), 1).long().squeeze(1)

            # execute the game and log data.
            next_roadmap_state, reward, next_done, info = envs.step(action_index)

            # check the episode end points and log the relevant episodic return (not considering parallel envs)
            if info["episodic_outcome"] is not None:
                print(f"[Training Info]: episode={episode_ita + 1}, "
                      f"global_step={global_step}, outcome={info['episodic_outcome']}, "
                      f"episodic_return={info['episodic_return']}, episodic_length={info['episodic_length']}, "
                      f"success: {info['outcome_statistic']['success']}, "
                      f"collision: {info['outcome_statistic']['collision']}, "
                      f"timeout: {info['outcome_statistic']['timeout']}, "
                      f"success rate: {info['outcome_statistic']['success'] / (episode_ita - 900 + 1)}%\n")
                episode_ita += 1
                break

    eval_period = time.time() - start_time
    hours = int(eval_period // 3600)
    minutes = int((eval_period % 3600) // 60)
    seconds = int(eval_period % 60)
    print(f"Total training time: {hours} h, {minutes} mins, {seconds} s")


if __name__ == "__main__":
    main()
