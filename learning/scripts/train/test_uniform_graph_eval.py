import os
import time
from datetime import datetime
import random
import numpy as np
import rospy

import torch

from config_sac import get_config
from envs.uniform_graph_nav_env import UniformGraphNavEnv
from algorithm.attention_networks import PolicyNet


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
    envs = UniformGraphNavEnv(args=args, device=device, is_train=False)

    # load attention policy network
    policy = PolicyNet(args.input_dim, args.embedding_dim).to(device)
    if device == 'cpu':
        checkpoint = torch.load(f'{args.model_path}/context_aware_nav/checkpoint.pth', map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(f'{args.model_path}/context_aware_nav/checkpoint.pth', map_location=device)
    policy.load_state_dict(checkpoint['policy_model'])

    # run the experiment
    global_step = 0
    episode_ita = 0
    start_time = time.time()
    while episode_ita < args.eval_num_episodes:
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
            roadmap_state, reward, next_done, info = envs.step(action_index)

            # check the episode end points and log the relevant episodic return (not considering parallel envs)
            if info["episodic_outcome"] is not None:
                print(f"[Evaluation Info]: episode={episode_ita}, "
                      f"global_step={global_step}, outcome={info['episodic_outcome']}, "
                      f"episodic_return={info['episodic_return']:.2f}, \n"
                      f"episodic_length={info['episodic_length']}, "
                      f"success: {info['outcome_statistic']['success']}, "
                      f"collision: {info['outcome_statistic']['collision']}, "
                      f"timeout: {info['outcome_statistic']['timeout']}, "
                      f"success rate: {(100 * info['outcome_statistic']['success'] / (episode_ita + 1)):.1f}% \n")
                episode_ita += 1
                if episode_ita < args.eval_num_episodes:
                    print(
                        "*******************************************************************************************")
                    print(f"  Episode {episode_ita}: ")
                    print(
                        "*******************************************************************************************")
                break

    eval_period = time.time() - start_time
    hours = int(eval_period // 3600)
    minutes = int((eval_period % 3600) // 60)
    seconds = int(eval_period % 60)
    print(f"Total evaluation time: {hours} h, {minutes} mins, {seconds} s")
    envs.close()


if __name__ == "__main__":
    main()
