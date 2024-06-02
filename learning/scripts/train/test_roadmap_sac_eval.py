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


def get_time_str():
    date_time = datetime.fromtimestamp(time.time())
    formatted_date_time = date_time.strftime('%Y%m%d%H%M%S')
    return formatted_date_time


def main():
    args = get_config()
    args.exp_name = os.path.basename(__file__)[: -len(".py")]
    run_name = f"{args.env_name}__{args.exp_name}__seed-{args.seed}__{get_time_str()}"
    # init logger
    if args.use_wandb:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=False,
            save_code=True,
        )

    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # cuda
    if args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(args.n_training_threads)
        if args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(args.n_training_threads)

    # env setup
    rospy.init_node("decision_roadmap_agent")
    envs = DecisionRoadmapNavEnv(args=args, device=device, is_train=False)

    # load attention policy network
    policy = PolicyNet(args.input_dim, args.embedding_dim).to(device)
    if device == 'cuda':
        checkpoint = torch.load(f'{args.model_path}/context_aware_nav/checkpoint.pth')
    else:
        checkpoint = torch.load(f'{args.model_path}/checkpoint.pth', map_location=torch.device('cpu'))
    policy.load_state_dict(checkpoint['policy_model'])

    np.set_printoptions(precision=3)

    # run the experiment
    global_step = 0
    episode_ita = 0
    start_time = time.time()
    ''' record the total training time '''
    next_roadmap_state = envs.reset(episode_ita)  # need seeding?
    next_done = torch.zeros(args.num_envs).to(device)  # (1,)

    while episode_ita < args.num_episodes:
        # interact with the environment and collect data
        # for step in range(0, args.num_steps):
        while not rospy.is_shutdown():
            global_step += args.num_envs

            node_inputs, edge_inputs, current_index, node_padding_mask, edge_padding_mask, edge_mask = next_roadmap_state
            with torch.no_grad():
                logp_list = policy(node_inputs, edge_inputs, current_index, node_padding_mask, edge_padding_mask, edge_mask)

            if args.greedy:
                action_index = torch.argmax(logp_list, dim=1).long()
            else:
                action_index = torch.multinomial(logp_list.exp(), 1).long().squeeze(1)

            # execute the game and log data.
            next_roadmap_state, reward, next_done, info = envs.step(action_index)

            # check the episode end points and log the relevant episodic return (not considering parallel envs)
            if info["episodic_outcome"] is not None:
                print(f"[Training Info]: episode={episode_ita+1}, "
                      f"global_step={global_step}, outcome={info['episodic_outcome']}, "
                      f"episodic_return={info['episodic_return']}, episodic_length={info['episodic_length']}, "
                      f"success: {info['outcome_statistic']['success']}, "
                      f"collision: {info['outcome_statistic']['collision']}, "
                      f"timeout: {info['outcome_statistic']['timeout']}\n")
                episode_ita += 1
                if episode_ita == args.num_episodes:
                    break
                next_roadmap_state = envs.reset(episode_ita)


if __name__ == "__main__":
    main()
