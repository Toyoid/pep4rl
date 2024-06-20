import os
import time
from datetime import datetime
import random
import numpy as np
from collections import deque
from copy import deepcopy
import rospy

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from config_sac import get_config
from envs.drm_navigation_env import DecisionRoadmapNavEnv
from algorithm.attention_networks import PolicyNet, SoftQNetwork


def get_time_str():
    date_time = datetime.fromtimestamp(time.time())
    formatted_date_time = date_time.strftime('%Y%m%d%H%M%S')
    return formatted_date_time


class RoadmapReplayBuffer:
    def __init__(self, buffer_size, device):
        self.buffer = deque(maxlen=buffer_size)
        self.device = device
        self.sampled_data = {
            "node_inputs": None, "edge_inputs": None, "current_index": None,
            "node_padding_mask": None, "edge_padding_mask": None, "edge_mask": None,
            "action": None, "reward": None, "done": None,
            "next_node_inputs": None, "next_edge_inputs": None, "next_current_index": None,
            "next_node_padding_mask": None, "next_edge_padding_mask": None, "next_edge_mask": None
        }

    def add(self, roadmap_state, action, reward, done, next_roadmap_state):
        node_inputs, edge_inputs, current_index, node_padding_mask, edge_padding_mask, edge_mask = roadmap_state
        nx_node_inputs, nx_edge_inputs, nx_current_index, nx_node_padding_mask, nx_edge_padding_mask, nx_edge_mask = next_roadmap_state
        self.buffer.append(
            (node_inputs, edge_inputs, current_index,
             node_padding_mask, edge_padding_mask, edge_mask,
             action, reward, done,
             nx_node_inputs, nx_edge_inputs, nx_current_index,
             nx_node_padding_mask, nx_edge_padding_mask, nx_edge_mask))

    def sample(self, batch_size):
        node_inputs_batch, edge_inputs_batch, current_index_batch, node_padding_mask_batch, edge_padding_mask_batch, edge_mask_batch, action_batch, reward_batch, done_batch, nx_node_inputs_batch, nx_edge_inputs_batch, nx_current_index_batch, nx_node_padding_mask_batch, nx_edge_padding_mask_batch, nx_edge_mask_batch = zip(
            *random.sample(self.buffer, batch_size))

        self.sampled_data["node_inputs"] = torch.stack(node_inputs_batch).squeeze(1).to(self.device)  # (n_batchs, node_padding_size, n_features)
        self.sampled_data["edge_inputs"] = torch.stack(edge_inputs_batch).squeeze(1).to(self.device)   # (n_batchs, 1, k_neighbor_size)
        self.sampled_data["current_index"] = torch.stack(current_index_batch).squeeze(1).to(self.device)  # (n_batchs, 1, 1)
        self.sampled_data["node_padding_mask"] = torch.stack(node_padding_mask_batch).squeeze(1).to(self.device)  # (n_batchs, 1, node_padding_size)
        self.sampled_data["edge_padding_mask"] = torch.stack(edge_padding_mask_batch).squeeze(1).to(self.device)  # (n_batchs, 1, k_neighbor_size)
        self.sampled_data["edge_mask"] = torch.stack(edge_mask_batch).squeeze(1).to(self.device)  # (n_batchs, k_neighbor_size, k_neighbor_size)
        self.sampled_data["action"] = torch.stack(action_batch).squeeze(1).to(self.device)  # (n_batchs, 1, 1)
        self.sampled_data["reward"] = torch.stack(reward_batch).squeeze(1).to(self.device)  # (n_batchs, 1, 1)
        self.sampled_data["done"] = torch.stack(done_batch).squeeze(1).to(self.device)  # (n_batchs, 1, 1)
        self.sampled_data["next_node_inputs"] = torch.stack(nx_node_inputs_batch).squeeze(1).to(self.device)
        self.sampled_data["next_edge_inputs"] = torch.stack(nx_edge_inputs_batch).squeeze(1).to(self.device)
        self.sampled_data["next_current_index"] = torch.stack(nx_current_index_batch).squeeze(1).to(self.device)
        self.sampled_data["next_node_padding_mask"] = torch.stack(nx_node_padding_mask_batch).squeeze(1).to(self.device)
        self.sampled_data["next_edge_padding_mask"] = torch.stack(nx_edge_padding_mask_batch).squeeze(1).to(self.device)
        self.sampled_data["next_edge_mask"] = torch.stack(nx_edge_mask_batch).squeeze(1).to(self.device)

        return self.sampled_data


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
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

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
    envs = DecisionRoadmapNavEnv(args=args, device=device, is_train=True)

    # define actor-critic networks
    actor = PolicyNet(args.input_dim, args.embedding_dim).to(device)
    qf1 = SoftQNetwork(args.input_dim, args.embedding_dim).to(device)
    qf2 = SoftQNetwork(args.input_dim, args.embedding_dim).to(device)
    qf1_target = SoftQNetwork(args.input_dim, args.embedding_dim).to(device)
    qf2_target = SoftQNetwork(args.input_dim, args.embedding_dim).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # automatic entropy tuning
    if args.autotune:
        target_entropy = 0.01 * (-np.log(1 / args.k_neighbor_size))  # TBD after reading SAC
        # log_alpha = torch.zeros(1, requires_grad=True, device=device)  # TBD after reading SAC
        log_alpha = torch.tensor([-2.], requires_grad=True, device="cuda:0")  # TBD after reading SAC
        alpha = log_alpha.exp().item()
        alpha_optimizer = optim.Adam([log_alpha], lr=1e-4)  # TBD after reading SAC: 1e-4 (context) q_lr in cleanrl
    else:
        alpha = args.alpha

    # set learning rate decay step | not use in context?
    actor_lr_decay = optim.lr_scheduler.StepLR(actor_optimizer, step_size=256, gamma=0.96)
    q_net_lr_decay = optim.lr_scheduler.StepLR(q_optimizer, step_size=256, gamma=0.96)

    # replay buffer
    replay_buffer = RoadmapReplayBuffer(args.buffer_size, device)

    np.set_printoptions(precision=3)  # print option for debug

    # run the experiment
    global_step = 0
    episode_ita = 450
    start_time = time.time()
    ''' record the total training time '''
    while episode_ita < args.num_episodes:
        roadmap_state = envs.reset(episode_ita)
        while not rospy.is_shutdown():
            global_step += 1
            # interact with the environment and collect data
            node_inputs, edge_inputs, current_index, node_padding_mask, edge_padding_mask, edge_mask = roadmap_state
            if global_step <= args.learning_starts:
                # randomly sample actions
                flat_mask = edge_padding_mask.view(-1)
                action_space = torch.nonzero(flat_mask != 1).squeeze()
                action_index = torch.randint(0, len(action_space), (1,)).to(device)
            else:
                # sample action using actor policy
                with torch.no_grad():
                    log_pi_list = actor(node_inputs, edge_inputs, current_index, node_padding_mask, edge_padding_mask, edge_mask)
                    action_index = torch.multinomial(log_pi_list.exp(), 1).long().squeeze(1)

            next_roadmap_state, reward, done, info = envs.step(action_index)

            replay_buffer.add(deepcopy(roadmap_state), action_index.view(1, 1, 1).clone(), reward.clone(), done.clone(),
                              deepcopy(next_roadmap_state))
            roadmap_state = next_roadmap_state

            # check the episode end points and log the relevant episodic return (not considering parallel envs)
            if info["episodic_outcome"] is not None:
                print(f"[Training Info]: episode={episode_ita + 1}, "
                      f"global_step={global_step}, outcome={info['episodic_outcome']}, "
                      f"episodic_return={info['episodic_return']}, episodic_length={info['episodic_length']}, "
                      f"success: {info['outcome_statistic']['success']}, "
                      f"collision: {info['outcome_statistic']['collision']}, "
                      f"timeout: {info['outcome_statistic']['timeout']}, "
                      f"success rate: {info['outcome_statistic']['success'] / (episode_ita - 450 + 1)}%\n")
                writer.add_scalar("charts/episodic_return", info["episodic_return"], global_step)
                writer.add_scalar("charts/episodic_length", info["episodic_length"], global_step)
                writer.add_scalar("charts/success_rate", info['outcome_statistic']['success'] / (episode_ita - 450 + 1), global_step)
                episode_ita += 1
                break

            # training
            # TBD after empirical tuning : training for n times each step like: for j in range(8)
            if global_step > args.learning_starts:
                data = replay_buffer.sample(args.batch_size)

                node_inputs_batch = data["node_inputs"]
                edge_inputs_batch = data["edge_inputs"]
                current_inputs_batch = data["current_index"]
                node_padding_mask_batch = data["node_padding_mask"]
                edge_padding_mask_batch = data["edge_padding_mask"]
                edge_mask_batch = data["edge_mask"]
                action_batch = data["action"]
                reward_batch = data["reward"]
                done_batch = data["done"]
                next_node_inputs_batch = data["next_node_inputs"]
                next_edge_inputs_batch = data["next_edge_inputs"]
                next_current_inputs_batch = data["next_current_index"]
                next_node_padding_mask_batch = data["next_node_padding_mask"]
                next_edge_padding_mask_batch = data["next_edge_padding_mask"]
                next_edge_mask_batch = data["next_edge_mask"]

                with torch.no_grad():
                    next_log_pi = actor(next_node_inputs_batch, next_edge_inputs_batch, next_current_inputs_batch,
                                        next_node_padding_mask_batch, next_edge_padding_mask_batch, next_edge_mask_batch)
                    next_q1_values, _ = qf1_target(next_node_inputs_batch, next_edge_inputs_batch,
                                                   next_current_inputs_batch, next_node_padding_mask_batch,
                                                   next_edge_padding_mask_batch, next_edge_mask_batch)
                    next_q2_values, _ = qf2_target(next_node_inputs_batch, next_edge_inputs_batch,
                                                   next_current_inputs_batch, next_node_padding_mask_batch,
                                                   next_edge_padding_mask_batch, next_edge_mask_batch)
                    next_q_values = torch.min(next_q1_values, next_q2_values)
                    value_prime_batch = torch.sum(
                        next_log_pi.unsqueeze(2).exp() * (next_q_values - log_alpha.exp() * next_log_pi.unsqueeze(2)),
                        dim=1).unsqueeze(1)
                    target_q_batch = reward_batch + args.gamma * (1 - done_batch) * value_prime_batch
                    # ? equation doesn't align
                    # min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                    # next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (
                    #     min_qf_next_target).view(-1)

                q1_values, _ = qf1(node_inputs_batch, edge_inputs_batch, current_inputs_batch,
                                   node_padding_mask_batch, edge_padding_mask_batch, edge_mask_batch)
                q2_values, _ = qf2(node_inputs_batch, edge_inputs_batch, current_inputs_batch,
                                   node_padding_mask_batch, edge_padding_mask_batch, edge_mask_batch)
                q1 = torch.gather(q1_values, 1, action_batch)
                q2 = torch.gather(q2_values, 1, action_batch)
                qf1_loss = F.mse_loss(q1, target_q_batch.detach())
                qf2_loss = F.mse_loss(q2, target_q_batch.detach())
                qf_loss = qf1_loss + qf2_loss

                # optimize the model
                q_optimizer.zero_grad()
                qf_loss.backward()
                # TBD:
                # q1_grad_norm = torch.nn.utils.clip_grad_norm_(qf1.parameters(), max_norm=20000, norm_type=2)
                # q2_grad_norm = torch.nn.utils.clip_grad_norm_(qf2.parameters(), max_norm=20000, norm_type=2)
                q_optimizer.step()

                if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                    for _ in range(
                            args.policy_frequency
                    ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                        with torch.no_grad():
                            # using torch.no_grad() because the Q-Net doesn't take actor output as input
                            q1_values, _ = qf1(node_inputs_batch, edge_inputs_batch, current_inputs_batch,
                                               node_padding_mask_batch, edge_padding_mask_batch, edge_mask_batch)
                            q2_values, _ = qf2(node_inputs_batch, edge_inputs_batch, current_inputs_batch,
                                               node_padding_mask_batch, edge_padding_mask_batch, edge_mask_batch)
                            q_values = torch.min(q1_values, q2_values)

                        log_pi = actor(node_inputs_batch, edge_inputs_batch, current_inputs_batch,
                                       node_padding_mask_batch, edge_padding_mask_batch, edge_mask_batch)
                        actor_loss = torch.sum((log_pi.exp().unsqueeze(2) * (
                                    log_alpha.exp().detach() * log_pi.unsqueeze(2) - q_values.detach())), dim=1).mean()
                        # ? equation doesn't align
                        # actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                        actor_optimizer.zero_grad()
                        actor_loss.backward()
                        # TBD: actor_grad_norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=100, norm_type=2)
                        actor_optimizer.step()

                        if args.autotune:
                            entropy = (log_pi * log_pi.exp()).sum(dim=-1)
                            alpha_loss = -(log_alpha * (entropy.detach() + target_entropy)).mean()
                            # ? equation doesn't align
                            # with torch.no_grad():
                            #     _, log_pi, _ = actor.get_action(data.observations)
                            # alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                            alpha_optimizer.zero_grad()
                            alpha_loss.backward()
                            alpha_optimizer.step()
                            alpha = log_alpha.exp().item()

                # context not use?
                # actor_lr_decay.step()
                # q_net_lr_decay.step()

                # update the target networks
                if global_step % args.target_network_frequency == 0:  # 64
                    for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)  # soft or hard???
                    for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

                # save net parameters
                if args.save_model:
                    if (global_step % args.model_save_frequency == 0) or (episode_ita == args.num_episodes - 1):
                        save_dir = f"{args.model_path}/drm_nav"
                        try:
                            os.mkdir(save_dir)
                            print("[Save Model]: Directory ", save_dir, " Created")
                        except FileExistsError:
                            print("[Save Model]: Directory", save_dir, " already exists")

                        checkpoint = {"actor_network": actor.state_dict(),
                                      "qf1_network": qf1.state_dict(),
                                      "qf2_network": qf2.state_dict(),
                                      "log_alpha": log_alpha,
                                      }
                        path_checkpoint = f"{save_dir}/checkpoint_{episode_ita}.pth"
                        torch.save(checkpoint, path_checkpoint)
                        print(f'[Save Model]: Model weights on episode {episode_ita} saved \n')

                # record training data
                if global_step % 100 == 0:
                    writer.add_scalar("losses/qf1_values", q1.mean().item(), global_step)
                    writer.add_scalar("losses/qf2_values", q2.mean().item(), global_step)
                    writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                    writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                    writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                    writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                    writer.add_scalar("losses/alpha", alpha, global_step)
                    # print("SPS:", int(global_step / (time.time() - start_time)))
                    # writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                    if args.autotune:
                        writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

    training_period = time.time() - start_time
    hours = int(training_period // 3600)
    minutes = int((training_period % 3600) // 60)
    seconds = int(training_period % 60)
    print(f"Total training time: {hours} h, {minutes} mins, {seconds} s")
    writer.close()


if __name__ == "__main__":
    main()
