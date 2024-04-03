import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class MLPPPONets(nn.Module):
    """
    MLP Actor-critic neural networks of PPO with continuous action space
    """
    def __init__(self, observation_dim, action_dim):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(observation_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(observation_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


class CNNPPONets(nn.Module):
    """
    Actor-critic networks of PPO with CNN for processing depth image
    """
    def __init__(self, img_obs_shape, robot_obs_dim, action_dim):
        super().__init__()
        self.cnn = nn.Sequential(
            layer_init(nn.Conv2d(in_channels=img_obs_shape[0], out_channels=16, kernel_size=(6, 8), stride=8)),
            nn.ReLU(),
            layer_init(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 4), stride=3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # layer_init(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2)),
            nn.Flatten(),
            layer_init(nn.Linear(32 * 9 * 12, 512)),
            nn.ReLU(),
        )

        # X = torch.randn(1, 10, 480, 640)
        # for layer in self.cnn:
        #     X = layer(X)
        #     print(layer.__class__.__name__, 'output shape:\t', X.shape)
        # raise

        self.fcn = nn.Sequential(
            layer_init(nn.Linear(robot_obs_dim, 128)),
            nn.ReLU(),
        )
        self.critic = nn.Sequential(     # whether to share AC networks?
            layer_init(nn.Linear(512+128, 256)),
            # nn.Tanh(),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(512+128, 256)),
            # nn.Tanh(),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, action_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def get_value(self, img, obs_robot):
        img_embedding = self.cnn(img)
        robot_embedding = self.fcn(obs_robot)
        obs_merge = torch.cat((robot_embedding, img_embedding), dim=1)
        return self.critic(obs_merge)

    def get_action_and_value(self, img, obs_robot, action=None):
        img_embedding = self.cnn(img)
        robot_embedding = self.fcn(obs_robot)
        obs_merge = torch.cat((robot_embedding, img_embedding), dim=1)
        action_mean = self.actor_mean(obs_merge)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(obs_merge)