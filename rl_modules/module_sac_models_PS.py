import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

"""
module network 2 for the SAC algorithm
task_hidden_dim=64
interface_dim=32
robot_hidden_dim=64
"""

LOG_SIG_MAX = 2
LOG_SIG_MIN = -9
epsilon = 1e-7  # 1e-6

class TaskModule(nn.Module):
    # The module for processing the task. Both Q and policy network can use this module.
    # interface_dim is the dimension of the interface between task and robot module. 16.
    # task input is 12+3 D for push and pick, 3D for reach
    # anchor tensor should be a 128*15 tensor, which is interface_dim*15 tensor
    def __init__(self, num_task_inputs, task_hidden_dim, interface_dim, anchor_tensor):
        super(TaskModule, self).__init__()

        self.norm_input = nn.BatchNorm1d(num_task_inputs)
        self.linear1 = nn.Linear(num_task_inputs, task_hidden_dim)
        self.linear2 = nn.Linear(task_hidden_dim, task_hidden_dim)
        self.linear3 = nn.Linear(task_hidden_dim, interface_dim)
        self.anchor = anchor_tensor
        self.interface_dim = interface_dim

    def forward(self, task_state):
        # task_state has the size 256(batch size)*15
        task_anchor_state = torch.cat((task_state, self.anchor), 0)
        x = self.norm_input(task_anchor_state)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        # Use normalize activation for the last layer of task module
        x = F.normalize(self.linear3(x), p=2.0, dim=1)
        # the x here is (256+128)*128 dim, because interface_dim=128
        # x_anchor is 256*128 dim, x_task_state is 128*128 dim
        x_task = x[0:-self.interface_dim, :]  # [0:-128, :]
        x_anchor = x[-self.interface_dim:, :]  # [-128:, :]
        # originally I need to do torch.nn.CosineSimilarity here
        # but notice that the x_anchor and x_task already have norm=1
        # I just need to do matrix multipilcation, and it will be equivalent to CosineSimilarity
        relative_interface = torch.mm(x_task, x_anchor.transpose(0, 1))
        return relative_interface


class RobotQModule(nn.Module):
    # The module for processing the robot info. Q network can use this module.
    # interface_dim is the dimension of the interface between task and robot module. 16.
    # num_action is 3 for reach and push, 4 for pick
    # num_robot_inputs is 7 for reach and push, 8 for pick
    def __init__(self, num_actions, num_robot_inputs, interface_dim, robot_hidden_dim, env_params):
        super(RobotQModule, self).__init__()
        self.max_action = env_params['action_max']  # 1.0

        self.norm_input = nn.BatchNorm1d(num_actions + num_robot_inputs)
        self.linearR = nn.Linear(num_actions + num_robot_inputs, robot_hidden_dim - interface_dim)
        self.linear1 = nn.Linear(robot_hidden_dim, robot_hidden_dim)
        self.linear2 = nn.Linear(robot_hidden_dim, robot_hidden_dim)
        self.linear3 = nn.Linear(robot_hidden_dim, 1)
        # self.dropout = nn.Dropout(0.25)

    def forward(self, action, robot_state, task_value):
        # use dropout to avoid the forming of receptors on different task
        # task_value = self.dropout(task_value)
        ro_input = torch.cat([action / self.max_action, robot_state], 1)
        x = self.norm_input(ro_input)
        action_emb = F.relu(self.linearR(x))
        xu = torch.cat([action_emb, task_value], 1)
        x = F.relu(self.linear1(xu))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class GausRobotPiModule(nn.Module):
    # output the mean and std for the gaussian distribution of the action, for SAC algorithm
    # The module for processing the robot info. Pi (policy) network can use this module.
    # interface_dim is the dimension of the interface between task and robot module. 16.
    # num_action is 3 for reach and push, 4 for pick
    # num_robot_inputs is 7 for reach and push, 8 for pick
    def __init__(self, num_actions, num_robot_inputs, interface_dim, robot_hidden_dim, env_params):
        super(GausRobotPiModule, self).__init__()

        # self.max_action = env_params['action_max']
        self.norm_input = nn.BatchNorm1d(num_robot_inputs)
        self.linearR = nn.Linear(num_robot_inputs, robot_hidden_dim - interface_dim)
        self.linear1 = nn.Linear(robot_hidden_dim, robot_hidden_dim)
        self.linear2 = nn.Linear(robot_hidden_dim, robot_hidden_dim)

        self.action_mean = nn.Linear(robot_hidden_dim, num_actions)
        self.action_log_std = nn.Linear(robot_hidden_dim, num_actions)
        # self.dropout = nn.Dropout(0.25)

    def forward(self, robot_state, task_value):
        x = self.norm_input(robot_state)
        action_emb = F.relu(self.linearR(x))
        x = torch.cat([action_emb, task_value], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        mean = self.action_mean(x)
        log_std = self.action_log_std(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)

        return mean, log_std


class QNetwork(nn.Module):
    # anchor tensor should be a 128*15 tensor
    def __init__(self, num_task_inputs, task_hidden_dim, interface_dim, num_actions, num_robot_inputs, robot_hidden_dim, env_params, anchor_tensor):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.Qtask = TaskModule(num_task_inputs, task_hidden_dim, interface_dim, anchor_tensor)
        self.Qrobot = RobotQModule(num_actions, num_robot_inputs, interface_dim, robot_hidden_dim, env_params)

    def forward(self, task_state, robot_state, action):

        task_value = self.Qtask(task_state)
        q_value = self.Qrobot(action, robot_state, task_value)

        return q_value


class GausPiNetwork(nn.Module):
    def __init__(self, num_task_inputs, task_hidden_dim, interface_dim, num_actions, num_robot_inputs, robot_hidden_dim, env_params, anchor_tensor):
        # anchor tensor should be a 128*15 tensor
        super(GausPiNetwork, self).__init__()

        self.Ptask = TaskModule(num_task_inputs, task_hidden_dim, interface_dim, anchor_tensor)
        self.Probot = GausRobotPiModule(num_actions, num_robot_inputs, interface_dim, robot_hidden_dim, env_params)

        self.action_scale = torch.tensor(env_params['action_max'])
        self.action_bias = torch.tensor(0.)

    def forward(self, task_state, robot_state):
        x = self.Ptask(task_state)
        mean, log_std = self.Probot(robot_state, x)
        return mean, log_std

    def sample(self, task_state, robot_state):
        mean, log_std = self.forward(task_state, robot_state)
        # mean = torch.nan_to_num(mean)
        # std = (log_std+epsilon).exp()
        # std = torch.nan_to_num(log_std.exp(), nan=torch.rand(1)[0])
        # std = (torch.nan_to_num(log_std)).exp()
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        dist_ent = normal.entropy().mean()
        return action, log_prob, dist_ent
