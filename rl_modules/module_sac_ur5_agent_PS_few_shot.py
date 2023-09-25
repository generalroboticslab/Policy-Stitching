import torch
import os
from datetime import datetime
import numpy as np
from mpi4py import MPI
from mpi_utils.mpi_utils import sync_networks, sync_grads
from rl_modules.module_replay_buffer import module_replay_buffer
from rl_modules.module_sac_models_PS import GausPiNetwork, QNetwork
from her_modules.her import her_sampler
import time
from robot_env.utilities import distance
import pybullet as p

"""
module sac with HER (MPI-version)
This version refers to the ASL code
"""


class module_sac_ur5_agent_PS_few_shot:
    def __init__(self, num_task_inputs, num_robot_inputs, args, env, env_params, ta_model_path, ro_model_path):
        print(num_task_inputs)
        self.args = args
        self.env = env
        self.env_params = env_params
        self.device = torch.device(args.device)
        # create the dict for store the model and data
        if MPI.COMM_WORLD.Get_rank() == 0:
            if not os.path.exists(self.args.save_dir2):
                os.mkdir(self.args.save_dir2)
            if not os.path.exists(self.args.data_dir2):
                os.mkdir(self.args.data_dir2)
            # path to save the model and data
            self.model_path = os.path.join(self.args.save_dir2, self.args.env_name)
            self.data_path = os.path.join(self.args.data_dir2, self.args.env_name)
            if not os.path.exists(self.model_path):
                os.mkdir(self.model_path)
            if not os.path.exists(self.data_path):
                os.mkdir(self.data_path)
        # create the network
        print(args.task_hidden_dim)
        print(args.interface_dim)
        print(args.robot_hidden_dim)
        print(self.args.save_model)
        print(self.args.save_data)
        data1 = torch.load(ta_model_path, map_location=torch.device(self.args.device))  # data1 for task module
        data2 = torch.load(ro_model_path, map_location=torch.device(self.args.device))  # data2 for robot module
        self.data1 = data1
        self.data2 = data2
        anchor_numpy = np.load('saved_anchors/Ur5Pos3Push1_Push4_task_state_k_means_128closest_mixed_anchors.npy')
        # be aware of dtype here, might be wrong
        anchor_tensor = torch.Tensor(anchor_numpy).to(self.device)

        self.actor_network = GausPiNetwork(num_task_inputs, args.task_hidden_dim, args.interface_dim, self.env.action_shape, num_robot_inputs, args.robot_hidden_dim, env_params, anchor_tensor)
        self.critic_network = QNetwork(num_task_inputs, args.task_hidden_dim, args.interface_dim, self.env.action_shape, num_robot_inputs, args.robot_hidden_dim, env_params, anchor_tensor)
        print("actor network parameters number:")
        actor_total_params = sum(p.numel() for p in self.actor_network.parameters())
        print(actor_total_params)
        print("critic network parameters number:")
        critic_total_params = sum(p.numel() for p in self.critic_network.parameters())
        print(critic_total_params)
        self.actor_network.Ptask.load_state_dict(data1[0])
        self.actor_network.Probot.load_state_dict(data2[1])
        self.critic_network.Qtask.load_state_dict(data1[2])
        self.critic_network.Qrobot.load_state_dict(data2[3])

        # sync the networks across the cpus
        sync_networks(self.actor_network)
        sync_networks(self.critic_network)
        # build up the target network
        self.actor_target_network = GausPiNetwork(num_task_inputs, args.task_hidden_dim, args.interface_dim, self.env.action_shape, num_robot_inputs, args.robot_hidden_dim, env_params, anchor_tensor)
        self.critic_target_network = QNetwork(num_task_inputs, args.task_hidden_dim, args.interface_dim, self.env.action_shape, num_robot_inputs, args.robot_hidden_dim, env_params, anchor_tensor)
        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())
        # to cpu or gpu
        self.actor_network.to(self.device)
        self.critic_network.to(self.device)
        self.actor_target_network.to(self.device)
        self.critic_target_network.to(self.device)

        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)
        # her sampler
        self.her_module = her_sampler(self.args.replay_strategy, self.args.replay_k, self.env.compute_reward)
        # create the replay buffer
        self.buffer = module_replay_buffer(self.env_params, self.args.buffer_size, self.her_module.sample_her_transitions)

        # critic network 2 for SAC
        self.critic_network_2 = QNetwork(num_task_inputs, args.task_hidden_dim, args.interface_dim, self.env.action_shape, num_robot_inputs, args.robot_hidden_dim, env_params, anchor_tensor)
        self.critic_network_2.Qtask.load_state_dict(data1[4])
        self.critic_network_2.Qrobot.load_state_dict(data2[5])
        sync_networks(self.critic_network_2)

        self.critic_target_network_2 = QNetwork(num_task_inputs, args.task_hidden_dim, args.interface_dim, self.env.action_shape, num_robot_inputs, args.robot_hidden_dim, env_params, anchor_tensor)
        # load the weights into the target networks
        self.critic_target_network_2.load_state_dict(self.critic_network_2.state_dict())
        # if use gpu
        self.critic_network_2.to(self.device)
        self.critic_target_network_2.to(self.device)

        # create the optimizer
        self.critic_optim_2 = torch.optim.Adam(self.critic_network_2.parameters(), lr=self.args.lr_critic)
        self._alpha = args.alpha
        self.target_entropy = -torch.prod(torch.Tensor(self.env.action_shape).to(self.device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=args.alpha_lr)
        self.sample_epoch = args.sample_epoch

    def learn(self):
        """
        train the network
        """
        reward_record = []
        success_rate_record = []
        # start to collect samples
        for epoch in range(self.args.n_epochs):
            for _ in range(self.args.n_cycles):
                mb_obs, mb_ag, mb_g, mb_actions, mb_joins = [], [], [], [], []
                # start to collect samples, with eval mode
                self.actor_network.eval()
                for _ in range(self.args.num_rollouts_per_mpi):
                    # reset the rollouts
                    ep_obs, ep_ag, ep_g, ep_actions, ep_joins = [], [], [], [], []
                    # reset the environment
                    observation = self.env.reset()
                    obs = observation['observation']
                    ag = observation['achieved_goal']
                    g = observation['desired_goal']
                    joins = observation['joint_pos']
                    # start to collect samples
                    for t in range(self.env_params['max_timesteps']):
                        with torch.no_grad():
                            task_input_tensor, joins_tensor = self._preproc_inputs(obs, g, joins)
                            if epoch < self.sample_epoch:
                                pi, _ = self.actor_network(task_input_tensor, joins_tensor)
                            else:
                                pi, _, _ = self.actor_network.sample(task_input_tensor, joins_tensor)
                            action = self._select_actions(pi)
                            # print(action)
                        # feed the actions into the environment
                        observation_new, _, _, info = self.env.step(action, control_method=self.args.control_type)
                        obs_new = observation_new['observation']
                        ag_new = observation_new['achieved_goal']
                        joins_new = observation_new['joint_pos']
                        # append rollouts
                        ep_obs.append(obs.copy())
                        ep_ag.append(ag.copy())
                        ep_g.append(g.copy())
                        ep_actions.append(action.copy())
                        ep_joins.append(joins.copy())
                        # re-assign the observation
                        obs = obs_new
                        ag = ag_new
                        joins = joins_new
                    ep_obs.append(obs.copy())
                    ep_ag.append(ag.copy())
                    ep_joins.append(joins.copy())
                    mb_obs.append(ep_obs)
                    mb_ag.append(ep_ag)
                    mb_g.append(ep_g)
                    mb_actions.append(ep_actions)
                    mb_joins.append(ep_joins)
                # convert them into arrays
                mb_obs = np.array(mb_obs)
                mb_ag = np.array(mb_ag)
                mb_g = np.array(mb_g)
                mb_actions = np.array(mb_actions)
                mb_joins = np.array(mb_joins)
                # store the episodes
                self.buffer.store_episode([mb_obs, mb_ag, mb_g, mb_actions, mb_joins])
                # start training, with train mode
                if epoch > (self.sample_epoch - 1):
                    self.actor_network.train()
                    for _ in range(self.args.n_batches):
                        # train the network
                        self._update_network()
                # soft update
                self._soft_update_target_network(self.actor_target_network, self.actor_network)
                self._soft_update_target_network(self.critic_target_network, self.critic_network)
                self._soft_update_target_network(self.critic_target_network_2, self.critic_network_2)
            # start to do the evaluation
            success_rate, mean_reward = self._eval_agent()
            # print(MPI.COMM_WORLD.Get_rank())
            if MPI.COMM_WORLD.Get_rank() == 0:
                print('[{}] epoch is: {}, eval success rate is: {:.3f}, reward is: {:.3f}'.format(datetime.now(), epoch, success_rate, mean_reward))
                success_rate_record.append(success_rate)
                reward_record.append(mean_reward)
                if epoch % 10 == 0 and self.args.save_data is True:
                    np.save(self.data_path + "/task_hidden_dim{}-interface_dim{}-robot_hidden_dim{}-{}_control-{}_cycles-seed{}-ta{}-ro{}-".format(
                        self.args.task_hidden_dim, self.args.interface_dim, self.args.robot_hidden_dim,
                        self.args.control_type, self.args.n_cycles, self.args.seed, self.args.ta_env_name, self.args.ro_env_name) + 'module_sac_push14_norm_few_shot_sr9', success_rate_record)
                    np.save(self.data_path + "/task_hidden_dim{}-interface_dim{}-robot_hidden_dim{}-{}_control-{}_cycles-seed{}-ta{}-ro{}-".format(
                        self.args.task_hidden_dim, self.args.interface_dim, self.args.robot_hidden_dim,
                        self.args.control_type, self.args.n_cycles, self.args.seed, self.args.ta_env_name, self.args.ro_env_name) + 'module_sac_push14_norm_few_shot_rd9', reward_record)

                if epoch % 5 == 0 and self.args.save_model is True:
                    torch.save(
                        [self.actor_network.Ptask.state_dict(), self.actor_network.Probot.state_dict(),
                         self.critic_network.Qtask.state_dict(), self.critic_network.Qrobot.state_dict(),
                         self.critic_network_2.Qtask.state_dict(), self.critic_network_2.Qrobot.state_dict()],
                        self.model_path + "/task_hidden_dim{}-interface_dim{}-robot_hidden_dim{}-{}_control-seed{}-ta{}-ro{}-epoch{}-".format(
                            self.args.task_hidden_dim, self.args.interface_dim, self.args.robot_hidden_dim,
                            self.args.control_type, self.args.seed, self.args.ta_env_name, self.args.ro_env_name, epoch) + 'module_sac_push14_norm_few_shot_full_model9.pt', _use_new_zipfile_serialization=False)

        if MPI.COMM_WORLD.Get_rank() == 0:
            if self.args.save_data is True:
                np.save(
                    self.data_path + "/task_hidden_dim{}-interface_dim{}-robot_hidden_dim{}-{}_control-{}_cycles-seed{}-ta{}-ro{}-".format(
                        self.args.task_hidden_dim, self.args.interface_dim, self.args.robot_hidden_dim,
                        self.args.control_type, self.args.n_cycles, self.args.seed, self.args.ta_env_name, self.args.ro_env_name) + 'module_sac_push14_norm_few_shot_sr9',
                    success_rate_record)
                np.save(self.data_path + "/task_hidden_dim{}-interface_dim{}-robot_hidden_dim{}-{}_control-{}_cycles-seed{}-ta{}-ro{}-".format(
                    self.args.task_hidden_dim, self.args.interface_dim, self.args.robot_hidden_dim,
                    self.args.control_type, self.args.n_cycles, self.args.seed, self.args.ta_env_name, self.args.ro_env_name) + 'module_sac_push14_norm_few_shot_rd9', reward_record)

            if self.args.save_model is True:
                torch.save(
                    [self.actor_network.Ptask.state_dict(), self.actor_network.Probot.state_dict(),
                     self.critic_network.Qtask.state_dict(), self.critic_network.Qrobot.state_dict(),
                     self.critic_network_2.Qtask.state_dict(), self.critic_network_2.Qrobot.state_dict()],
                    self.model_path + "/task_hidden_dim{}-interface_dim{}-robot_hidden_dim{}-{}_control-seed{}-ta{}-ro{}-epoch{}-".format(
                        self.args.task_hidden_dim, self.args.interface_dim, self.args.robot_hidden_dim,
                        self.args.control_type, self.args.seed, self.args.ta_env_name, self.args.ro_env_name, self.args.n_epochs-1) + 'module_sac_push14_norm_few_shot_full_model9.pt', _use_new_zipfile_serialization=False)

    # pre_process the inputs
    def _preproc_inputs(self, obs, g, joins):
        # concatenate the stuffs
        if self.args.env_name == "Ur5Reach" or self.args.env_name == "Ur5ReachSize2":
            # this order is to let robot take goal as the object that it wants to reach
            task_inputs = np.concatenate([g, obs])
        else:
            task_inputs = np.concatenate([obs, g])
        task_inputs = torch.tensor(task_inputs, dtype=torch.float32).unsqueeze(0)
        joins = torch.tensor(joins, dtype=torch.float32).unsqueeze(0)
        task_inputs = task_inputs.to(self.device)
        joins = joins.to(self.device)

        return task_inputs, joins

    # this function will choose action for the agent and do the exploration
    def _select_actions(self, pi):
        action = pi.cpu().numpy().squeeze()
        # add the gaussian
        action += self.args.noise_eps * self.env_params['action_max'] * np.random.randn(*action.shape)
        action = np.clip(action, -self.env_params['action_max'], self.env_params['action_max'])
        # random actions...
        random_actions = np.random.uniform(low=-self.env_params['action_max'], high=self.env_params['action_max'], \
                                           size=self.env_params['action'])
        # choose if use the random actions
        action += np.random.binomial(1, self.args.random_eps, 1)[0] * (random_actions - action)
        return action

    def _preproc_ogj(self, o, g, j):
        o = np.clip(o, -self.args.clip_obs, self.args.clip_obs)
        g = np.clip(g, -self.args.clip_obs, self.args.clip_obs)
        j = np.clip(j, -self.args.clip_obs, self.args.clip_obs)
        return o, g, j

    # soft update
    def _soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)

    # update the network
    def _update_network(self):
        # sample the episodes
        transitions = self.buffer.sample(self.args.batch_size)
        # pre-process the observation and goal
        o, o_next, g, joins, joins_next = transitions['obs'], transitions['obs_next'], transitions['g'], transitions['joins'], transitions['joins_next']
        transitions['obs'], transitions['g'], transitions['joins'] = self._preproc_ogj(o, g, joins)
        transitions['obs_next'], transitions['g_next'], transitions['joins_next'] = self._preproc_ogj(o_next, g, joins_next)
        # start to do the update
        obs = transitions['obs']
        joins = transitions['joins']
        g = transitions['g']
        if self.args.env_name == "Ur5Reach" or self.args.env_name == "Ur5ReachSize2":
            task_inputs = np.concatenate([g, obs], axis=1)
        else:
            task_inputs = np.concatenate([obs, g], axis=1)
        obs_next = transitions['obs_next']
        joins_next = transitions['joins_next']
        g_next = transitions['g_next']
        if self.args.env_name == "Ur5Reach" or self.args.env_name == "Ur5ReachSize2":
            task_inputs_next = np.concatenate([g_next, obs_next], axis=1)
        else:
            task_inputs_next = np.concatenate([obs_next, g_next], axis=1)
        # transfer them into the tensor
        task_inputs_tensor = torch.tensor(task_inputs, dtype=torch.float32)
        task_inputs_next_tensor = torch.tensor(task_inputs_next, dtype=torch.float32)
        joins_tensor = torch.tensor(joins, dtype=torch.float32)
        joins_next_tensor = torch.tensor(joins_next, dtype=torch.float32)
        actions_tensor = torch.tensor(transitions['actions'], dtype=torch.float32)
        r_tensor = torch.tensor(transitions['r'], dtype=torch.float32)

        task_inputs_tensor = task_inputs_tensor.to(self.device)
        task_inputs_next_tensor = task_inputs_next_tensor.to(self.device)
        joins_tensor = joins_tensor.to(self.device)
        joins_next_tensor = joins_next_tensor.to(self.device)
        actions_tensor = actions_tensor.to(self.device)
        r_tensor = r_tensor.to(self.device)

        # calculate the target Q value function
        with torch.no_grad():
            # concatenate the stuffs
            actions_next, next_state_logp, _ = self.actor_target_network.sample(task_inputs_next_tensor, joins_next_tensor)
            q_next_value = self.critic_target_network(task_inputs_next_tensor, joins_next_tensor, actions_next)
            q_next_value = q_next_value.detach()
            target_q_value = r_tensor + self.args.gamma * q_next_value
            target_q_value = target_q_value.detach()

            q_next_value_2 = self.critic_target_network_2(task_inputs_next_tensor, joins_next_tensor, actions_next)
            q_next_value_2 = q_next_value_2.detach()
            target_q_value_2 = r_tensor + self.args.gamma * q_next_value_2
            target_q_value_2 = target_q_value_2.detach()
            # clip the q value
            clip_return = 1 / (1 - self.args.gamma)
            # make target_q_value between -clip_return and 0
            target_q_value = torch.clamp(target_q_value, -clip_return, 0) - self._alpha * next_state_logp
            target_q_value_2 = torch.clamp(target_q_value_2, -clip_return, 0)
        # the q loss
        real_q_value = self.critic_network(task_inputs_tensor, joins_tensor, actions_tensor)
        critic_loss = (target_q_value - real_q_value).pow(2).mean()
        # the q loss 2
        real_q_value_2 = self.critic_network_2(task_inputs_tensor, joins_tensor, actions_tensor)
        critic_loss_2 = (target_q_value_2 - real_q_value_2).pow(2).mean()
        # the actor loss
        actions_real, logp, _ = self.actor_network.sample(task_inputs_tensor, joins_tensor)

        actor_loss = (self._alpha * logp - torch.min(
            self.critic_network(task_inputs_tensor, joins_tensor, actions_real),
            self.critic_network_2(task_inputs_tensor, joins_tensor, actions_real))).mean()

        actor_loss += self.args.action_l2 * (actions_real / self.env_params['action_max']).pow(2).mean()
        alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
        # start to update the network
        self.actor_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_network.parameters(), self.args.clip_grad)
        sync_grads(self.actor_network)
        self.actor_optim.step()
        # update the critic_network
        self.critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_network.parameters(), self.args.clip_grad)
        sync_grads(self.critic_network)
        self.critic_optim.step()
        self.critic_optim_2.zero_grad()
        critic_loss_2.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_network_2.parameters(), self.args.clip_grad)
        sync_grads(self.critic_network_2)
        self.critic_optim_2.step()
        # update alpha
        # alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        self._alpha = self.log_alpha.exp()

    # do the evaluation
    def _eval_agent(self):
        total_success_rate = []
        reward_list = []
        self.actor_network.eval()
        for _ in range(self.args.n_test_rollouts):
            per_success_rate = []
            observation = self.env.reset()
            obs = observation['observation']
            g = observation['desired_goal']
            joins = observation['joint_pos']
            reward_sum = 0
            reward = 0
            for _ in range(self.env_params['max_timesteps']):
                with torch.no_grad():
                    task_input_tensor, joins_tensor = self._preproc_inputs(obs, g, joins)
                    # pi, _, _ = self.actor_network.sample(task_input_tensor, joins_tensor)
                    pi, _ = self.actor_network(task_input_tensor, joins_tensor)
                    # convert the actions
                    actions = pi.detach().cpu().numpy().squeeze()
                observation_new, reward, _, info = self.env.step(actions, control_method=self.args.control_type)
                reward_sum = reward_sum + reward
                obs = observation_new['observation']
                g = observation_new['desired_goal']
                joins = observation_new['joint_pos']
                per_success_rate.append(info['is_success'])
            total_success_rate.append(per_success_rate)
            reward_list.append(reward_sum)
        total_success_rate = np.array(total_success_rate)
        local_success_rate = np.mean(total_success_rate[:, -1])
        global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM)

        reward_list = np.array(reward_list)
        local_reward = np.mean(reward_list)
        global_reward = MPI.COMM_WORLD.allreduce(local_reward, op=MPI.SUM)
        return global_success_rate / MPI.COMM_WORLD.Get_size(), global_reward / MPI.COMM_WORLD.Get_size()

    def visualize(self, model_path):
        # agent 2 is adversary agent
        total_success_rate = []
        data = torch.load(model_path, map_location=torch.device('cpu'))
        self.actor_network.Ptask.load_state_dict(data[0])
        self.actor_network.Probot.load_state_dict(data[1])
        self.actor_network.eval()

        for _ in range(10):
            per_success_rate = []
            observation = self.env.reset()
            obs = observation['observation']
            g = observation['desired_goal']
            joins = observation['joint_pos']
            for t in range(self.env_params['max_timesteps']):
                with torch.no_grad():
                    task_input_tensor, joins_tensor = self._preproc_inputs(obs, g, joins)
                    # pi = self.actor_network(task_input_tensor, joins_tensor)
                    # pi, _, _ = self.actor_network.sample(task_input_tensor, joins_tensor)
                    pi, _ = self.actor_network(task_input_tensor, joins_tensor)
                    pi = pi[0]
                    # print(pi_1)
                    # convert the actions
                    # pi_1 = torch.zeros_like(pi_2)
                    action = pi.detach().cpu().numpy().squeeze()
                observation_new, _, _, info = self.env.step(action, control_method=self.args.control_type)
                time.sleep(0.01)
                # self.env.render()
                obs = observation_new['observation']
                joins = observation_new['joint_pos']
                per_success_rate.append(info['is_success'])

            total_success_rate.append(per_success_rate)

        total_success_rate = np.array(total_success_rate)
        local_success_rate = np.mean(total_success_rate[:, -1])
        print(local_success_rate)

    def permutation_visualize(self, task_path, robot_path):
        # agent 2 is adversary agent
        total_success_rate = []
        total_touch_rate = []
        data1 = torch.load(task_path, map_location=torch.device('cpu'))
        data2 = torch.load(robot_path, map_location=torch.device('cpu'))
        self.actor_network.Ptask.load_state_dict(data1[0])
        self.actor_network.Probot.load_state_dict(data2[1])
        self.actor_network.eval()
        touch_flag = 0  # indicate robot touch the object
        touch_success = 0
        distance_threshold = 0.05
        ini_success_num = 0  # how many games is successfull at the very beginning
        ini_success_flag = 0
        test_num = 10

        for _ in range(test_num):
            ini_success_flag = 0
            per_success_rate = []
            observation = self.env.reset()
            obs = observation['observation']
            g = observation['desired_goal']
            if distance(obs, g) < distance_threshold:
                ini_success_num += 1
                ini_success_flag = 1
            joins = observation['joint_pos']
            # touch_flag = 0
            touch_success = 0
            # t_len = 0
            for t in range(self.env_params['max_timesteps']):
                ee_pos = p.getLinkState(self.env.robot.id, self.env.robot.eef_id)[0]
                obj_pos = obs
                d = distance(ee_pos, obj_pos)
                # if d <= distance_threshold:
                #     # print(t)
                #     touch_flag = 1
                #     t_len += 1
                #
                # if touch_flag == 1 and d > distance_threshold:
                #     touch_flag = 0
                #     if t_len > 0:
                #         touch_success = 1
                #     t_len = 0

                if d <= distance_threshold and ini_success_flag == 0:
                    touch_success = 1



                with torch.no_grad():
                    task_input_tensor, joins_tensor = self._preproc_inputs(obs, g, joins)
                    # task_input_tensor[0][3:] = 0
                    # print(task_input_tensor)
                    # pi = self.actor_network(task_input_tensor, joins_tensor)
                    # pi, _, _ = self.actor_network.sample(task_input_tensor, joins_tensor)
                    pi, _ = self.actor_network(task_input_tensor, joins_tensor)
                    pi = pi[0]
                    # print(pi_1)
                    # convert the actions
                    # pi_1 = torch.zeros_like(pi_2)
                    action = pi.detach().cpu().numpy().squeeze()
                observation_new, _, _, info = self.env.step(action, control_method=self.args.control_type)
                time.sleep(0.05)  #0.02
                # self.env.render()
                obs = observation_new['observation']
                joins = observation['joint_pos']
                per_success_rate.append(info['is_success'])

            total_touch_rate.append(touch_success)
            total_success_rate.append(per_success_rate)

        total_success_rate = np.array(total_success_rate)
        local_success_rate = np.mean(total_success_rate[:, -1])
        print(ini_success_num)
        print(total_touch_rate)
        local_touch_rate = np.sum(total_touch_rate) / (test_num - ini_success_num)
        # print(local_success_rate)
        print('Success rate is: {:.3f}, touch rate is: {:.3f}'.format(local_success_rate, local_touch_rate))

    def permutation_visualize_3s(self, task_path, robot_path):
        # do 3 seeds of permutation visualize and calculate mean and std
        test_success_rate_list = []
        touch_rate_list = []
        for i in range(5):
            # self.env.seed(self.args.seed + MPI.COMM_WORLD.Get_rank() + i)
            total_success_rate = []
            total_touch_rate = []
            data1 = torch.load(task_path, map_location=torch.device('cpu'))
            data2 = torch.load(robot_path, map_location=torch.device('cpu'))
            self.actor_network.Ptask.load_state_dict(data1[0])
            self.actor_network.Probot.load_state_dict(data2[1])
            self.actor_network.eval()
            touch_flag = 0  # indicate robot touch the object
            touch_success = 0
            distance_threshold = 0.05
            ini_success_num = 0  # how many games is successfull at the very beginning
            ini_success_flag = 0
            test_num = 200

            for _ in range(test_num):
                ini_success_flag = 0
                per_success_rate = []
                observation = self.env.reset()
                obs = observation['observation']
                g = observation['desired_goal']
                if distance(obs, g) < distance_threshold:
                    ini_success_num += 1
                    ini_success_flag = 1
                joins = observation['joint_pos']
                # touch_flag = 0
                touch_success = 0
                # t_len = 0
                for t in range(self.env_params['max_timesteps']):
                    ee_pos = p.getLinkState(self.env.robot.id, self.env.robot.eef_id)[0]
                    ee_pos = np.array(ee_pos) - [0, 0, 0.14 + 0.055 - 0.03]
                    obj_pos = obs
                    d = distance(ee_pos, obj_pos)

                    if d <= distance_threshold and ini_success_flag == 0:
                        touch_success = 1

                    with torch.no_grad():
                        task_input_tensor, joins_tensor = self._preproc_inputs(obs, g, joins)
                        pi, _ = self.actor_network(task_input_tensor, joins_tensor)
                        pi = pi[0]
                        action = pi.detach().cpu().numpy().squeeze()
                    observation_new, _, _, info = self.env.step(action, control_method=self.args.control_type)
                    # time.sleep(0.02)
                    # self.env.render()
                    obs = observation_new['observation']
                    joins = observation_new['joint_pos']
                    per_success_rate.append(info['is_success'])

                total_touch_rate.append(touch_success)
                total_success_rate.append(per_success_rate)

            total_success_rate = np.array(total_success_rate)
            local_success_rate = np.mean(total_success_rate[:, -1])
            print(ini_success_num)
            print(total_touch_rate)
            local_touch_rate = np.sum(total_touch_rate) / (test_num - ini_success_num)
            # print(local_success_rate)
            print('Success rate is: {:.3f}, touch rate is: {:.3f}'.format(local_success_rate, local_touch_rate))
            test_success_rate_list.append(local_success_rate)
            touch_rate_list.append(local_touch_rate)
        print("========================")
        print("========================")
        mean_success_rate = np.mean(test_success_rate_list)
        mean_touch_rate = np.mean(touch_rate_list)
        std_success_rate = np.std(test_success_rate_list)
        std_touch_rate = np.std(touch_rate_list)
        print('The mean success rate is: {:.3f}, and its std is: {:.3f}'.format(mean_success_rate, std_success_rate))
        print('The mean touching rate is: {:.3f}, and its std is: {:.3f}'.format(mean_touch_rate, std_touch_rate))

    def permutation_measure(self, task_path, robot_path):
        total_success_rate = []
        task_state_list = []
        # data1 = torch.load(task_path, map_location=torch.device('cpu'))
        # data2 = torch.load(robot_path, map_location=torch.device('cpu'))
        data1 = torch.load(task_path, map_location=self.device)
        data2 = torch.load(robot_path, map_location=self.device)
        self.actor_network.Ptask.load_state_dict(data1[0])
        self.actor_network.Probot.load_state_dict(data2[1])
        self.actor_network.eval()

        for _ in range(200):
            per_success_rate = []
            observation = self.env.reset()
            obs = observation['observation']
            g = observation['desired_goal']
            joins = observation['joint_pos']
            for t in range(self.env_params['max_timesteps']):
                with torch.no_grad():
                    task_input_tensor, joins_tensor = self._preproc_inputs(obs, g, joins)
                    if t > 4 and t < 10:
                        # print(task_input_tensor.detach().cpu().tolist())
                        task_state_list.append(task_input_tensor.detach().cpu().tolist()[0])
                    pi, _ = self.actor_network(task_input_tensor, joins_tensor)
                    pi = pi[0]
                    # print(pi_1)
                    # convert the actions
                    # pi_1 = torch.zeros_like(pi_2)
                    action = pi.detach().cpu().numpy().squeeze()
                observation_new, _, _, info = self.env.step(action, control_method=self.args.control_type)
                time.sleep(0.02)
                # self.env.render()
                obs = observation_new['observation']
                joins = observation_new['joint_pos']
                per_success_rate.append(info['is_success'])

            total_success_rate.append(per_success_rate)

        total_success_rate = np.array(total_success_rate)
        local_success_rate = np.mean(total_success_rate[:, -1])
        # print(local_success_rate)
        print('Success rate is: {:.3f}'.format(local_success_rate))
        print(np.array(task_state_list).shape)
        np.save('saved_anchors/' + self.args.env_name + '/' + self.args.env_name + '_task_state_2000list9', np.array(task_state_list))
