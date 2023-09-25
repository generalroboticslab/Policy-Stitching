import argparse

"""
Here are the param for the training

"""

def get_args():
    parser = argparse.ArgumentParser()
    # the environment setting
    parser.add_argument('--env-name', type=str, default='PandaReach-v2', help='the environment name')
    parser.add_argument('--n-epochs', type=int, default=500, help='the number of epochs to train the agent')
    parser.add_argument('--n-cycles', type=int, default=50, help='the times to collect samples per epoch')
    parser.add_argument('--n-batches', type=int, default=40, help='the times to update the network')
    parser.add_argument('--save-interval', type=int, default=5, help='the interval that save the trajectory')
    parser.add_argument('--seed', type=int, default=101, help='random seed')
    parser.add_argument('--ta_seed', type=int, default=101, help='random seed for task module')
    parser.add_argument('--ro_seed', type=int, default=101, help='random seed for robot module')
    parser.add_argument('--num-workers', type=int, default=1, help='the number of cpus to collect samples')
    parser.add_argument('--replay-strategy', type=str, default='future', help='the HER strategy')
    parser.add_argument('--clip-return', type=float, default=50, help='if clip the returns')
    parser.add_argument('--net_type', type=int, default=2, help='type1 is original one, type2 has 64 task dim, type3 is the parallel one')
    parser.add_argument('--save-dir', type=str, default='saved_models/', help='the path to save the models')
    parser.add_argument('--data-dir', type=str, default='saved_data/', help='the path to save the reward and success rate')
    parser.add_argument('--noise-eps', type=float, default=0.2, help='noise eps')
    parser.add_argument('--random-eps', type=float, default=0.3, help='random eps')
    parser.add_argument('--buffer-size', type=int, default=int(1e6), help='the size of the buffer')
    parser.add_argument('--replay-k', type=int, default=4, help='ratio to be replace')
    parser.add_argument('--clip-obs', type=float, default=200, help='the clip ratio')
    parser.add_argument('--batch-size', type=int, default=256, help='the sample batch size')
    parser.add_argument('--gamma', type=float, default=0.98, help='the discount factor')
    parser.add_argument('--action-l2', type=float, default=1, help='l2 reg')
    parser.add_argument('--lr-actor', type=float, default=3e-4, help='the learning rate of the actor')  # sac
    parser.add_argument('--lr-critic', type=float, default=3e-4, help='the learning rate of the critic')  # sac
    parser.add_argument('--polyak', type=float, default=0.95, help='the average coefficient')
    parser.add_argument('--n-test-rollouts', type=int, default=10, help='the number of tests')
    parser.add_argument('--clip-range', type=float, default=5, help='the clip range')
    parser.add_argument('--robot-clip-range', type=float, default=100, help='the clip range for robot joints')
    parser.add_argument('--demo-length', type=int, default=20, help='the demo length')
    # parser.add_argument('--cuda', action='store_true', help='if use gpu do the acceleration')  # this is only used in module_sac_train to set cuda seed
    parser.add_argument('--num-rollouts-per-mpi', type=int, default=2, help='the rollouts per mpi')

    # parser.add_argument('--device', type=int, default=0, help='which gpu to use')
    parser.add_argument("--device", help="PyTorch device to be use (ex: cpu, cuda...)", default="cuda:0", type=str)  # cuda:0
    parser.add_argument('--task_hidden_dim', type=int, default=256, metavar='N',
                        help='task module hidden dimension (default: 64)')
    parser.add_argument('--robot_hidden_dim', type=int, default=256, metavar='N',
                        help='robot module hidden dimension (default: 64)')
    parser.add_argument('--interface_dim', type=int, default=128, metavar='N',
                        help='interface dimension (default: 16)')  # 16
    parser.add_argument('--alpha', type=float, default=0.2, help='alpha of sac')  # originally 0.2
    parser.add_argument('--alpha_lr', type=float, default=3e-4, help='alpha lr of sac')
    parser.add_argument("--control_type", help="Control the end effector or joints (ex: 'ee', 'joints')",
                        default='joints', type=str)
    # parser.add_argument('--save_data', type=bool, default=False, help='if save the success rate and reward')
    parser.add_argument('--save_data', action='store_true', help='if save the success rate and reward')
    # parser.add_argument('--save_model', type=bool, default=False, help='if save the model')
    parser.add_argument('--save_model', action='store_true', help='if use gpu do the acceleration')
    parser.add_argument('--ro-env-name', type=str, default='PandaPush-v2', help='the environment name for pretrained robot module')
    parser.add_argument('--ta-env-name', type=str, default='PandaPush-v3', help='the environment name for pretrained task module')
    parser.add_argument('--fix_ro', type=bool, default=False, help='if fix the robot module')
    parser.add_argument('--ro_fixed_init', type=bool, default=False, help='if fix the robot module')
    parser.add_argument('--fixedR_unfixedT', type=bool, default=False, help='if fix the robot module and load unfixed task module')
    parser.add_argument('--fixedD_unfixedU_EtoE', type=bool, default=False, help='if fix the down module and load unfixed up module in EtoE training')
    parser.add_argument('--fixedRT', type=bool, default=False, help='if fix the robot module and the task module')
    parser.add_argument('--fixedDU_EtoE', type=bool, default=False, help='if fix the down module and up module in EtoE training')
    parser.add_argument('--measure_permutation', type=str, default='none', help='tasks, joints')
    parser.add_argument('--test_standa', type=bool, default=False, help='test the model with standard interface')
    parser.add_argument('--clip_grad', type=float, default=1.0, help='clip the gradient')
    parser.add_argument('--anchor_mode', type=str, default='org',
                        help='anchor mode (org, kmeans_centroids_mix23, kmeans_real_mix23, kmeans_sphe_real_mix23, kmeans_large_real_mix23, kmeans_joints11)')
    parser.add_argument('--sample_epoch', type=int, default=10, metavar='N', help='sampling trajectories without update')
    parser.add_argument('--load_pre', type=bool, default=False, help='load previously trained model and continual train')
    # for larger anchor network
    parser.add_argument('--task_interface_dim', type=int, default=256, metavar='N', help='task interface dimension (default: 256)')
    parser.add_argument('--robot_interface_dim', type=int, default=384, metavar='N', help='task interface dimension (default: 384)')

    args = parser.parse_args()

    return args
