import argparse
import torch
from tianshou.data import Collector
from tianshou.policy import DQNPolicy
from tianshou.utils.net.common import Net


def get_args():
    parser = argparse.ArgumentParser()
    # experiment settings
    parser.add_argument('--task', type=str, default='pansim')
    parser.add_argument('--seed', type=int, default=112358)
    parser.add_argument('--device', type=str, 
        default='cuda' if torch.cuda.is_available() else 'cpu')
    # logging
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--expt_name', type=str, default="", help="optional additional name for logs")
    parser.add_argument('--log_interval', type=int, default=360, help="timesteps between logging returns")
    # dqn hyperparameter
    parser.add_argument('--eps_train_init', type=float, default=0.5)
    parser.add_argument('--eps_train_final', type=float, default=0.01)
    parser.add_argument('--eps_test', type=float, default=0.0)
    parser.add_argument('--buffer_size', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=0.013)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--n_step', type=int, default=4, )
    parser.add_argument('--target_update_freq', type=int, default=50, 
        help="update target network every <target_update_freq> updates")
    parser.add_argument('--n_epoch', type=int, default=8)
    parser.add_argument('--init_random_steps', type=int, default=1200)
    parser.add_argument('--step_per_epoch', type=int, default=3000)
    parser.add_argument('--step_per_collect', type=int, default=360)
    parser.add_argument('--update_per_step', type=float, default=0.0625)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--hidden_sizes', type=int, nargs='*',
        default=[64, 64])
    parser.add_argument('--dueling_q_hidden_sizes', type=int, nargs='*',
        default=[64, 64])
    parser.add_argument('--dueling_v_hidden_sizes', type=int, nargs='*',
        default=[64, 64])
    return parser.parse_args()

def make_agent(args):
    # specify network structure
    Q_param = {"hidden_sizes": args.dueling_q_hidden_sizes}
    V_param = {"hidden_sizes": args.dueling_v_hidden_sizes}
    net = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.hidden_sizes,
        device=args.device,
        dueling_param=(Q_param, V_param)
    ).to(args.device)
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    policy = DQNPolicy(
        net,
        optim,
        args.gamma,
        args.n_step,
        target_update_freq=args.target_update_freq
    )
    return policy, optim
