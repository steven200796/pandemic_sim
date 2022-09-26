# Confidential, Copyright 2020, Sony Corporation of America, All rights reserved.
import random
from tqdm import trange
import numpy as np

import pandemic_simulator as ps
from pandemic_simulator.environment.interfaces import InfectionSummary
from pandemic_simulator.environment.done import ORDone, DoneFunctionFactory, DoneFunctionType 


def init_pandemic_env():
    # init globals
    ps.init_globals(seed=112358)
    sim_config = ps.sh.small_town_config
    done_threshold = sim_config.max_hospital_capacity * 3
    done_fn = ORDone(
            done_fns=[
                DoneFunctionFactory.default(
                    DoneFunctionType.INFECTION_SUMMARY_ABOVE_THRESHOLD,
                    summary_type=InfectionSummary.CRITICAL,
                    threshold=done_threshold,
                ),
                DoneFunctionFactory.default(DoneFunctionType.NO_PANDEMIC, num_days=40),
            ]
        )

    env = ps.env.PandemicGymEnv3Act.from_config(sim_config=sim_config, 
                                                pandemic_regulations=ps.sh.austin_regulations,
                                                done_fn=done_fn)
    # setup viz
    viz = ps.viz.GymViz.from_config(sim_config=sim_config)

    return env, viz

def eval_policy(policy, env, n_episodes=5):
    rets = []
    ep_lens = []
    for i in trange(n_episodes, desc='Simulating episode'):
        cumu_reward = 0
        obs = env.reset()
        done = False
        ep_len = 0
        while not done:
            action = policy(Batch(obs=obs, info=None))["act"]
            obs, reward, done, aux = env.step(action=action)
            cumu_reward += reward
            ep_len += 1
        rets.append(cumu_reward)
        ep_lens.append(ep_len)
    return np.mean(rets), np.std(rets), rets, ep_lens

def vis_policy(policy, env, viz):
    done = False
    obs = env.reset(flatten_obs=False)
    while not done:
        flattened_obs = env.flatten_obs(obs)
        action = policy(Batch(obs=flattened_obs, info=None))["act"]
        print("ACT IS ", action)
        obs, reward, done, aux = env.step(action=action, flatten_obs=False)
        viz.record((obs, reward))
    viz.plot()


if __name__ == '__main__':
    # TODO: consider combining Tutorials 12 and 14!
    n_eval_episodes = 5
    env, viz = init_pandemic_env()
    env.seed(111111)
    POLICY_NAME = "" # change this to the name of your run

    from tianshou_utils import get_args, make_agent
    from tianshou.data import Batch
    import torch

    args = get_args()
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n

    policy, _ = make_agent(args)
    policy.load_state_dict(torch.load(f"log/pansim/{POLICY_NAME}/policy.pth"))

    mean_rets, std_rets, rets, ep_lens = eval_policy(policy, env, n_eval_episodes)
    print(f"MEAN/STD RETURN OF POLICY: {mean_rets}, {std_rets}\n")
    vis_policy(policy, env, viz)
    