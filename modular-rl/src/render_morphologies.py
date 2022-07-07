from __future__ import print_function
import numpy as np
import torch
import types
from typing import Optional, Any, Union, Callable
from torch import Tensor
import os
import utils
import TD3
import pandas as pd
from arguments import get_args
import checkpoint as cp
from config import *
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pandas as pd


def pretty(text):
    """Convert a string into a consistent format for 
    presentation in a matplotlib pyplot:

    this version looks like: One Two Three Four

    """

    text = text.replace("_", " ")
    text = text.replace("-", " ")
    text = text.strip()
    prev_c = None
    out_str = []
    for c in text:
        if prev_c is not None and \
                prev_c.islower() and c.isupper():
            out_str.append(" ")
            prev_c = " "
        if prev_c is None or prev_c == " ":
            c = c.upper()
        out_str.append(c)
        prev_c = c
    return "".join(out_str)


def plot_morphologies(args):

    exp_name = args.expID
    exp_path = os.path.join(DATA_DIR, exp_name)

    if not os.path.exists(exp_path):
        raise FileNotFoundError("checkpoint does not exist")
    print("*** folder fetched: {} ***".format(exp_path))

    # Retrieve MuJoCo XML files for visualizing ========================================
    env_names = []
    args.graphs = dict()
    args.action_ids = dict()

    # existing envs
    for name in os.listdir(args.custom_xml):
        if ".xml" in name:
            env_names.append(name[:-4])
            args.graphs[name[:-4]], args.action_ids[name[:-4]] = utils.getGraphStructure(
                os.path.join(args.custom_xml, name), args.observation_graph_type, 
                return_action_ids=True
            )

    print(utils.GLOBAL_SET_OF_NAMES)

    env_names.sort()

    # Set up env and policy ================================================
    args.limb_obs_size, args.max_action = utils.registerEnvs(
        env_names, args.max_episode_steps, args.custom_xml, use_restricted_obs=args.use_restricted_obs
    )
    # determine the maximum number of children in all the envs
    if args.max_children is None:
        args.max_children = utils.findMaxChildren(env_names, args.graphs)
    # setup agent policy
    policy = TD3.TD3(args)

    try:
        model_name = f"model_6.pyth"
        cp.load_model_only(exp_path, policy, model_name)
    except:
        raise Exception(
            "policy loading failed; check policy params (hint 1: max_children must be the same as the trained policy; hint 2: did the trained policy use torchfold (consider pass --disable_fold)?"
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    plt.rcParams['text.usetex'] = False
    matplotlib.rc('font', family='DejaVu Serif', serif='cm10')
    matplotlib.rc('mathtext', fontset='cm')

    fig, axs = plt.subplots(2, len(env_names) // 2, figsize=(18 * (len(env_names) // 2), 28))

    # visualize ===========================================================
    for i, env_name in enumerate(env_names):

        # create env
        env = utils.makeEnvWrapper(env_name, seed=args.seed, obs_max_len=None)()
        policy.change_morphology(args.graphs[env_name], args.action_ids[env_name])
            
        obs = env.reset()
        done = False
        episode_reward = 0
        step = 0

        while not done:
            action = policy.select_action(np.array(obs))
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            step += 1

            if step == 500:

                axis = axs[i // (len(env_names) // 2), i % (len(env_names) // 2)]

                axis.imshow(env.render(
                    mode='rgb_array', width=256, height=256)[::-1][96:])
                        
                axis.set_axis_off()
                
                axis.set_title(pretty(
                    env_name), fontsize=64, fontweight='bold', pad=64)

        print("Episode Return: " + str(episode_reward), env_name)

    plt.tight_layout(pad=12.0)

    plt.savefig("morphologies.pdf")
    plt.savefig("morphologies.png")

if __name__ == "__main__":
    plot_morphologies(get_args())
