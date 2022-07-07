from __future__ import print_function
import numpy as np
import torch
import os
import utils
import TD3
import pandas as pd
from arguments import get_args
import checkpoint as cp
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pandas as pd
from config import *


if __name__ == "__main__":

    args = get_args()

    plt.rcParams['text.usetex'] = False
    matplotlib.rc('font', family='Times New Roman', serif='cm10')
    matplotlib.rc('mathtext', fontset='cm')

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

    model_name = f"model_6.pyth"
    cp.load_model_only(exp_path, policy, model_name)

    obs_embeddings = policy.actor.obs_embeddings.weight.detach().cpu().numpy().astype(np.float32)
    act_embeddings = policy.actor.act_embeddings.weight.detach().cpu().numpy().astype(np.float32)

    print(utils.GLOBAL_SET_OF_NAMES)

    from sklearn.decomposition import PCA

    obs_embeddings = PCA(n_components=2).fit_transform(obs_embeddings)
    act_embeddings = PCA(n_components=2).fit_transform(act_embeddings)

    colors = (sns.color_palette("bright") + sns.color_palette("pastel") + sns.color_palette("dark"))[:len(utils.GLOBAL_SET_OF_NAMES)]

    plt.clf()
    plt.scatter(obs_embeddings[:, 0], obs_embeddings[:, 1], c=[c for c in colors for i in range(args.limb_obs_size)])
    plt.savefig("obs_embeddings.png")


    plt.clf()
    plt.scatter(act_embeddings[:, 0], act_embeddings[:, 1])
    plt.savefig("act_embeddings.png")
