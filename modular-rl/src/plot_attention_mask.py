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


def plot_attention(args):

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

    fig, axs = plt.subplots(1, 1, figsize=(12, 4))

    # visualize ===========================================================
    for i, env_name in enumerate(env_names):

        # create env
        env = utils.makeEnvWrapper(env_name, seed=args.seed, obs_max_len=None)()
        policy.change_morphology(args.graphs[env_name], args.action_ids[env_name])

        attentions = []

        def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
            """Pass the inputs (and mask) through the decoder layer.

            Args:
                tgt: the sequence to the decoder layer (required).
                memory: the sequence from the last layer of the encoder (required).
                tgt_mask: the mask for the tgt sequence (optional).
                memory_mask: the mask for the memory sequence (optional).
                tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
                memory_key_padding_mask: the mask for the memory keys per batch (optional).

            Shape:
                see the docs in Transformer class.
            """
            tgt2, mask = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                                        key_padding_mask=tgt_key_padding_mask)
            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)
            tgt2, mask = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                            key_padding_mask=memory_key_padding_mask)
            attentions.append(mask.detach().cpu().numpy()[0])
            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
            tgt = tgt + self.dropout3(tgt2)
            tgt = self.norm3(tgt)
            return tgt

        policy.actor.actor.transformer.decoder.layers[-1].forward = \
            types.MethodType(forward, 
                policy.actor.actor.transformer.decoder.layers[-1])
        
        obs = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = policy.select_action(np.array(obs))
            obs, reward, done, _ = env.step(action)
            episode_reward += reward

        print("Episode Return: " + str(episode_reward), env_name)

        if "full" in env_name:

            np_attentions = np.stack(attentions, axis=0).max(axis=0)

            axis = axs

            axis.imshow(np_attentions, interpolation='nearest', aspect='auto')
                    
            axis.xaxis.set_ticks([])
            axis.yaxis.set_ticks([])

            axis.set_xlabel("Sensors", fontsize=24, 
                            fontweight='bold', labelpad=12)
            
            axis.set_ylabel("Actuators", fontsize=24, 
                            fontweight='bold', labelpad=12)
            
            axis.set_title(pretty(
                env_name), fontsize=24, fontweight='bold', pad=12)

    plt.tight_layout()

    plt.savefig("attention_masks.pdf")
    plt.savefig("attention_masks.png")


if __name__ == "__main__":
    plot_attention(get_args())
