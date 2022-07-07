import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pandas as pd
import numpy as np
import argparse
import os
import glob
import json
import time
from collections import defaultdict


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


def pretty2(text):
    """Convert a string into a consistent format for 
    presentation in a matplotlib pyplot

    this version looks like: one_two_three_four

    """

    return pretty(text).replace("/", "_").replace(" ", "_").lower()


if __name__ == "__main__":

    plt.rcParams['text.usetex'] = False
    matplotlib.rc('font', family='Times New Roman', serif='cm10')
    matplotlib.rc('mathtext', fontset='cm')

    fig, axs = plt.subplots(1, 3, figsize=(28, 8))

    data = pd.read_csv("generalization.csv")

    for i, morphology in enumerate(["cheetah", "walker", "humanoid"]):

        df = data[data['env'].str.contains(morphology)]

        g = sns.barplot(data=df, 
                        x="method", 
                        y="rewards",
                         linewidth=4, ax=axs[i])

        axis = g
        axis.set(xlabel=None)
        axis.set(ylabel=None)
        
        axis.spines['right'].set_visible(False)
        axis.spines['top'].set_visible(False)

        axis.yaxis.set_ticks_position('left')
        axis.xaxis.set_ticks_position('bottom')

        axis.yaxis.set_tick_params(labelsize=16)
        axis.xaxis.set_tick_params(labelsize=16)

        axis.set_xlabel("Method", fontsize=24, 
                        fontweight='bold', labelpad=12)
        
        if i == 0:
            axis.set_ylabel("Average Return", fontsize=24, 
                            fontweight='bold', labelpad=12)

        axis.set_title(pretty(morphology) + "s", 
                        fontsize=24, fontweight='bold', pad=12)

        axis.grid(color='grey', linestyle='dotted', linewidth=2)

        plt.tight_layout(pad=3.0)

        plt.savefig("section_three.pdf")
        plt.savefig("section_three.png")

