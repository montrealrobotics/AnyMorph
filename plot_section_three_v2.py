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

    fig, axs = plt.subplots(2, 4, figsize=(28, 12))

    data = pd.read_csv("generalization.csv")
    data = pd.concat([
        data[data['method'].str.contains("Ours")],
        data[data['method'].str.contains("Amorpheus")],
        data[data['method'].str.contains("Shared Modular Policies")],
    ])

    for i, morphology in enumerate(["Cheetahs", "Walkers", 
                                    "Humanoids", "Walker Humanoids", 
                                    "Walker Humanoid Hoppers", 
                                    "Cheetah Walker Humanoids", 
                                    "Cheetah Walker Humanoid Hoppers"]):

        if i > 2:
            i += 1

        df = data[data['family'] == morphology]

        g = sns.barplot(data=df, 
                        x="method", 
                        y="rewards",
                        linewidth=4, ax=axs[i // 4, i % 4])

        axis = g
        axis.set(xlabel=None)
        axis.set(ylabel=None)
        
        axis.spines['right'].set_visible(False)
        axis.spines['top'].set_visible(False)

        axis.yaxis.set_ticks_position('left')
        axis.xaxis.set_ticks_position('bottom')
        axis.xaxis.set_ticks([])

        axis.yaxis.set_tick_params(labelsize=16)
        axis.xaxis.set_tick_params(labelsize=16)
        
        if i % 4 == 0:
            axis.set_ylabel("Average Return", fontsize=24, 
                            fontweight='bold', labelpad=12)

        axis.set_title(pretty(morphology), 
                        fontsize=24, fontweight='bold', pad=12)

        axis.grid(color='grey', linestyle='dotted', linewidth=2)

    g = sns.barplot(data=data, 
                    x="method", 
                    y="rewards",
                    linewidth=4, ax=axs[0, 3])

    axis = g
    axis.set(xlabel=None)
    axis.set(ylabel=None)
    
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)

    axis.yaxis.set_ticks_position('left')
    axis.xaxis.set_ticks_position('bottom')
    axis.xaxis.set_ticks([])

    axis.yaxis.set_tick_params(labelsize=16)
    axis.xaxis.set_tick_params(labelsize=16)

    axis.set_title("Overall", 
                    fontsize=24, fontweight='bold', pad=12)

    axis.grid(color='grey', linestyle='dotted', linewidth=2)

    legend = fig.legend(["Ours", "Amorpheus", "Shared Modular Policies"],
                        loc="lower center",
                        prop={'size': 24, 'weight':'bold'}, ncol=3)

    for i, legobj in enumerate(legend.legendHandles):
        legobj.set_linewidth(4.0)
        legobj.set_color(sns.color_palette()[i])

    plt.tight_layout(pad=3.0)
    fig.subplots_adjust(bottom=0.15)

    plt.savefig("section_three-2.pdf")
    plt.savefig("section_three-2.png")

