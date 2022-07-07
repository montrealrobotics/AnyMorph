import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pandas as pd
import numpy as np
import argparse
import os
import glob
import json
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

    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_pattern", type=str)
    parser.add_argument("--label", type=str, default="cheetah_variational")
    parser.add_argument("--title", type=str, default="Average Returns")
    parser.add_argument("--round", type=int, default=20000)
    parser.add_argument("--require-min-trials", action='store_true')
    args = parser.parse_args()

    plt.rcParams['text.usetex'] = False
    matplotlib.rc('font', family='Times New Roman', serif='cm10')
    matplotlib.rc('mathtext', fontset='cm')
    color_palette = ['#d73027', '#fc8d59', 
                     '#fee090', '#91bfdb', '#4575b4']
    palette = sns.color_palette(color_palette)
    sns.palplot(palette)
    sns.set_palette(palette)

    def load_json(file):
        with open(file, "r") as f:
            return json.load(f)

    files = glob.glob(args.experiment_pattern)

    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(files[0])))

    progress = [load_json(f) for f in files]
    variants = [load_json(os.path.join(os.path.dirname(f), "config.json")) for f in files]

    max_time_step = 1e15
    max_points = 1e15

    for progress_file, variant_file in zip(progress, variants):
        if variant_file["label"] == args.label:

            max_time_step = min(max_time_step, args.round * round(
                progress_file["total_timesteps"]["values"][-1] / args.round))

            samples_per_bin = defaultdict(int)

            for step in progress_file["total_timesteps"]["values"]:
                step = args.round * round(step / args.round)
                samples_per_bin[step] += 1

            max_points = min(max_points, min(list(samples_per_bin.values())[:-1]))

    if not args.require_min_trials:
        max_time_step = 1e15

    search_params = []

    def generate_search_keys(v):
        for key, value in v.items():
            if isinstance(value, dict):
                for inner_key in generate_search_keys(value):
                    yield key + "." + inner_key
            else:
                yield key

    for key in generate_search_keys(variants[0]):
        
        values_set = set()
        for sample in variants:

            value = sample
            for inner_key in key.split('.'):
                value = value[inner_key]

            if (
                isinstance(value, float) or
                isinstance(value, int) or
                isinstance(value, str)
            ) and value not in values_set:
                print(key, value)
                values_set.add(value)

        if len(values_set) > 1 \
                and key != "seed" and key != "expID":
            search_params.append(key)

    pandas_df = pd.DataFrame(columns=["time_steps", "rewards", "env"])

    for progress_file, variant_file in zip(progress, variants):
        extra_values = {extra_key: str(variant_file[extra_key]) for extra_key in search_params}
        if variant_file["label"] == args.label:

            time_steps = progress_file["total_timesteps"]["values"]

            for key, value in progress_file.items():
                if "reward" in key:

                    rewards = progress_file[key]["values"]
                    samples_per_bin = defaultdict(int)

                    for step, r in zip(time_steps, rewards):
                        step = args.round * round(step / args.round)
                        if step < max_time_step:

                            if samples_per_bin[step] < max_points:
                                samples_per_bin[step] += 1

                                pandas_df = pandas_df.append(dict(
                                    time_steps=step, 
                                    rewards=r, 
                                    env=key.replace("_episode_reward", ""),
                                    **extra_values), ignore_index=True)

                    print(key, pandas_df)

    data = pandas_df

    for plot_hue in ["env", *search_params]:

        plt.clf()
        plt.figure(figsize=(18, 7))

        g = sns.lineplot(data=data, 
                        x="time_steps", 
                        y="rewards", 
                        hue=plot_hue, 
                        linewidth=4)

        axis = g
        
        axis.spines['right'].set_visible(False)
        axis.spines['top'].set_visible(False)
        axis.yaxis.set_ticks_position('left')
        axis.xaxis.set_ticks_position('bottom')
        axis.yaxis.set_tick_params(labelsize=16)
        axis.xaxis.set_tick_params(labelsize=16)

        axis.set_xlabel("Time Steps", fontsize=24, fontweight='bold')
        axis.set_ylabel("Average Return", fontsize=24, fontweight='bold')
        axis.set_title(f"{args.title}", fontsize=24, fontweight='bold')

        axis.grid(color='grey', linestyle='dotted', linewidth=2)

        legend = plt.legend(bbox_to_anchor=(1.02, 0.65), 
                            loc='upper left', 
                            borderaxespad=0, 
                            prop={'size': 16, 'weight':'bold'})

        for legobj in legend.legendHandles:
            legobj.set_linewidth(4.0)

        plt.tight_layout()

        plt.savefig(os.path.join(
            parent_dir, 
            f"{pretty2(args.title)}-{args.label}-{plot_hue}.png"))

