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

    parser = argparse.ArgumentParser()
    parser.add_argument("logdir", type=str)
    parser.add_argument("--round", type=int, default=20000)
    parser.add_argument("--max-steps", type=int, default=3e6 + 1)
    parser.add_argument("--require-min-trials", action='store_true')
    args = parser.parse_args()

    plt.rcParams['text.usetex'] = False
    matplotlib.rc('font', family='Times New Roman', serif='cm10')
    matplotlib.rc('mathtext', fontset='cm')

    fig, axs = plt.subplots(1, 4, figsize=(28, 8))

    def load_json(file):
        for i in range(10):
            try:
                with open(file, "r") as f:
                    return json.load(f)
            except json.decoder.JSONDecodeError:
                time.sleep(0.5)
        return None

    for exp_i, exp_name in enumerate([
            "walker-humanoids",
            "walker-humanoid-hoppers",
            "cheetah-walker-humanoids",
            "cheetah-walker-humanoid-hoppers"]):

        pandas_df = pd.DataFrame(columns=["time_steps", "rewards", "env", "method"])

        for method_name, exp_path in zip([
            "Ours", "Amorpheus", "SMP"], [
            "restricted-" + exp_name + "-0.00005-0.003-universal-?/*/metrics.json", 
            "restricted-" + exp_name + "-amorpheus-?/*/metrics.json",
            "restricted-" + exp_name + "-smp-?/*/metrics.json"]):

            files = glob.glob(os.path.join(args.logdir, exp_path))

            parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(files[0])))

            progress = []
            variants = []
            for f in files:
                result = load_json(f)
                if result is not None:
                    progress.append(result)
                    variants.append(load_json(
                        os.path.join(os.path.dirname(f), "config.json")))

            max_time_step = 1e15
            max_points = 1e15

            for progress_file, variant_file in zip(progress, variants):

                max_time_step = min(max_time_step, args.round * round(
                    progress_file["total_timesteps"]["values"][-1] / args.round))

                samples_per_bin = defaultdict(int)

                for step in progress_file["total_timesteps"]["values"]:
                    step = args.round * round(step / args.round)
                    samples_per_bin[step] += 1

                max_points = min(max_points, min(list(samples_per_bin.values())[:-1]))

            if not args.require_min_trials:
                max_time_step = 1e15

            if args.max_steps > 0:
                max_time_step = min(max_time_step, args.max_steps)

            for progress_file, variant_file in zip(progress, variants):

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
                                        method=method_name,
                                        env=key.replace("_episode_reward", 
                                                        "")), ignore_index=True)

                        print(key, pandas_df)

        data = pandas_df

        g = sns.lineplot(data=data, 
                         x="time_steps", 
                         y="rewards", hue="method",
                         linewidth=4, ax=axs[exp_i % 4])

        axis = g
        axis.set(xlabel=None)
        axis.set(ylabel=None)
        axis.get_legend().remove()
        
        axis.spines['right'].set_visible(False)
        axis.spines['top'].set_visible(False)

        axis.yaxis.set_ticks_position('left')
        axis.xaxis.set_ticks_position('bottom')

        axis.yaxis.set_tick_params(labelsize=16)
        axis.xaxis.set_tick_params(labelsize=16)

        if exp_i // 4 == 0:
            axis.set_xlabel("Time Steps", fontsize=24, 
                            fontweight='bold', labelpad=12)
        
        if exp_i % 4 == 0:
            axis.set_ylabel("Average Return", fontsize=24, 
                            fontweight='bold', labelpad=12)

        axis.set_title(pretty(exp_name.replace("humanoids-hopper", 
                                               "humanoid-hoppers")), 
                        fontsize=24, fontweight='bold', pad=12)

        axis.grid(color='grey', linestyle='dotted', linewidth=2)

    legend = fig.legend(["Ours", "Amorpheus", "Shared Modular Policies"],
                        loc="lower center",
                        prop={'size': 24, 'weight':'bold'}, ncol=3)

    for legobj in legend.legendHandles:
        legobj.set_linewidth(4.0)

    plt.tight_layout(pad=3.0)
    fig.subplots_adjust(bottom=0.3)

    plt.savefig(os.path.join(parent_dir, "section_two.pdf"))
    plt.savefig(os.path.join(parent_dir, "section_two.png"))

