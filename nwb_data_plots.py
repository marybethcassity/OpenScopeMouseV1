#!/opt/anaconda3/envs/openscope_env_new/bin/python

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse

import os, glob
from pathlib import Path

from dandi import dandiapi
from pynwb import NWBHDF5IO
from dandi.dandiapi import DandiAPIClient
from nwbwidgets.view import default_neurodata_vis_spec
from pathlib import Path

import pynwb
from nwbwidgets import nwb2widget


from typing import Union, Iterator, Callable, Tuple, Dict
import os
from pathlib import Path
from dotenv import load_dotenv


def parse_arguments():
    parser = argparse.ArgumentParser(description='Analyze preferred metrics from NWB files')
    
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Path to directory containing NWB files'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='../results',
        help='Directory to save results'
    )
    parser.add_argument(
        '--no_plots',
        action='store_true',
        help='Skip generating plots'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print progress messages'
    )
    # Add more arguments here as needed

    return parser.parse_args()

# define functions to download files with a progress bar
MAX_CHUNK_SIZE = int(os.environ.get("DANDI_MAX_CHUNK_SIZE", 1024 * 1024 * 8))  

def get_download_file_iter_with_steps(
    file, chunk_size: int = MAX_CHUNK_SIZE
) -> Tuple[Callable[[int], Iterator[bytes]], Dict[str, int]]:

    url = file.base_download_url
    steps_dict = {"total_steps": None}
    result = file.client.session.get(url, stream=True)

    total_size = int(result.headers.get('content-length', 0))
    steps_dict["total_steps"] = total_size // chunk_size
    print(f"Downloading {total_size} bytes in {steps_dict['total_steps']} steps")

    def downloader(start_at: int = 0) -> Iterator[bytes]:
        headers = None
        if start_at > 0:
            headers = {"Range": f"bytes={start_at}-"}
        result = file.client.session.get(url, stream=True, headers=headers)
        result.raise_for_status()
        for chunk in result.iter_content(chunk_size=chunk_size):
            if chunk:  
                yield chunk

    return downloader, steps_dict

def download_with_progressbar(
    file, filepath: Union[str, Path], chunk_size: int = MAX_CHUNK_SIZE
) -> None:
    downloader, steps_dict = get_download_file_iter_with_steps(file)
    with open(filepath, "wb") as fp:
        for chunk in tqdm(downloader(0), total=steps_dict["total_steps"], unit="chunk", unit_scale=True, unit_divisor=1024):
            fp.write(chunk)

def main():
    args = parse_arguments()
    print("Data directory:", args.data_dir)
    print("Output directory:", args.output_dir)
    print("Verbose:", args.verbose)

    download_loc = args.data_dir
    # open the downloaded NWB file
    ROOT_DIR = download_loc

    # Look for the first NWB file under the dandiset folder
    candidates = glob.glob(str(Path(ROOT_DIR) / "**" / "*.nwb"), recursive=True)
    if not candidates:
        raise FileNotFoundError("No .nwb file found under " + ROOT_DIR)
    nwb_path = candidates[0]
    print("Opening:", nwb_path)

    io = NWBHDF5IO(nwb_path, "r", load_namespaces=True)

    # number of neurons per probe per mouse that pass the threshold (r-squared of 0.5)
    # basic statistics of how many good receptive fields, plotted, percent of total receptive fields that passed per probe per mouse

    num = nwb_path.split("-")[1]
    subject_num = num.split("/")[0]
    print(f"Subject number: {subject_num}")

    nwb = io.read()

    # dict_keys([
    #   'drifting_gratings_field_block_presentations',
    #   'flash_field_block_presentations',
    #   'receptive_field_block_presentations',
    #   'spontaneous_presentations'
    # ])

    blocks = {
        "Gabor patches": "receptive_field_block_presentations",
        "Drifting gratings": "drifting_gratings_field_block_presentations",
        "Flash": "flash_field_block_presentations",
        "Spontaneous": "spontaneous_presentations",
    }

    pairing = {}
    stats = {}


    for i in blocks:
        pairing[i] = {"Trials": None, "Time": None}
        stats[i] = {}
        
        rf = nwb.intervals[blocks[i]] # this is where the gabor patches live
        df = rf.to_dataframe() # each row is 1 gabor stimulus representation
        gabor_trials_num = len(df)
        df['duration'] = df['stop_time'] - df['start_time']
        total_gabor_time = df['duration'].sum()
        print(f"{i} : {gabor_trials_num} trials = {total_gabor_time} minutes")
        pairing[i]["Trials"] = gabor_trials_num
        pairing[i]["Time"] = total_gabor_time

        # statistics for the csv file
        stats[i]["Median"] = df['duration'].median()
        stats[i]["Mean"] = df['duration'].mean()
        stats[i]["Max"] = df['duration'].max()
        stats[i]["Min"] = df['duration'].min()

    
    if args.verbose:
        print(stats)
    
    # create csv file of statistics
    df_stats = pd.DataFrame.from_dict(stats, orient='index')
    parent_dir = Path(args.output_dir)
    df_stats.to_csv(parent_dir / 'output.csv')
    
    if args.verbose:
        print(f"Saved output file to {parent_dir / 'output.csv'}")
    

    labels = list(pairing.keys())
    times = [round(pairing[label]["Time"]) for label in labels] # round the times just for simplicity
    if args.verbose:
        for i in range(len(labels)):
            print(f"{labels[i]}: {times[i]} minutes")

    colors = [
        "#C44E52",  # red
        "#55A868",  # green
        "#4C72B0",  # blue
        "#8172B3",  # purple
    ]

    plt.figure(figsize=(16, 2))
    plt.title(f"Ecephys: sub-{subject_num}", y = 1.65, fontsize=13, fontweight="bold", pad=20)

    left = 0  # where the current segment starts

    MIN_WIDTH_FOR_INSIDE = 100

    label_slots = [0.6, -0.6, 1.0, -1.0]
    used_slots = []

    for label, time, color in zip(labels, times, colors):
        plt.barh(
            y=0,
            width=time,
            left=left,
            color=color,
            edgecolor="none"
        )

        center_x = left + time / 2

        plt.axvline(
            x=left + time,
            color="gray",
            linewidth=2,
            zorder=5
        )

        if time >= MIN_WIDTH_FOR_INSIDE:
            # Inside label
            plt.text(
                center_x,
                0,
                label,
                ha="center",
                va="center",
                color="black",
                fontsize=10,
            )

        else:
            # Choose a free slot
            for slot in label_slots:
                if slot not in used_slots:
                    y_offset = slot
                    used_slots.append(slot)
                    break
            else:
                # Fallback: stack higher if all slots used
                y_offset = max(label_slots) + 0.4
                label_slots.append(y_offset)
                used_slots.append(y_offset)

            plt.annotate(
                label,
                xy=(left + time, 0),
                xytext=(left + time + 5, y_offset),
                ha="left",
                va="center",
                fontsize=10,
                arrowprops=dict(
                    arrowstyle="-",
                    color=color,
                    lw=1
                )
            )

        left += time
    plt.yticks([])
    plt.xlabel("Time (minutes)", fontweight = "bold")

    # add a summary box
    summary_text = ""
    summed = 0
    for i in pairing:
        summary_text += f"{i}: {pairing[i]['Trials']} Trials = {round(pairing[i]['Time'])} Minutes\n"
        summed += round(pairing[i]['Time'])
    summary_text += f"Total: {summed} Minutes"




    plt.text(
        0.5, 1.4,                      
        summary_text,
        transform=plt.gca().transAxes, 
        ha="center",
        va="center",
        fontsize=11,
        #fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="white",
            edgecolor="black"
        )
    )

    plt.savefig(parent_dir / f"{subject_num}_ecephys_plot.png", dpi=300, bbox_inches="tight")
    if args.verbose:
        print(f"Plot saved to {parent_dir / f"{subject_num}_ecephys_plot.png"}")
    
if __name__ == "__main__":
    main()