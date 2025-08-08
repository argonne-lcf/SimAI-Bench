#!/usr/bin/env python3
"""
Timeline visualization for AI/Sim logs.
Plots timesteps and data transport events (where actual dt time > 0) for multiple log files.
"""

import re
import argparse
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import os
import scienceplots
from log_parser import *
# Use scienceplots for better aesthetics
plt.style.use(['science', 'no-latex'])

plt.rcParams.update({
    'font.size': 8,
    'axes.titlesize': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.titlesize': 8
})

def plot_timelines(logfiles, labels=None, output=None):
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    fig, ax = plt.subplots(figsize=(3.5, 1))
    
    sim_log = os.path.join(os.path.dirname(logfiles[0]), "sim_0_datastore.log")
    start_time = get_start_time(sim_log)
    for idx, file in enumerate(logfiles):
        events = parse_log_file(file)
        with open(file, 'r') as f:
            first_line = f.readline().strip()
            init_timestamp = parse_timestamp(first_line)

        if not events:
            print(f"No events found in {file}")
            continue
            
        # start_time = events[0]['timestamp']
        rel_times = [(e['timestamp'] - start_time).total_seconds() for e in events]
        tstep_times = [e['tstep_time'] for e in events]
        dt_times = [e['actual_dt_time'] for e in events]
        init_time = (init_timestamp - start_time).total_seconds()
        init_duration = (rel_times[0] - init_time)
        # Get label for this file
        base_label = labels[idx] if labels and idx < len(labels) else file.split('/')[-1].replace('.log', '')
        
        # Plot timesteps as horizontal bars
        ax.barh(idx, tstep_times, left=rel_times, height=0.4, 
               color=colors[idx%len(colors)], alpha=0.7, 
               label=f"{base_label} Timesteps")
        
        # Plot data transport events overlaid on top
        transport_rel_times = []
        transport_dt_values = []
        
        for t, dt, rel in zip(tstep_times, dt_times, rel_times):
            if dt > 0:
                transport_rel_times.append(rel)
                transport_dt_values.append(dt)
                # Plot data transport as red bars on top of timesteps
                ax.barh(idx, 0.0005, left=rel, height=0.4, 
                       color='red', alpha=0.9, edgecolor='darkred', linewidth=1)
        
        if labels[idx] in ["Training", "Inference"]:
            ax.barh(idx, init_duration, left=init_time, height=0.4, color='gray', alpha=0.5)
        # Print statistics for this file
        total_events = len(events)
        transport_events = len(transport_rel_times)
        print(f"\n{base_label}:")
        print(f"  Total events: {total_events}")
        print(f"  Data transport events: {transport_events}")
        if transport_events > 0:
            print(f"  Avg transport duration: {np.mean(transport_dt_values):.3f}s")
            print(f"  Total transport time: {sum(transport_dt_values):.3f}s")
    
    # Set up y-axis labels - simplified to one row per file
    yticks = list(range(len(logfiles)))
    yticklabels = []
    for idx, file in enumerate(logfiles):
        base = labels[idx] if labels and idx < len(labels) else file.split('/')[-1].replace('.log', '')
        yticklabels.append(base)
    
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_xlabel('Time (s)')
    # ax.set_title('Timeline')
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_ylim(-0.3, len(logfiles)-0.7)
    ax.set_xlim(left=0,right = 25)
    
    # Create custom legend
    # from matplotlib.patches import Patch
    # legend_elements = []
    # for idx, file in enumerate(logfiles):
    #     base = labels[idx] if labels and idx < len(labels) else file.split('/')[-1].replace('.log', '')
    #     legend_elements.append(Patch(facecolor=colors[idx%len(colors)], alpha=0.7, label=f"{base} Timesteps"))
    # legend_elements.append(Patch(facecolor='red', alpha=0.9, label="Data Transport Events"))
    
    # ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    if output:
        plt.savefig(output, dpi=300, bbox_inches='tight')
        plt.savefig(output.replace("pdf","png"), dpi=300, bbox_inches='tight')
        print(f"Timeline plot saved to: {output}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot timelines for AI/Sim logs.')
    parser.add_argument('basedir', type=str, help='Base directory containing log files')
    parser.add_argument('-o', '--output', help='Output file for plot (optional)', default="figs/timeline_plot.pdf")
    args = parser.parse_args()
    logfiles = [os.path.join(args.basedir,"logs_redis_0.32m_1db_exp0", f) for f in ["sim_0_datastore.log","train_ai_0_datastore.log"]]#,"infer_ai_0.log", ]]
    labels = ["Simulation", "Training"]#, "Inference"]
    plot_timelines(logfiles, labels, "figs/all_timeline_plot.pdf")

if __name__ == '__main__':
    main()
