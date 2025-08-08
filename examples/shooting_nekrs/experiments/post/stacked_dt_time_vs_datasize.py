
import os
import numpy as np
import matplotlib.pyplot as plt
from log_parser import get_mean_times
import scienceplots
plt.style.use(['science', 'no-latex'])
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 10,
    'axes.labelsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 10
})

def plot_stacked_dt_time_vs_datasize(node_count, backends, num_elements_list, dtype='float32', mode_labels=None):
    """
    For each backend and data size, plot stacked bars of mean tstep time and mean dt time for read and write.
    log_dir: directory containing log files, assumed to be named as '{backend}_read_{num_elements}.log' and '{backend}_write_{num_elements}.log'
    backends: list of backend names
    num_elements_list: list of data sizes (number of elements)
    dtype: data type string
    mode_labels: optional dict for labeling 'read' and 'write' bars
    """
    if mode_labels is None:
        mode_labels = {'read': 'AI', 'write': 'Sim'}
    width = 0.35  # Width of the bars
    x = np.arange(len(num_elements_list))
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    hatch_styles = ['//', '', 'xx', '']
    for backend_idx, backend in enumerate(backends):
        tstep_times = {'read': [], 'write': []}
        dt_times = {'read': [], 'write': []}
        for mode in ['read', 'write']:
            for ne in num_elements_list:
                log_file = os.path.join(f"../{node_count}nodes", f"logs_{backend}_{ne/1e6:.2f}m_1db_exp0",f"{'train_ai_0_datastore.log' if mode == 'read' else 'sim_0_datastore.log'}")
                if os.path.exists(log_file):
                    print(f"Processing {log_file} for {mode} mode")
                    print(get_mean_times(log_file, mode)["mean"])
                    dt_times[mode].append(get_mean_times(log_file, mode)["mean"])
                    tstep_times[mode].append(get_mean_times(log_file, "iteration")["mean"])
                else:
                    tstep_times[mode].append(0)
                    dt_times[mode].append(0)
        fig, ax = plt.subplots(figsize=(1.75, 2.7))
        # Stacked bars for read (left) and write (right)
        rects1 = ax.bar(x - width/2, tstep_times['read'], width, label=f"{mode_labels['read']} iter", color=colors[0], alpha=0.85, hatch=hatch_styles[0], edgecolor='black')
        rects2 = ax.bar(x - width/2, dt_times['read'], width, bottom=tstep_times['read'], label=f"read", color=colors[1], alpha=0.85, hatch=hatch_styles[1], edgecolor='black')
        rects3 = ax.bar(x + width/2, tstep_times['write'], width, label=f"{mode_labels['write']} iter", color=colors[2], alpha=0.85, hatch=hatch_styles[2], edgecolor='black')
        rects4 = ax.bar(x + width/2, dt_times['write'], width, bottom=tstep_times['write'], label=f"write", color=colors[3], alpha=0.85, hatch=hatch_styles[3], edgecolor='black')
        ax.set_ylabel('Mean Time per Event (s)')
        ax.set_xlabel('Data Size (MB)')
        ax.set_title(f'{backend.capitalize()}')#, {node_count} nodes')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{ne*4/1e6:.1f}' for ne in num_elements_list])
        # ax.set_ylim([0,1.5])
        if backend == "nodelocal":
            ax.legend(loc='best', handlelength=1.0, handleheight=0.5, handletextpad=0.1, labelspacing=0.1)
        plt.tight_layout()
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        outname = f'figs/stacked_dt_time_{backend}_{node_count}nodes.png'
        plt.savefig(outname, dpi=300, bbox_inches='tight')
        plt.savefig(outname.replace("png","pdf"), dpi=300, bbox_inches='tight')
        print(f"Saved {outname}")
        plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot stacked bar of mean tstep and dt time for read/write.")
    parser.add_argument('--nodes', nargs='+', default=[8, 32, 128, 512], help='Node count (single value, e.g. 8)')
    parser.add_argument('--backends', nargs='+', default=['redis', 'filesystem', 'dragon', 'nodelocal'], help='List of backends')
    parser.add_argument('--num_elements_list', nargs='+', type=int, default=[100000, 500000, int(2e6), int(8e6)], help='List of num_elements values')
    parser.add_argument('--dtype', type=str, default='float32', help='Data type of the array elements')
    args = parser.parse_args()
    for node_count in args.nodes:
        plot_stacked_dt_time_vs_datasize(node_count, args.backends, args.num_elements_list, args.dtype)
