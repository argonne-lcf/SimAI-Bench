#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
from compare_throughput_num_elements import collect_throughput_data
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

def plot_horizontal_bar_per_node(all_data_read, all_data_write, node_counts, num_elements_list, backends):
    """
    For each node count, plot a grouped horizontal bar chart with read throughput to the left (negative), write to the right (positive).
    all_data_read, all_data_write: dict mapping num_elements to throughput data (as returned by collect_throughput_data)
    node_counts: list of node counts
    num_elements_list: list of num_elements values
    backends: list of backends
    """
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    for idx, node in enumerate(node_counts):
        plt.figure(figsize=(3.5, 2.7))
        n_groups = len(num_elements_list)
        n_bars = len(backends)
        bar_height = 0.9 / n_bars
        y = np.arange(n_groups)
        import matplotlib.patches as mpatches
        legend_handles = []
        for j, backend in enumerate(backends):
            read_vals = []
            write_vals = []
            write_err_vals = []
            read_err_vals = []
            for i, num_elements in enumerate(num_elements_list):
                # Read
                data_r = all_data_read[num_elements]
                if backend in data_r['read'] and node in data_r['read'][backend]['nodes']:
                    bidx = data_r['read'][backend]['nodes'].index(node)
                    read_vals.append(-data_r['read'][backend]['throughput'][bidx])  # negative for left
                    read_err_vals.append(-data_r['read'][backend]['std'][bidx])
                else:
                    read_vals.append(0)
                    read_err_vals.append(0)
                # Write
                data_w = all_data_write[num_elements]
                if backend in data_w['write'] and node in data_w['write'][backend]['nodes']:
                    bidx = data_w['write'][backend]['nodes'].index(node)
                    write_vals.append(data_w['write'][backend]['throughput'][bidx])
                    write_err_vals.append(data_w['write'][backend]['std'][bidx])
                else:
                    write_vals.append(0)
                    write_err_vals.append(0)
            # Plot read (left)
            plt.barh(y + j*bar_height, read_vals, bar_height, color=colors[j%len(colors)], alpha=0.85, hatch='//')
            # Error bars for read
            plt.errorbar(read_vals, y + j*bar_height, xerr=read_err_vals, fmt='none', ecolor=colors[j%len(colors)], capsize=2, elinewidth=1)
            # Plot write (right)
            plt.barh(y + j*bar_height, write_vals, bar_height, color=colors[j%len(colors)], alpha=0.85)
            # Error bars for write
            plt.errorbar(write_vals, y + j*bar_height, xerr=write_err_vals, fmt='none', ecolor=colors[j%len(colors)], capsize=2, elinewidth=1)
            if idx == 0:
                legend_handles.append(mpatches.Patch(color=colors[j%len(colors)], label=backends[j].capitalize(), linewidth=0.1, alpha=0.85))
                
        plt.ylabel('Data Size (MB)')
        plt.xlabel('Throughput per Process (GB/s)')
        plt.title(f'Read (left) and Write (right) Throughput')
        plt.yticks(y + bar_height*(n_bars-1)/2, [f'{ne*4/1e6:.2f}' for ne in num_elements_list])
        plt.gca().yaxis.set_minor_locator(plt.NullLocator())
        # Set symmetric x-ticks, but show only positive labels
        xticks = np.linspace(-2, 2, 9)
        xtick_labels = [f'{abs(tick):.1f}' for tick in xticks]
        plt.xticks(xticks, xtick_labels)
        plt.axvline(0, color='black', linewidth=1)
        plt.legend(handles=legend_handles, loc='upper left', title_fontsize=9, handleheight=0.7, handlelength=0.5)
        plt.tight_layout()
        plt.grid(axis='x')
        plt.xlim(-2,2)
        plt.savefig(f'figs/horizontal_bar_nodes_{node}.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'figs/horizontal_bar_nodes_{node}.png', dpi=300, bbox_inches='tight')
        print(f"Horizontal bar chart saved as 'figs/horizontal_bar_nodes_{node}.pdf'")
        plt.show()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Plot per-node horizontal bar charts for read and write throughput.")
    parser.add_argument('--dtype', type=str, default='float32', help='Data type of the array elements (default: float32)')
    parser.add_argument('--backends', nargs='+', default=['redis', 'filesystem', 'dragon', 'nodelocal'], help='List of backends to compare')
    parser.add_argument('--nodes', nargs='+', type=int, default=[8, 32, 128, 512], help='List of node counts')
    parser.add_argument('--num_elements_list', nargs='+', type=int, default=[100000, 500000, int(2e6), int(8e6)], help='List of num_elements values')
    args = parser.parse_args()
    dtype = args.dtype
    backends = args.backends
    node_counts = args.nodes
    num_elements_list = args.num_elements_list
    all_data_read = {}
    all_data_write = {}
    for ne in num_elements_list:
        all_data_read[ne] = collect_throughput_data(ne, dtype, backends=backends, nodes=node_counts)
        all_data_write[ne] = all_data_read[ne]  # same structure, but for clarity
    print("\nCreating horizontal bar chart for each node count...")
    plot_horizontal_bar_per_node(all_data_read, all_data_write, node_counts, num_elements_list, backends)

if __name__ == "__main__":
    main()
