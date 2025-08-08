#!/usr/bin/env python3
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from log_parser import compute_io_throughput
import json
import scienceplots

# Use scienceplots for better aesthetics
plt.style.use(['science', 'no-latex'])

plt.rcParams.update({
    'font.size': 20,
    'axes.titlesize': 20,
    'axes.labelsize': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20,
    'figure.titlesize': 20
})

def get_backend_throughput(node_count, node_dir, backend, num_elements, dtype, operation='read'):
    """Get throughput for a specific backend in a node directory."""
    base_path = os.path.join(node_dir, f"logs_{backend}_{num_elements/1e6:.2f}m_1db_exp0")
    
    if not os.path.exists(base_path):
        return None,None
    
    # Choose log file based on operation
    if operation == 'read':
        log_file = os.path.join(base_path, "train_ai_0_datastore.log")
    elif operation == 'write':
        log_file = os.path.join(base_path, "sim_0_datastore.log")
    else:
        return None,None
    
    if not os.path.exists(log_file):
        return None,None
    
    throughput_data = compute_io_throughput(log_file, num_elements, dtype)
    # Extract node count from directory name
    try:
        total_throughput = throughput_data['mean_throughput_MB_per_sec']
        total_std = throughput_data.get('std_throughput_MB_per_sec', 0)
        return total_throughput, total_std
    except Exception as e:
        print(f"Error parsing {log_file}: {e}")
        return None, None

def collect_throughput_data(num_elements, dtype, backends=['redis', 'filesystem', 'dragon'],nodes=[8, 32, 128, 512, 2048]):
    """Collect throughput data for all backends and node counts."""
    
    data = {
        'read': {backend: {'nodes': [], 'throughput': [], 'std': []} for backend in backends},
        'write': {backend: {'nodes': [], 'throughput': [], 'std': []} for backend in backends}
    }

    for node_count in nodes:
        node_dir = os.path.abspath(f"../{node_count}nodes")
        print(f"\nProcessing {node_dir} ({node_count} nodes):")
        
        for backend in backends:
            # print(f"  Backend: {backend}")
            # Get read throughput (from train_ai_0.log)
            read_throughput, read_std = get_backend_throughput(node_count, node_dir, backend, num_elements, dtype, 'read')
            if read_throughput is not None:
                data['read'][backend]['nodes'].append(node_count)
                data['read'][backend]['throughput'].append(read_throughput/1024)  # Convert to GB/s
                data['read'][backend]['std'].append(read_std/1024)
                # print(f"  {backend} read: {read_throughput:.2f} MB/s, std: {read_std:.2f} MB/s")
            else:
                # print(f"  Backend: {backend}")
                print(f"  {backend} {num_elements/1e6:.2f} read: No data")
            
            # Get write throughput (from sim_0.log)
            write_throughput, write_std = get_backend_throughput(node_count, node_dir, backend, num_elements, dtype, 'write')
            if write_throughput is not None:
                data['write'][backend]['nodes'].append(node_count)
                data['write'][backend]['throughput'].append(write_throughput/1024)  # Convert to GB/s
                data['write'][backend]['std'].append(write_std/1024)
                # print(f"  {backend} write: {write_throughput:.2f} MB/s, std: {write_std:.2f} MB/s")
            else:
                # print(f"  Backend: {backend}")
                print(f"  {backend} {num_elements/1e6:.2f} write: No data")
    
    return data

def plot_throughput_comparison(data, backends=['redis', 'filesystem', 'dragon'], num_elements=1e6):
    # Compute global min/max for y-axis limits
    all_vals = []
    for op in ['read', 'write']:
        for backend in backends:
            if backend in data[op] and data[op][backend]['throughput']:
                all_vals.extend(data[op][backend]['throughput'])
    if all_vals:
        y_min = min(all_vals)
        y_max = max(all_vals)
    else:
        y_min, y_max = 0, 1
    """Create plot comparing throughput across backends for a single operation (read or write)."""
    # Use different linestyles for each backend, and different colors for read/write
    backend_linestyles = {
        'redis': '-',        # solid
        'filesystem': '--',  # dashed
        'nodelocal': ':',    # dotted
        'dragon': '-.'       # dash-dot
    }
    op_colors = {
        'read': 'tab:blue',
        'write': 'tab:orange'
    }
    marker = 'o'  # Use same marker for all
    fig, ax1 = plt.subplots(figsize=(8,6))
    ax2 = ax1.twinx()

    # Plot read throughput on ax1 (left y-axis)
    for backend in backends:
        if backend in data['read'] and data['read'][backend]['nodes']:
            ax1.plot(data['read'][backend]['nodes'], data['read'][backend]['throughput'],
                     color=op_colors['read'], linewidth=2,
                     marker=marker, linestyle=backend_linestyles[backend.split('_')[0]])

    # Plot write throughput on ax2 (right y-axis)
    for backend in backends:
        if backend in data['write'] and data['write'][backend]['nodes']:
            ax2.plot(data['write'][backend]['nodes'], data['write'][backend]['throughput'],
                     color=op_colors['write'], linewidth=2,
                     marker=marker, linestyle=backend_linestyles[backend.split('_')[0]])

    ax1.set_xscale('log')
    # ax1.set_yscale('log')
    ax1.set_xlabel('Number of Nodes')
    ax1.set_ylabel('Read Throughput per Process (GB/s)', color=op_colors['read'])
    ax2.set_ylabel('Write Throughput per Process (GB/s)', color=op_colors['write'])
    ax1.set_ylim(y_min, y_max)
    ax2.set_ylim(y_min, y_max)
    plt.title(f'Read & Write Throughput (size = {num_elements*4/1e6:.2f} MB)')

    # Custom legend: black lines with different linestyles for each backend
    import matplotlib.lines as mlines
    legend_lines = []
    legend_labels = []
    for backend in backends:
        line = mlines.Line2D([], [], color='black', linestyle=backend_linestyles[backend.split('_')[0]],
                             marker=marker, linewidth=2, label=backend.split('_')[0].capitalize())
        legend_lines.append(line)
        legend_labels.append(backend.split('_')[0].capitalize())
    ax1.legend(legend_lines, legend_labels, loc='best')

    ax1.grid(True)
    fig.tight_layout()
    plt.savefig(f'figs/throughput_comparison_{num_elements/1e6:.2f}m_read_write.png', dpi=300, bbox_inches='tight')
    print(f"Plot saved as 'figs/throughput_comparison_{num_elements/1e6:.2f}m_read_write.png'")
    plt.show()

def main():
    import argparse
    global OPERATION
    parser = argparse.ArgumentParser(description="Compare IO throughput across different backends.")
    parser.add_argument('--num_elements', type=int, default=1e5, help='Number of elements in the array (default: 319488)')
    parser.add_argument('--dtype', type=str, default='float32', help='Data type of the array elements (default: float32)')
    parser.add_argument('--backends', nargs='+', default=['redis', 'filesystem', 'dragon', "nodelocal"], 
                        help='List of backends to compare (default: redis filesystem dragon)')
    parser.add_argument('--operation', type=str, default='read', choices=['read', 'write'], help='Operation: read or write')
    args = parser.parse_args()
    OPERATION = args.operation
    num_elements = args.num_elements
    dtype = args.dtype
    backends = args.backends
    node_counts = [8, 32, 128, 512]
    print(f"Using {len(backends)} backends: {', '.join(backends)}")
    print(f"Collecting throughput data for {num_elements} elements of type {dtype}...")
    data = collect_throughput_data(num_elements, dtype, backends=backends, nodes=node_counts)
    # Filter out backends with no data
    filtered_data = {
        'read': {backend: values for backend, values in data['read'].items() if values['nodes']},
        'write': {backend: values for backend, values in data['write'].items() if values['nodes']}
    }
    if not filtered_data['read'] and not filtered_data['write']:
        print("No throughput data found!")
        return
    print(f"\nFound read data for backends: {list(filtered_data['read'].keys())}")
    print(f"Found write data for backends: {list(filtered_data['write'].keys())}")
    print("\nCreating throughput comparison plot...")
    plot_throughput_comparison(filtered_data, backends=backends, num_elements=num_elements)


if __name__ == "__main__":
    main()
