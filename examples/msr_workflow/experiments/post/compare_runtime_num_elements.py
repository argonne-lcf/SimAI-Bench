#!/usr/bin/env python3
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from log_parser import *
import json
import scienceplots

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

def get_backend_runtime(node_count, node_dir, backend, operation='read', num_elements=None):
    """Get runtime for a specific backend in a node directory. Optionally, use num_elements in dir name."""
    if num_elements is not None:
        # Use logs_{backend}_{num_elements/1e6:.2f}m_1db_exp0
        base_path = os.path.join(node_dir, f"logs_{backend}_{num_elements/1e6:.2f}m_1db_exp0")
    else:
        base_path = os.path.join(node_dir, f"logs_{backend}")
    if not os.path.exists(base_path):
        return None, None
    # Choose log file based on operation
    if operation == 'read':
        log_file = os.path.join(base_path, "train_ai_0_datastore.log")
    elif operation == 'write':
        log_file = os.path.join(base_path, "sim_0_0_datastore.log")
    else:
        return None, None
    if not os.path.exists(log_file):
        return None, None
    try:
        runtime_sec = get_runtime_per_iteration(log_file)
        return runtime_sec["mean"], runtime_sec["std"]
    except Exception as e:
        print(f"Error parsing {log_file}: {e}")
        return None, None

def collect_runtime_data(backends=['redis', 'filesystem', 'dragon'], location='simulation', nodes=[8, 32, 128, 512, 2048]):
    """Collect runtime data for all backends and node counts."""
    backends = [f"{b}_{location}" if b != 'filesystem' else f"{b}_training" for b in backends]
    data = {
        'read': {backend: {'nodes': [], 'runtime': [], 'std': []} for backend in backends},
        'write': {backend: {'nodes': [], 'runtime': [], 'std': []} for backend in backends}
    }

    for node_count in nodes:
        node_dir = os.path.abspath(f"../{node_count}nodes")
        print(f"\nProcessing {node_dir} ({node_count} nodes):")
        
        for backend in backends:
            print(f"  Backend: {backend}")
            # Get read runtime (from train_ai_0_datastore.log)
            read_runtime, read_runtime_std = get_backend_runtime(node_count, node_dir, backend, 'read')
            if read_runtime is not None and read_runtime > 0:
                data['read'][backend]['nodes'].append(node_count)
                data['read'][backend]['runtime'].append(read_runtime)
                data['read'][backend]['std'].append(read_runtime_std)
                print(f"  {backend} read: {read_runtime:.2f} s (± {read_runtime_std:.2f} s)")
            else:
                print(f"  {backend} read: No data")
            # Get write runtime (from sim_0_0_datastore.log)
            write_runtime, write_runtime_std = get_backend_runtime(node_count, node_dir, backend, 'write')
            if write_runtime is not None and write_runtime > 0:
                data['write'][backend]['nodes'].append(node_count)
                data['write'][backend]['runtime'].append(write_runtime)
                data['write'][backend]['std'].append(write_runtime_std)
                print(f"  {backend} write: {write_runtime:.2f} s (± {write_runtime_std:.2f} s)")
            else:
                print(f"  {backend} write: No data")
    return data

# New: collect runtime vs num_elements for a given node count
def collect_vs_elements_runtime(node_count, num_elements_list, backends=['redis', 'filesystem', 'dragon'], location='simulation'):
    """Collect runtime vs num_elements for all backends at a given node count."""
    backends = [f"{b}_{location}" if b != 'filesystem' else f"{b}_training" for b in backends]
    data = {
        'read': {backend: {'num_elements': [], 'runtime': [], 'std': []} for backend in backends},
        'write': {backend: {'num_elements': [], 'runtime': [], 'std': []} for backend in backends}
    }
    node_dir = os.path.abspath(f"../{node_count}nodes")
    print(f"\nProcessing {node_dir} ({node_count} nodes):")
    for backend in backends:
        print(f"  Backend: {backend}")
        for num_elements in num_elements_list:
            read_runtime, read_runtime_std = get_backend_runtime(node_count, node_dir, backend, 'read', num_elements=num_elements)
            if read_runtime is not None and read_runtime > 0:
                data['read'][backend]['num_elements'].append(num_elements)
                data['read'][backend]['runtime'].append(read_runtime)
                data['read'][backend]['std'].append(read_runtime_std)
                print(f"    {num_elements:.0f} read: {read_runtime:.2f} s (± {read_runtime_std:.2f} s)")
            else:
                print(f"    {num_elements:.0f} read: No data")
            write_runtime, write_runtime_std = get_backend_runtime(node_count, node_dir, backend, 'write', num_elements=num_elements)
            if write_runtime is not None and write_runtime > 0:
                data['write'][backend]['num_elements'].append(num_elements)
                data['write'][backend]['runtime'].append(write_runtime)
                data['write'][backend]['std'].append(write_runtime_std)
                print(f"    {num_elements:.0f} write: {write_runtime:.2f} s (± {write_runtime_std:.2f} s)")
            else:
                print(f"    {num_elements:.0f} write: No data")
    return data

def plot_runtime_comparison(data, backends=['redis', 'filesystem', 'dragon'], location='simulation'):
    """Create plots comparing read and write runtime across backends."""
    backends = [f"{b}_{location}" if b != 'filesystem' else f"{b}_training" for b in backends]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    colors = {
        'redis': 'red',
        'filesystem': 'blue', 
        'nodelocal': 'green',
        'dragon': 'purple'
    }
    markers = {
        'redis': 'o',
        'filesystem': 'o',
        'nodelocal': 'o',
        'dragon': 'o'
    }
    for backend in backends:
        if data['read'][backend]['nodes']:
            ax1.plot(data['read'][backend]['nodes'], data['read'][backend]['runtime'],
                     color=colors[backend.split('_')[0]], linewidth=2, label=backend.split('_')[0].capitalize(), marker=markers[backend.split('_')[0]])
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Number of Nodes')
    ax1.set_ylabel('Excution time/Iteration (s)')
    ax1.set_title('Training (db={})'.format(location))
    ax1.legend()
    ax1.grid(True)
    for backend in backends:
        try:
            if data['write'][backend]['nodes']:
                ax2.plot(data['write'][backend]['nodes'], data['write'][backend]['runtime'],
                         color=colors[backend.split('_')[0]], linewidth=2, label=backend.split('_')[0].capitalize(), marker=markers[backend.split('_')[0]])
        except Exception as e:
            print(f"Error plotting {backend} write runtime: {e}")
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Number of Nodes')
    ax2.set_ylabel('Runtime/Iteration (s)')
    ax2.set_title('Simulation (db={})'.format(location))
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig(f'figs/runtime_comparison_{location}.png', dpi=300, bbox_inches='tight')
    plt.show()

# New: plot runtime vs num_elements for a given node count
def plot_vs_elements_runtime(data, backends=['redis', 'filesystem', 'dragon'], location='simulation', node_count=8):
    """Plot runtime vs num_elements for each backend at a given node count."""
    backends = [f"{b}_{location}" if b != 'filesystem' else f"{b}_training" for b in backends]
    colors = {
        'redis': 'red',
        'filesystem': 'blue',
        'nodelocal': 'green',
        'dragon': 'purple'
    }
    markers = {
        'redis': 'o',
        'filesystem': 's',
        'nodelocal': '^',
        'dragon': 'D'
    }
    plt.figure(figsize=(1.75,1.75))
    for backend in backends:
        if data['read'][backend]['num_elements']:
            x = [n*4/1e6 for n in data['read'][backend]['num_elements']]  # Data size per process (MB)
            y = data['read'][backend]['runtime']
            plt.plot(x, y, label=backend.split('_')[0].capitalize(),
                     color=colors.get(backend.split('_')[0], None), marker='o',markersize=1.5)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Data Size per Process (MB)')
    plt.ylabel('Execution Time/Iteration (s)')
    # plt.title(f'Training')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    outname = f'figs/runtime_vs_elements_read_{node_count}nodes.png'
    plt.savefig(outname, dpi=300, bbox_inches='tight')
    plt.savefig(outname.replace('png','pdf'), dpi=300, bbox_inches='tight')
    print(f"Plot saved as {outname}")
    plt.show()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compare IO runtime across different backends.")
    parser.add_argument('--backends', nargs='+', default=['redis', 'filesystem', 'dragon'], 
                        help='List of backends to compare (default: redis filesystem dragon)')
    parser.add_argument('--location', type=str, default='simulation', help='Database location (simulation/training) (default: simulation)')
    parser.add_argument('--nodes', type=int, default=None, help='Node count (for vs num_elements plot)')
    parser.add_argument('--num_elements_list', type=int, nargs='+', default=[10000, 100000, 500000, 1000000, 2000000, 4000000, 8000000, 16000000, 32000000, 64000000], help='List of num_elements to sweep (for vs num_elements plot)')
    args = parser.parse_args()
    backends = args.backends
    location = args.location

    # If --num_elements_list and --nodes are provided, do vs num_elements plot
    if args.num_elements_list and args.nodes:
        print(f"Plotting runtime vs num_elements for node_count={args.nodes}")
        data = collect_vs_elements_runtime(args.nodes, args.num_elements_list, backends=backends, location=location)
        # Filter out backends with no data
        filtered_data = {
            'read': {backend: values for backend, values in data['read'].items() if values['num_elements']},
            'write': {backend: values for backend, values in data['write'].items() if values['num_elements']}
        }
        if not filtered_data['read'] and not filtered_data['write']:
            print("No runtime data found!")
            return
        print(f"\nFound read data for backends: {list(filtered_data['read'].keys())}")
        print(f"Found write data for backends: {list(filtered_data['write'].keys())}")
        print("\nCreating runtime vs num_elements plot...")
        plot_vs_elements_runtime(filtered_data, backends=backends, location=location, node_count=args.nodes)
        return

    # Otherwise, do the default runtime vs node count plot
    print(f"Using {len(backends)} backends: {', '.join(backends)}")
    print(f"Collecting runtime data...")
    data = collect_runtime_data(backends=backends, location=location, nodes=[8, 32, 128, 512])
    # Filter out backends with no data
    filtered_data = {
        'read': {backend: values for backend, values in data['read'].items() if values['nodes']},
        'write': {backend: values for backend, values in data['write'].items() if values['nodes']}
    }
    if not filtered_data['read'] and not filtered_data['write']:
        print("No runtime data found!")
        return
    print(f"\nFound read data for backends: {list(filtered_data['read'].keys())}")
    print(f"Found write data for backends: {list(filtered_data['write'].keys())}")
    print("\nCreating runtime comparison plots...")
    plot_runtime_comparison(filtered_data, backends=backends, location=location)
    print(f"Plot saved as 'figs/runtime_comparison_{location}.png'")

if __name__ == "__main__":
    main()
