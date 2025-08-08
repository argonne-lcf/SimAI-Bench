import re
import statistics
import os
import matplotlib.pyplot as plt
import scienceplots
from glob import glob
from log_parser import *

plt.style.use(["science", "no-latex"])
plt.rcParams.update({
    'font.size': 8,
    'axes.titlesize': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.titlesize': 8
})

def extract_dt_times(log_file_path, actual=False):
    if actual:
        dt_pattern = re.compile(r'actual dt time: (\d+\.\d+)')    
    else:
        dt_pattern = re.compile(r'dt time: (\d+\.\d+)')
    dt_times = []

    if not os.path.exists(log_file_path):
        return dt_times
    
    with open(log_file_path, 'r') as f:
        for line in f:
            match = dt_pattern.search(line)
            if match:
                dt_time = float(match.group(1))
                if dt_time > 0:
                    dt_times.append(dt_time)
    return dt_times

def compute_throughput_stats(log_files: list, nranks: int, data_size_per_rank: int, actual=False):
    throughputs = []
    for log_file in log_files:
        dt_times = extract_dt_times(log_file, actual=actual)
        if not dt_times:
            continue
        throughputs += [(nranks * data_size_per_rank * 4 / dt / 1024 / 1024 / 1024) for dt in dt_times]  # GB/s
    mean_t = statistics.mean(throughputs)
    std_t = statistics.stdev(throughputs) if len(throughputs) > 1 else 0
    return mean_t, std_t

# Data collection
results = {
    backend: {
        "data_sizes": [], "read_t": [], "read_err": [],
        "write_t": [], "write_err": []
    } for backend in ["dragon", "filesystem", "redis"]#,"dragon_increase_pool"]
}

for backend in ["dragon","filesystem","redis"]:#,"dragon_increase_pool"]:
    for count in [1e4, 1e5, 1e6, 2e6, 4e6, 8e6, 16e6, 32e6, 64e6]:
        data_size = "%.2fm"%(count/1e6)
        for ndb in [1]:
            dirname = f"../2nodes/logs_{backend}_{'simulation' if backend != 'filesystem' else 'training'}_{data_size}_{ndb}db"
            AI_logfile = dirname+"_exp0/train_ai_0_datastore.log"
            sim_logfile = dirname+"_exp0/sim_0_0_datastore.log"

            data = compute_io_throughput_new(AI_logfile, 1e6, np.float32)
            read_t = data["mean_throughput_MB_per_sec"]/1024
            read_err = data["std_throughput_MB_per_sec"]/1024
            data = compute_io_throughput_new(sim_logfile, 1e6, np.float32)
            write_t = data["mean_throughput_MB_per_sec"]/1024
            write_err = data["std_throughput_MB_per_sec"]/1024

            if read_t is None or write_t is None:
                continue  # skip if log files are missing or malformed

            data_MB = count * 4 / 1000 / 1000
            results[backend]["data_sizes"].append(data_MB)
            results[backend]["read_t"].append(read_t)
            results[backend]["read_err"].append(read_err)
            results[backend]["write_t"].append(write_t)
            results[backend]["write_err"].append(write_err)


colors = {
    "redis": "tab:blue",
    "filesystem": "tab:orange",
    "dragon": "tab:green",
}
for label in ["read_t", "write_t"]:
    # Plotting
    fig, ax = plt.subplots(1, 1, figsize=(1.75, 1.75), sharey=False)

    for backend in ['redis', 'filesystem', 'dragon']:
        x = results[backend]["data_sizes"]
        ax.errorbar(x, results[backend][label], yerr = results[backend][label.replace("_t","_err")], marker='o', markersize=1.5, label=backend.capitalize(), color=colors[backend], capsize=1.5)
        # ax[0].errorbar(x, results[backend]["read_t"], yerr=results[backend]["read_err"], marker='o', label=backend, capsize=5)
        # ax[1].errorbar(x, results[backend]["write_t"], yerr=results[backend]["write_err"], marker='s', label=backend, capsize=5)

    ax.set_xscale('log')

    # ax.set_title(f"{label.replace('_t', '').title()}")
    ax.set_xlabel("Data Size (MB)")
    ax.set_ylabel("Throughput/Process (GB/s)")
    ax.legend(loc="upper left",handlelength=1,handleheight=0.5)
    ax.set_ylim([0.0,2.5])
    ax.grid(True)


    plt.tight_layout()
    fig.savefig(f"figs/throughput_{label}.png", dpi=300, bbox_inches='tight')
    fig.savefig(f"figs/throughput_{label}.pdf", dpi=300, bbox_inches='tight')
    # plt.show()

# # Plotting
# fig, ax = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

# for backend in results:
#     x = results[backend]["data_sizes"]
#     ax[0].errorbar(x, results[backend]["read_t"], yerr=results[backend]["read_err"], marker='o', label=backend, capsize=5)
#     ax[1].errorbar(x, results[backend]["write_t"], yerr=results[backend]["write_err"], marker='s', label=backend, capsize=5)

# for a in ax:
#     a.set_xscale('log')

# ax[0].set_title("AI Read Throughput")
# ax[0].set_xlabel("Data Size (MB)")
# ax[0].set_ylabel("Throughput (GB/s)")
# ax[0].legend()
# ax[0].grid(True)

# ax[1].set_title("Simulation Write Throughput")
# ax[1].set_xlabel("Data Size (MB)")
# ax[1].legend()
# ax[1].grid(True)

# plt.tight_layout()
# plt.savefig("throughput.png", dpi=300)
# plt.show()
