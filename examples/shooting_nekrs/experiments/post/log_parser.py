import re
from datetime import datetime
import numpy as np

def parse_timestamp(line):
    ts = line.split(" - ")[0]
    return datetime.strptime(ts, '%Y-%m-%d %H:%M:%S,%f')

def parse_log_file(file_path):
    """Parse log file for tstep and data transport events."""
    events = []
    # Handle both "dt time: " and "dt time:" formats
    pattern = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - .*?tstep time: ([\d\.]+), dt time:\s*([\d\.]+)')
    with open(file_path, 'r') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                timestamp = datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S,%f')
                tstep_time = float(m.group(2))
                actual_dt_time = float(m.group(3))
                events.append({
                    'timestamp': timestamp,
                    'tstep_time': tstep_time,
                    'actual_dt_time': actual_dt_time
                })
    return events

def parse_log_file_for_io(file_path):
    pattern1 = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - .*?Write the data')
    pattern2 = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - .*?Read data')
    pattern = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - .*?tstep time: ([\d\.]+), dt time:\s*([\d\.]+)')
    events = []
    with open(file_path, 'r') as f:
        num_dt = 0
        for line in f:
            if pattern1.match(line):
                num_dt += 1
            elif  pattern2.match(line):
                num_dt += 2
            m = pattern.search(line)
            if m:
                timestamp = datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S,%f')
                tstep_time = float(m.group(2))
                actual_dt_time = float(m.group(3))
                # if actual_dt_time > 100.0:
                #     continue
                events.append({
                    'timestamp': timestamp,
                    'tstep_time': tstep_time,
                    'actual_dt_time': actual_dt_time,
                    "num_dt": num_dt
                })
                num_dt = 0
    return events

def compute_io_throughput(file_path, num_elements, dtype):
    """
    Compute IO throughput based on log file events.
    
    Args:
        file_path: Path to the log file
        num_elements: Number of array elements transferred
        dtype: Data type of the array elements (e.g., np.float32, np.float64, 'float32', 'float64')
    
    Returns:
        Dictionary containing throughput metrics
    """
    
    pattern_read = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - .*?Mean read throughput \(MB/s\): ([\d\.]+), Std: ([\d\.]+),?')
    # "Mean read throughput (MB/s): 570.5310197853083, Std: 45.585023720688724,"
    pattern_write = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - .*?Mean write throughput \(MB/s\): ([\d\.]+), Std: ([\d\.]+),?')
    
    with open(file_path,"r") as f:
        lines= f.readlines()
        for line in lines:
            # print(line)
            m_read = pattern_read.search(line)
            m_write = pattern_write.search(line)
            if m_read or m_write:
                # print(line)
                if m_read:
                    m = m_read
                else:
                    m = m_write
                timestamp = datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S,%f')
                mean = float(m.group(2))
                std = float(m.group(3))

                return {
                    'mean_throughput_MB_per_sec': mean,
                    'std_throughput_MB_per_sec': std,
                }

    return {'mean_throughput_MB_per_sec': None,
            'std_throughput_MB_per_sec': None,}

def get_runtime(file_path):
    events = parse_log_file(file_path)
    return (events[-1]['timestamp'] - events[0]['timestamp']).total_seconds() if events else 0


def get_start_time(file_path):
    if "train_ai" in file_path:
        with open(file_path, 'r') as f:
            for line in f:
                if "First simulation data file detected, starting training loop" in line:
                    return parse_timestamp(line)
    else:
        events = parse_log_file(file_path)
        return events[0]['timestamp'] if events else None


def get_mean_times(file_path, mode):
    
    if mode == "iteration":
        pattern = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - .*?Mean iteration time \(s\): ([\d\.]+), Std: ([\d\.]+),?')
    elif mode == "read":
        # "Mean read throughput (MB/s): 570.5310197853083, Std: 45.585023720688724,"
        pattern = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - .*?Mean read time \(s\): ([\d\.]+), Std: ([\d\.]+),?')
    elif mode == "write":
        pattern = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - .*?Mean data write time \(s\): ([\d\.]+), Std: ([\d\.]+),?')
    else:
        raise 
    
    with open(file_path,"r") as f:
        lines= f.readlines()
        for line in lines:
            # print(line)
            m = pattern.search(line)
            if m:
                timestamp = datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S,%f')
                mean = float(m.group(2))
                std = float(m.group(3))

                return {
                    'timestamp': timestamp,
                    'mean': mean,
                    'std': std
                }

    return {'mean': None,'std': None,}

def print_number_of_events(filepath):
    events = parse_log_file_for_io(filepath)
    num_tsteps = len(events)
    total_num_dt = sum([e["num_dt"] for e in events])
    print(f"Metadata for file {filepath}")
    print(f"Number of timesteps:{num_tsteps}")
    print(f"Number of data transports: {total_num_dt}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Parse log file and print number of events.")
    parser.add_argument("filepath", type=str, help="Path to the log file")
    args = parser.parse_args()
    print_number_of_events(args.filepath)
