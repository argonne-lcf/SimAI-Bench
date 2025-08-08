import re
from datetime import datetime
import os
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
            if pattern1.search(line) or pattern2.search(line):
                num_dt += 1
            m = pattern.search(line)
            if m:
                timestamp = datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S,%f')
                tstep_time = float(m.group(2))
                actual_dt_time = float(m.group(3))
                events.append({
                    'timestamp': timestamp,
                    'tstep_time': tstep_time,
                    'actual_dt_time': actual_dt_time,
                    "num_dt": num_dt
                })
                print(f"Parsed event: {timestamp}, tstep_time: {tstep_time}, actual_dt_time: {actual_dt_time}, num_dt: {num_dt}")
                num_dt = 0
    print(f"Total data transport operations: {len(events)} in {file_path}")
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
    events = parse_log_file_for_io(file_path)
    
    if not events:
        return {'throughput_MB_per_sec': 0, 'total_data_MB': 0, 'total_time_sec': 0}
    
    # Convert dtype to numpy dtype if it's a string
    if isinstance(dtype, str):
        dtype = np.dtype(dtype)
    elif hasattr(dtype, 'dtype'):
        dtype = dtype.dtype
    
    # Calculate data size in bytes
    element_size_per_write = (dtype.itemsize * num_elements)/(1024 * 1024)  # Convert to MB

    count = 0
    mean_throughput_MB_per_sec = 0
    total_data_MB = 0
    throughputs = []
    read_first = False
    for event in events:
        if event["num_dt"] > 0:
            if not read_first:
                read_first = True
                continue
            # Calculate total data transferred in MB
            total_data_MB += element_size_per_write * event['num_dt']
            throughput = (element_size_per_write * event['num_dt'] / event['actual_dt_time'] if event['actual_dt_time'] > 0 else 0)
            mean_throughput_MB_per_sec += throughput
            throughputs.append(throughput)
            count += 1
    if count > 0:
        mean_throughput_MB_per_sec /= count
        std_throughput_MB_per_sec = float(np.std(throughputs))
    else:
        std_throughput_MB_per_sec = 0.0

    # Calculate total time for all data transport operations
    total_dt_time = sum(event['actual_dt_time'] for event in events)
    
    return {
        'mean_throughput_MB_per_sec': mean_throughput_MB_per_sec,
        'std_throughput_MB_per_sec': std_throughput_MB_per_sec,
        'total_data_MB': total_data_MB,
        'total_time_sec': total_dt_time,
        'total_num_dt': sum([e["num_dt"] for e in events]),
        'avg_time_per_dt': total_dt_time / sum([e["num_dt"] for e in events])
    }

def get_runtime(file_path):
    events = parse_log_file(file_path)
    return (events[-1]['timestamp'] - events[0]['timestamp']).total_seconds() if events else 0


def get_runtime_per_iteration(file_path):
    """
    Returns a list of runtime durations (in seconds) for each iteration/event in the log file.
    Iteration is defined by each event parsed by parse_log_file.
    """
    events = parse_log_file(file_path)
    if len(events) < 2:
        return {'mean': 0.0, 'variance': 0.0, 'runtimes': []}
    runtimes = []
    for i in range(2, len(events)):
        dt = (events[i]['timestamp'] - events[i-1]['timestamp']).total_seconds()
        runtimes.append(dt)
    mean = float(np.mean(runtimes)) if runtimes else 0.0
    std = float(np.std(runtimes)) if runtimes else 0.0
    return {'mean': mean, 'std': std, 'runtimes': runtimes}

def get_mean_times(file_path, event_type):
    if event_type == "read":
        pattern = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - .*?Mean read time \(s\): ([\d\.]+), Std: ([\d\.]+)')
    elif event_type == "write":
        pattern = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - .*?Mean data write time \(s\): ([\d\.]+), Std: ([\d\.]+)')
    elif event_type == "iteration":
        pattern = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - .*?Mean iteration time \(s\): ([\d\.]+), Std: ([\d\.]+)')
    else:
        raise ValueError("Invalid event type. Use 'read' or 'write'.")

    if not os.path.exists(file_path):
        return {'mean': 0.0, 'std': 0.0, 'timestamp': None}
    with open(file_path, 'r') as f:
        for line in f:
            m = pattern.search(line)
            # print(f"Processing line: {line.strip()}")
            if m:
                timestamp = datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S,%f')
                mean_read_time = float(m.group(2))
                std_read_time = float(m.group(3))
                return {
                    'timestamp': timestamp,
                    'mean': mean_read_time,
                    'std': std_read_time
                }
    return {'timestamp': None, 'mean': 0.0, 'std': 0.0}


def compute_io_throughput_new(file_path, num_elements, dtype):
    """
    Compute IO throughput based on log file events.
    
    Args:
        file_path: Path to the log file
        num_elements: Number of array elements transferred
        dtype: Data type of the array elements (e.g., np.float32, np.float64, 'float32', 'float64')
    
    Returns:
        Dictionary containing throughput metrics
    """
    pattern_write = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - .*?Mean write throughput \(MB/s\): ([\d\.]+), Std: ([\d\.]+)')
    pattern_read = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - .*?Mean read throughput \(MB/s\): ([\d\.]+), Std: ([\d\.]+)')
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            m_write = pattern_write.search(line)
            if m_write:
                mean_write_throughput = float(m_write.group(2))
                std_write_throughput = float(m_write.group(3))
                return {
                    'mean_throughput_MB_per_sec': mean_write_throughput,
                    'std_throughput_MB_per_sec': std_write_throughput
                }
            m_read = pattern_read.search(line)
            if m_read:
                mean_read_throughput = float(m_read.group(2))
                std_read_throughput = float(m_read.group(3))
                return {
                    'mean_throughput_MB_per_sec': mean_read_throughput,
                    'std_throughput_MB_per_sec': std_read_throughput
                }

    return None