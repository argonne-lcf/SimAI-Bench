from .timer import Timer
from typing import List
import statistics

def compute_mean_std(timers: List[Timer]):
    times = [timer.get_duration() for timer in timers]
    mean = statistics.mean(times)
    std_dev = statistics.stdev(times)
    return mean, std_dev