"""
This module reads the unitrace profile data from a CFD simulation. 
It is assumed that each iteration is capture using user-defined events named as "step<id>".
The module extracts the gpu and cpu events.

Note: The order of the kernel calls is not guaranteed to be the same as the order of the events in the unitrace file.
"""

import os
import json
from typing import Callable

def read_unitrace_file(filename):
    """
    Reads the unitrace file and returns the data as a dictionary.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File {filename} does not exist.")
    
    with open(filename, 'r') as f:
        data = json.load(f)
    
    return data

def extract_steps_events(data, step_prefix="step"):
    """
    Extracts the steps and events from the unitrace data.
    """
    steps = []
    for event in data['traceEvents']:
        if event['name'].startswith(step_prefix):
            steps.append(event)
    return sorted(steps, key=lambda x: x['ts'])

def extract_events(data:dict,steps:list[dict],cat:str):
    for step in steps:
        step[f"{cat}_events"] = []

    for event in data["traceEvents"]:
        if "cat" in event and event["cat"] == cat:
            for step in steps:
                if event["ts"] >= step["ts"] and event["ts"] <= step["ts"] + step["dur"]:
                    step[f"{cat}_events"].append(event)
                    break
    for step in steps:
        step[f"{cat}_events"] = sorted(step[f"{cat}_events"],key= lambda x: x["ts"])
    
    return steps

def extract_gpu_op_events(data:dict,steps:list[dict]):
    return extract_events(data,steps,"gpu_op")

def extract_cpu_op_events(data:dict,steps:list[dict]):
    return extract_events(data,steps,"cpu_op")

def get_mem_copy_events(events:list[dict]):
    cp_cvents = []
    for event in events:
        if "copy" in event["name"].lower():
            cp_cvents.append(event)
    return cp_cvents


if __name__ == "__main__":
    fname = "/Users/hari/Downloads/nekrs.114366.json"
    data = read_unitrace_file(fname)
    steps = extract_steps_events(data,step_prefix="NekRS::step")
    steps = extract_gpu_op_events(data, steps)
    steps = extract_cpu_op_events(data, steps)

    for i, step in enumerate(steps):
        print(f"Step {i+1}: {step['name']}")
        print(f"  Duration: {step['dur'] / 1000:.2f} ms")
        print(f"  Start time: {step['ts'] / 1000:.2f} ms")
        
        # print(f"  GPU Events: {len(step.get('gpu_op_events', []))}")
        # for j, event in enumerate(step.get('gpu_op_events', [])[:5]):  # Show first 5 GPU events
        #     print(f"    {j+1}. {event['name']} - Duration: {event.get('dur', 0) / 1000:.2f} ms")
        # if len(step.get('gpu_op_events', [])) > 5:
        #     print(f"    ... and {len(step.get('gpu_op_events', [])) - 5} more GPU events")
        
        # print(f"  CPU Events: {len(step.get('cpu_op_events', []))}")
        # for j, event in enumerate(step.get('cpu_op_events', [])[:5]):  # Show first 5 CPU events
        #     print(f"    {j+1}. {event['name']} - Duration: {event.get('dur', 0) / 1000:.2f} ms")
        # if len(step.get('cpu_op_events', [])) > 5:
        #     print(f"    ... and {len(step.get('cpu_op_events', [])) - 5} more CPU events")
        
        mem_copy_events = get_mem_copy_events(step.get('gpu_op_events', []))
        print(f"  Memory Copy Events: {len(mem_copy_events)}")
        mem_copy_events = sorted(mem_copy_events,key=lambda x:x["dur"],reverse=True)
        for j, event in enumerate(mem_copy_events[:5]):
            print(f"    {j+1}. {event['name']} - Duration: {event.get('dur', 0) / 1000:.2f} ms")
        if len(mem_copy_events) > 5:
            print(f"    ... and {len(mem_copy_events) - 5} more Memory Copy events")
        print()  # Add a blank line between steps