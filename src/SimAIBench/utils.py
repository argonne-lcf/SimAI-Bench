import os
import math
import json
import time
import subprocess
import copy
import logging
from glob import glob
from typing import Dict, List, Any, Union, Callable
import logging
import os

logger = logging.getLogger(__name__)
try:
    import dragon
    DRAGON_AVAILABLE = True
except:
    DRAGON_AVAILABLE = False

def get_nodes():
    import socket
    nodefile = os.environ.get("PBS_NODEFILE","/dev/null")
    with open(nodefile,"r") as f:
        nodes = [l.split(".")[0] for l in f.readlines()]
    if len(nodes) == 0:
        nodes = [socket.gethostname()]
    logger.debug(f"Found nodes {nodes}")
    return nodes


def create_logger(logger_name:str):
    log_level_str = os.environ.get("SIMAIBENCH_LOGLEVEL","INFO")
    if log_level_str == "INFO":
        log_level = logging.INFO
    elif log_level_str == "DEBUG":
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    if not logger.handlers:
        log_dir = os.path.join(os.getcwd(), "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{logger_name}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger