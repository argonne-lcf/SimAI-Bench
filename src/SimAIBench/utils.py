import os
import math
import json
import time
import subprocess
import copy
import logging
from glob import glob
from typing import Dict, List, Any, Union, Callable

try:
    import dragon
    DRAGON_AVAILABLE = True
except:
    DRAGON_AVAILABLE = False

def get_nodes():
    import socket
    nodefile = os.environ.get("PBS_NODEFILE","/dev/null")
    with open(nodefile,"r") as f:
        nodes = [l.split(".") for l in f.readlines()]
    return nodes if nodes else socket.gethostname()