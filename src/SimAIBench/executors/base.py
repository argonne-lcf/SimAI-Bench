from abc import ABC, abstractmethod
from typing import Any,Tuple,Dict
from SimAIBench.dag import DAG, NodeStatus
from SimAIBench.config import OchestratorConfig, SystemConfig
from SimAIBench.resources import NodeResourceList
from networkx import DiGraph, is_directed_acyclic_graph, topological_sort
import logging
import os


logger = logging.getLogger(__name__)


class BaseExecutor(ABC):

    def __init__(self, config: OchestratorConfig, sys_info: NodeResourceList):
        self.logger = None
        self.config = config
        self.sys_info = sys_info
        self._init_logger()
    
    def _init_logger(self):
        log_level_str = os.environ.get("SIMAIBENCH_LOGLEVEL","INFO")
        if log_level_str == "INFO":
            log_level = logging.INFO
        elif log_level_str == "DEBUG":
            log_level = logging.DEBUG
        else:
            log_level = logging.INFO

        self.logger = logging.getLogger("Executor")
        self.logger.setLevel(log_level)
        if not self.logger.handlers:
            log_dir = os.path.join(os.getcwd(), "logs")
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, f"executor.log")
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def __enter__(self):
        return self

    @abstractmethod
    def submit_dag(self,*args,**kwargs):
        pass
    
    @abstractmethod
    def cleanup(self,*args,**kwargs):
        """Cleans up the resources"""
        pass