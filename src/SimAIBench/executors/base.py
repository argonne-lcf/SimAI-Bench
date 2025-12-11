from abc import ABC, abstractmethod
from typing import Any,Tuple,Dict
from SimAIBench.dag import DAG, NodeStatus
from SimAIBench.config import OchestratorConfig, SystemConfig
from SimAIBench.resources import NodeResourceList
from networkx import DiGraph, is_directed_acyclic_graph, topological_sort
import logging
import os
from SimAIBench.utils import create_logger


logger = logging.getLogger(__name__)


class BaseExecutor(ABC):

    def __init__(self, config: OchestratorConfig, sys_info: NodeResourceList):
        self.logger = None
        self.config = config
        self.sys_info = sys_info
        self.logger = create_logger("Executor", subdir="executor")

    def __enter__(self):
        return self

    @abstractmethod
    def submit_dag(self,*args,**kwargs):
        pass
    
    @abstractmethod
    def cleanup(self,*args,**kwargs):
        """Cleans up the resources"""
        pass