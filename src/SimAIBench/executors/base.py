from abc import ABC, abstractmethod
from typing import Any,Tuple,Dict
from SimAIBench.dag import DAG, NodeStatus
from networkx import DiGraph, is_directed_acyclic_graph, topological_sort
import logging
import os


logger = logging.getLogger(__name__)


class BaseExecutor(ABC):

    def __init__(self):
        self.logger = None
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
    def submit(self, *args, **kwargs):
        """submits a callable for execution"""
        pass

    def submit_dag(self, cluster_resource: Any, dag: DAG) -> Tuple[DAG, Dict]:
        futures = {}
        graph: DiGraph = dag.graph
        node_execution_order = list(topological_sort(graph))
        for i, node in enumerate(node_execution_order):
            node_obj = graph.nodes[node]
            if node_obj['status'] == NodeStatus.NOT_SUBMITTED:
                try:
                    self.logger.debug(f"Submitting {node} for execution")
                    args = [cluster_resource]
                    # Iterate through dependencies and collect their futures
                    predecessors = list(graph.predecessors(node))
                    if predecessors:
                        self.logger.debug(f"Node {node} has {len(predecessors)} dependencies: {predecessors}")
                        for predecessor in predecessors:
                            args.append(futures[predecessor])
                    else:
                        self.logger.debug(f"Node {node} has no dependencies")
                    
                    futures[node] = self.submit(node_obj["callable"],args)
                    
                    node_obj["status"] = next(node_obj["status"])
                except Exception as e:
                    self.logger.error(f"Submitting {node} failed with exception {e}")
                    node_obj["status"] = NodeStatus.FAILED   
        return dag, futures
    
    @abstractmethod
    def cleanup(self,*args,**kwargs):
        """Cleans up the resources"""
        pass