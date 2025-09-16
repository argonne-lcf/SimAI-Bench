from abc import ABC, abstractmethod
from SimAIBench.dag import DAG
from networkx import DiGraph, is_directed_acyclic_graph
import logging

logger = logging.getLogger(__name__)


class BaseExecutor(ABC):
    def __init__(self, dag: DAG):
        self._dag = dag
        if not is_directed_acyclic_graph(dag.graph):
            logger.error("Graph is not acyclic!")
            raise ValueError("Graph is no acyclic")

    def __enter__(self):
        return self
    
    @abstractmethod
    def _transform_dag(self, *args, **kwargs):
        """Method that takes in the SimAI-Bench dag representation and converts it to specific framework"""
        pass

    @abstractmethod
    def run(self, *args, **kwargs):
        """runs the dag"""
        pass