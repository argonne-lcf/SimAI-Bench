from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

from SimAIBench.dag import DAG
from SimAIBench.component import WorkflowComponent
from SimAIBench.executors import BaseExecutor
from SimAIBench.resources import ClusterResource
from SimAIBench.config import OchestratorConfig

from networkx import DiGraph, topological_sort

logger = logging.getLogger(__name__)

try:
    from taps.engine import Engine, as_completed, TaskFuture, task
    from taps.executor.python import ThreadPoolConfig
    from taps.plugins import get_executor_configs
    TAPS_AVAILABLE = True
    logger.info("TAPS successfully imported")
except ImportError:
    TAPS_AVAILABLE = False
    logger.warning("TAPS is not available - TapsExecutor will not function")
    # Create dummy types for when taps is not available


class TapsExecutor(BaseExecutor):
    """
    An interface class to execute my explicit DAG using taps engine.
    """

    def __init__(self, dag: DAG, config: OchestratorConfig):
        if not TAPS_AVAILABLE:
            logger.error("TAPS is not installed - cannot create TapsExecutor")
            raise ImportError('TAPS is not installed. Please install it to use TapsExecutor.')
        
        logger.info("Initializing TapsExecutor")
        super().__init__(dag)
        # Using a simple ThreadPoolExecutor from taps for local execution.
        # This can be made configurable later.
        available_executors: Dict = get_executor_configs()
        try:
            self.executor_config = available_executors[config.name]()
        except Exception as e:
            logger.error(f"Executor config '{config.name}' not found in available executors: {list(available_executors.keys())}")
            raise
        self.engine = Engine(self.executor_config.get_executor())
        self.futures: Dict[str, TaskFuture] = {}
        logger.info("TapsExecutor initialized successfully")
    
    def _transform_dag(self):
        """Transforms the node['callable'] into a taps task"""
        logger.info("Transforming DAG nodes into TAPS tasks")
        graph: DiGraph = self._dag.graph
        node_count = 0
        for node, c in graph.nodes(data="callable"):
            logger.debug(f"Transforming node: {node}")
            graph.nodes[node]["task"] = task(c)
            node_count += 1
        logger.info(f"Successfully transformed {node_count} nodes into TAPS tasks")

    def run(self, cluster_resource: ClusterResource) -> None:
        """
        Executes the DAG using the TAPS engine.

        Args:
            ClusterResource object
        """
        logger.info('Starting DAG execution with TAPS executor.')
        self._transform_dag()
        
        graph: DiGraph = self._dag.graph
        node_execution_order = list(topological_sort(graph))
        logger.info(f"Executing {len(node_execution_order)} nodes in topological order")
        logger.debug(f"Execution order: {node_execution_order}")
        
        for i, node in enumerate(node_execution_order):
            logger.debug(f"Executing node {i+1}/{len(node_execution_order)}: {node}")
            node_obj = graph.nodes[node]
            args = [cluster_resource]
            
            # Iterate through dependencies and collect their futures
            predecessors = list(graph.predecessors(node))
            if predecessors:
                logger.debug(f"Node {node} has {len(predecessors)} dependencies: {predecessors}")
                for predecessor in predecessors:
                    args.append(graph.nodes[predecessor]["future"])
            else:
                logger.debug(f"Node {node} has no dependencies")
            
            logger.debug(f"Submitting task for node: {node}")
            node_obj["future"] = self.engine.submit(node_obj["task"],*args)

        logger.info("DAG execution completed successfully")
        return node_obj["future"]
    
    def __exit__(self, exc_type, exc_value, traceback):
        """Exit context manager - cleanup resources"""
        logger.info("Exiting TapsExecutor context")
        try:
            if hasattr(self, 'engine'):
                self.engine.shutdown()
                logger.info("TAPS engine shutdown successfully")
        except Exception as e:
            logger.error(f"Error shutting down TAPS engine: {e}")
        
        if exc_type:
            logger.error(f"Exception in TapsExecutor: {exc_value}")
        
        return False  # Don't suppress exceptions