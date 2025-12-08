from __future__ import annotations

import logging, os
from pathlib import Path
from typing import Any, Dict, Tuple, List, Sequence

from SimAIBench.dag import DAG, NodeStatus,DagFuture
from SimAIBench.component import WorkflowComponent
from SimAIBench.executors import BaseExecutor
from SimAIBench.resources import ClusterResource
from SimAIBench.config import OchestratorConfig, SystemConfig
from SimAIBench.resources import NodeResourceList
from concurrent.futures import Future

from networkx import DiGraph, topological_sort
from SimAIBench.profiling import CallableProfiler

try:
    import taps
    from taps.engine import Engine, as_completed, TaskFuture, task
    from taps.executor.python import ThreadPoolConfig
    from taps.plugins import get_executor_configs
    TAPS_AVAILABLE = True
    print("TAPS successfully imported")
    from taps.executor.parsl import HTExConfig, ParslHTExConfig
except ImportError:
    TAPS_AVAILABLE = False
    print("TAPS is not available - TapsExecutor will not function")
    # Create dummy types for when taps is not available


class TapsExecutor(BaseExecutor):
    """
    A interface class to execute my explicit DAG using taps engine.
    """
    def __init__(self, config: OchestratorConfig, sys_info: NodeResourceList):
        super().__init__(config, sys_info)
        if not TAPS_AVAILABLE:
            self.logger.error("TAPS is not installed - cannot create TapsExecutor")
            raise ImportError('TAPS is not installed. Please install it to use TapsExecutor.')
        
        self.logger.info("Initializing TapsExecutor")
        if self.config.name == "ray":
            os.environ["TMPDIR"]="/tmp"
        
        if self.config.name == "dask":
            import dask.config
            self.logger.warning("Resetting some dask config")
            dask.config.set({"distributed.comm.tls.max-version":None})
            dask.config.set({"distributed.scheduler.idle-timeout":None})
            dask.config.set({"distributed.scheduler.no-workers-timeout":None})
            dask.config.set({"distributed.worker.lifetime.duration":None})

        if self.config.name == "parsl-htex":
            self.executor_config = ParslHTExConfig(htex=HTExConfig())
        else:
            available_executors: Dict = get_executor_configs()
            try:
                self.executor_config = available_executors[config.name]()
            except Exception as e:
                self.logger.error(f"Executor config '{config.name}' not found in available executors: {list(available_executors.keys())}")
                raise
        self.engine = Engine(self.executor_config.get_executor())
        self.futures: Dict[str, TaskFuture] = {}
        self.logger.info("TapsExecutor initialized successfully")

    def submit_dag(self, cluster_resource: Any, dag: DAG, old_futures: Dict) -> Tuple[DAG, Dict]:
        """
        Submit a DAG for execution
        
        :param self: Description
        :param cluster_resource: Description
        :type cluster_resource: Any
        :param dag: Description
        :type dag: DAG
        :return: Description
        :rtype: Tuple[DAG, Dict]
        """
        futures = {}
        graph: DiGraph = dag.graph
        node_execution_order = list(topological_sort(graph))
        count = 0
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
                            try:
                                args.append(old_futures[predecessor])
                            except KeyError:
                                args.append(futures[predecessor])
                    else:
                        self.logger.debug(f"Node {node} has no dependencies")
                    
                    futures[node] = self.engine.submit(task(node_obj["callable"]), *args)
                    
                    node_obj["status"] = next(node_obj["status"])
                except Exception as e:
                    self.logger.error(f"Submitting {node} failed with exception {e}")
                    node_obj["status"] = NodeStatus.FAILED   
        return dag, futures

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit context manager - cleanup resources"""
        self.logger.info("Exiting TapsExecutor context")
        
        self.cleanup()
        
        if exc_type:
            self.logger.error(f"Exception in TapsExecutor: {exc_value}")
        
        return False  # Don't suppress exceptions
    
    def cleanup(self):
        try:
            if hasattr(self, 'engine'):
                self.engine.shutdown()
                self.logger.info("TAPS engine shutdown successfully")
        except Exception as e:
            self.logger.error(f"Error shutting down TAPS engine: {e}")