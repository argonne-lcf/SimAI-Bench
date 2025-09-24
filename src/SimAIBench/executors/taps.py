from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Tuple, List, Sequence

from SimAIBench.dag import DAG
from SimAIBench.component import WorkflowComponent
from SimAIBench.executors import BaseExecutor
from SimAIBench.resources import ClusterResource
from SimAIBench.config import OchestratorConfig
from concurrent.futures import Future

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
    A stateless interface class to execute my explicit DAG using taps engine.
    """
    def __init__(self, config: OchestratorConfig):
        if not TAPS_AVAILABLE:
            logger.error("TAPS is not installed - cannot create TapsExecutor")
            raise ImportError('TAPS is not installed. Please install it to use TapsExecutor.')
        
        logger.info("Initializing TapsExecutor")
        available_executors: Dict = get_executor_configs()
        try:
            self.executor_config = available_executors[config.name]()
        except Exception as e:
            logger.error(f"Executor config '{config.name}' not found in available executors: {list(available_executors.keys())}")
            raise
        self.engine = Engine(self.executor_config.get_executor())
        self.futures: Dict[str, TaskFuture] = {}
        logger.info("TapsExecutor initialized successfully")

    def submit(self, f: Any, args: Sequence) -> Future:
        """
        Submits a task using the TAPS engine.

        Args:
            task
            args: arguments
        """
        return self.engine.submit(task(f),*args)
    
    def __exit__(self, exc_type, exc_value, traceback):
        """Exit context manager - cleanup resources"""
        logger.info("Exiting TapsExecutor context")
        
        self.cleanup()
        
        if exc_type:
            logger.error(f"Exception in TapsExecutor: {exc_value}")
        
        return False  # Don't suppress exceptions
    
    def cleanup(self):
        try:
            if hasattr(self, 'engine'):
                self.engine.shutdown()
                logger.info("TAPS engine shutdown successfully")
        except Exception as e:
            logger.error(f"Error shutting down TAPS engine: {e}")