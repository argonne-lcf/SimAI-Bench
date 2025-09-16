import subprocess
import time
import multiprocessing
import logging

from typing import List, Union, Dict, Any
from SimAIBench.component import WorkflowComponent
from SimAIBench.resources import *
from SimAIBench.dag import DAG
from SimAIBench.executors import TapsExecutor
from SimAIBench.utils import get_nodes
from concurrent.futures import Future

from SimAIBench.config import SystemConfig, OchestratorConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('orchestrator.log', mode='w')
    ]
)

class Orchestrator:
    """
    """
    def __init__(self, 
                 workflow_components: Dict[str, WorkflowComponent], 
                 sys_info: SystemConfig = SystemConfig(name="local"),
                 config: OchestratorConfig = OchestratorConfig()):
        """
        Initialize the BasicLauncher.
        
        Args:
            system: System name (e.g., "local", "aurora", "polaris")
            launcher_config: Configuration for the launcher
        """
        self.logger = logging.getLogger(__name__)
        self.sys_info = sys_info
        self.dag = DAG(workflow_components)
        self.executor = None
        self.sys_info = NodeResourceList.from_config(sys_info)
        self.cluster_resource = LocalClusterResource(nodes=get_nodes(),system_info=self.sys_info)
        self.logger.info("Orchestrator initialized with system: %s", sys_info.__repr__())
        self.config = config
    
    def launch(self):
        """launch the workflow"""
        with TapsExecutor(self.dag,config=self.config) as executor:
            self.executor = executor
            self.logger.info("Starting workflow execution")
            final_future: Future = self.executor.run(self.cluster_resource)

            try:
                result = final_future.result()
                self.logger.info("Workflow execution completed")
                self.logger.info("Final result: %s", result)
            except Exception as e:
                self.logger.error(f"Workflow execution failed with Exceptio {e}")
                raise e
        return 1

    def update(self,workflow_component: WorkflowComponent):
        """Update the dag"""
        self.logger.info("Updating DAG with component: %s", workflow_component)
        self.dag.update(workflow_component)
        self.logger.info("DAG updated successfully")
    
    