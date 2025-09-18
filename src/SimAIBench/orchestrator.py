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
    encoding='utf-8',
    errors='replace',
    # handlers=[
    #     logging.StreamHandler(),
    #     logging.FileHandler('orchestrator.log',  mode='w', encoding='utf-8')
    # ]
)

class Orchestrator:
    def __init__(self, 
                 workflow_components: Dict[str, WorkflowComponent], 
                 sys_info: SystemConfig = SystemConfig(name="local"),
                 config: OchestratorConfig = OchestratorConfig(name="parsl-local")):
        self.logger = logging.getLogger(__name__)
        self.sys_info = sys_info
        self.dag = DAG(workflow_components)
        self.executor = None
        self.sys_info = NodeResourceList.from_config(sys_info)
        self.cluster_resource = DistributedClusterResource(nodes=get_nodes(),system_info=self.sys_info)
        self.logger.info("Orchestrator initialized with system: %s", sys_info.__repr__())
        self.config = config
    
    def launch(self):
        """launch the workflow"""
        with TapsExecutor(self.dag,config=self.config) as executor:
            self.executor = executor
            self.logger.info("Starting workflow execution")
            success = self.executor.run(self.cluster_resource)
            try:
                while not all([self.dag.graph.nodes[node]["future"].done() for node in self.dag.graph.nodes()]):
                    time.sleep(5)
                self.logger.info("Workflow execution completed")
                # Count successful and failed tasks
                successful_tasks = 0
                failed_tasks = 0
                
                for node in self.dag.graph.nodes():
                    future = self.dag.graph.nodes[node]["future"]
                    if future.done():
                        try:
                            future.result()  # This will raise an exception if the task failed
                            successful_tasks += 1
                        except Exception:
                            failed_tasks += 1
                
                self.logger.info("Workflow execution completed - Successful tasks: %d, Failed tasks: %d", 
                               successful_tasks, failed_tasks)
                self.cluster_resource.cleanup()
            except Exception as e:
                self.logger.error(f"Workflow execution failed with Exceptio {e}")
                self.cluster_resource.cleanup()
                raise e
        return 1

    def update(self,workflow_component: WorkflowComponent):
        """Update the dag"""
        self.logger.info("Updating DAG with component: %s", workflow_component)
        self.dag.update(workflow_component)
        self.logger.info("DAG updated successfully")
    
    