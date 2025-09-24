import subprocess
import time
import multiprocessing as mp
from multiprocessing import Queue
import logging
import threading
import queue

from typing import List, Union, Dict, Any
from SimAIBench.component import WorkflowComponent
from SimAIBench.resources import *
from SimAIBench.dag import DAG, DagStore, NodeStatus, DagFuture
from SimAIBench.executors import TapsExecutor
from SimAIBench.utils import get_nodes
from concurrent.futures import Future
from SimAIBench.config import server_registry
from networkx import topological_sort, DiGraph
from pydantic import BaseModel
from typing import Literal
from .client import OrchetratorClient
from SimAIBench.profiling import DagStoreProfiler, CallableProfiler

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

class Message(BaseModel):
    instruction: Literal["start","stop","continue"]


class Orchestrator:
    def __init__(self, 
                 sys_info: SystemConfig = SystemConfig(name="local"),
                 config: OchestratorConfig = OchestratorConfig(name="parsl-local")):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.sys_info = NodeResourceList.from_config(sys_info)
        self.cluster_resource = DistributedClusterResource(nodes=get_nodes(),system_info=self.sys_info)
        self.logger.info("Orchestrator initialized with system: %s", sys_info.__repr__())

        #create DAG
        self.dag = None
        ##start the dagstore and put the dag in it
        self.dagstore = DagStore(config=server_registry.create_config(type="filesystem"))
        if self.config.profile:
            self.dagstore = DagStoreProfiler(self.dagstore)
        self.executor = None
        self.dag_futures: Dict[str, DagFuture] = {}
    
    def report_stats(self):
        """Wait for completion and check for updates"""
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
        return 1
    
    def _submit(self, instruction_queue: Queue, result_queue: Queue):
        """Checks the dag and submits any dag that is not_submitted status"""
        futures = {}
        dag_futures = {}
        while True:
            dag_futures = {}
            dag,last_update = self.dagstore.get_dag()
            if dag is not None:
                graph: DiGraph = dag.graph
                node_execution_order = list(topological_sort(graph))
                counter = 0
                for i, node in enumerate(node_execution_order):
                    node_obj = graph.nodes[node]
                    if node_obj['status'] == NodeStatus.NOT_SUBMITTED:
                        try:
                            self.logger.debug(f"Submitting {node} for execution")
                            args = [self.cluster_resource]
                            # Iterate through dependencies and collect their futures
                            predecessors = list(graph.predecessors(node))
                            if predecessors:
                                self.logger.debug(f"Node {node} has {len(predecessors)} dependencies: {predecessors}")
                                for predecessor in predecessors:
                                    args.append(futures[predecessor])
                            else:
                                self.logger.debug(f"Node {node} has no dependencies")
                            if self.config.profile:
                                futures[node] = self.executor.submit(CallableProfiler(node_obj["callable"]),args)
                            else:
                                futures[node] = self.executor.submit(node_obj["callable"],args)
                            dag_futures[node] = DagFuture(self.dagstore,node)
                            node_obj["status"] = next(node_obj["status"])
                            counter += 1
                        except Exception as e:
                            self.logger.error(f"Submitting {node} failed with exception {e}")
                            node_obj["status"] = NodeStatus.FAILED
                        
                if counter > 0 :
                    self.logger.info(f"Submitted {counter} nodes for execution")
                    self.dagstore.put_dag(dag)
                    result_queue.put(dag_futures)

            ##check the done futures and update the dag store
            ndone = 0
            for node, future in futures.items():
                if future.done():
                    ndone += 1
                    try:
                        result = future.result()
                        dag.graph.nodes[node]["status"] = NodeStatus.COMPLETED
                    except Exception as e:
                        dag.graph.nodes[node]["status"] = NodeStatus.FAILED
            if ndone > 0:
                self.dagstore.put_dag(dag)

            ##receive messages from main process
            try:
                message = instruction_queue.get_nowait()
                self.logger.info(f"Received message: {message.instruction}")
                if message.instruction == "stop":
                    self.logger.info("Stopping submit loop as per instruction.")
                    break
            except Exception:
                pass
            time.sleep(self.config.submit_loop_sleep_time)
        return counter

    def start(self):
        """Start the server submit loop using a thread"""
        self.logger.info("Starting the server submit loop (threaded)")
        self.executor = TapsExecutor(self.config)
        instruction_queue = queue.Queue()
        result_queue = queue.Queue()
        submit_thread = threading.Thread(target=self._submit, args=(instruction_queue, result_queue), daemon=True)
        submit_thread.start()
        self._instruction_queue = instruction_queue
        self._result_queue = result_queue
        self._submit_thread = submit_thread
        return OrchetratorClient(self)

    def stop(self):
        self.logger.info("Stopping the submit loop!")
        self._instruction_queue.put(Message(instruction="stop"))
        self._submit_thread.join(timeout=10)
        if self._submit_thread.is_alive():
            self.logger.warning("Submit thread did not stop in time.")
        self.executor.cleanup()
        self.cluster_resource.cleanup()

    def wait(self, timeout = None):
        start = time.time()
        while True:
            # Retrieve futures from the result queue if available
            try:
                while not self._result_queue.empty():
                    futures_update = self._result_queue.get_nowait()
                    if futures_update:
                        self.dag_futures.update(futures_update)
            except Exception:
                pass
            ndone = 0
            for node, future in self.dag_futures.items():
                if future.done():
                    ndone += 1
            
            time.sleep(1)
            if (ndone >= len(self.dag_futures)) or \
                (timeout is not None and time.time() > start + timeout):
                break


