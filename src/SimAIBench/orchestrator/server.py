import subprocess
import time
import os
import multiprocessing as mp
from multiprocessing import Queue
import logging
import threading
import queue

from typing import List, Union, Dict, Any
from SimAIBench.component import WorkflowComponent
from SimAIBench.resources import *
from SimAIBench.dag import DAG, DagStore, NodeStatus, DagFuture
from SimAIBench.executors import TapsExecutor,DragonExecutor
from SimAIBench.utils import get_nodes
from concurrent.futures import Future
from SimAIBench.config import server_registry
from networkx import topological_sort, DiGraph
from pydantic import BaseModel
from typing import Literal
from .client import OrchetratorClient
from SimAIBench.profiling import DagStoreProfiler, CallableProfiler

from SimAIBench.config import SystemConfig, OchestratorConfig, server_registry
from SimAIBench.datastore import DataStore, ServerManager

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     encoding='utf-8',
#     errors='replace',
#     # handlers=[
#     #     logging.StreamHandler(),
#     #     logging.FileHandler('orchestrator.log',  mode='w', encoding='utf-8')
#     # ]
# )

class Message(BaseModel):
    instruction: Literal["start","stop","continue"]


class Orchestrator:
    def __init__(self, 
                 sys_info: SystemConfig = SystemConfig(name="local"),
                 config: OchestratorConfig = OchestratorConfig(name="parsl-local")):
        self.logger = None
        self._init_logger()
        self.config = config
        self.sys_info = NodeResourceList.from_config(sys_info)
        self.cluster_resource = DistributedClusterResource(nodes=get_nodes(),system_info=self.sys_info)
        self.logger.info("Orchestrator initialized with system: %s", sys_info.__repr__())

        #create DAG
        self.dag = None
        ##start the dagstore and put the dag in it
        tmp_dir = os.environ.get("SIMAIBENCH_DAGSTORE_DIR",
                                 os.path.join(os.getcwd(), "./.dagstore_tmp"))
        self.dagstore = DagStore(config=server_registry.create_config(type="filesystem",server_address=tmp_dir))
        if self.config.profile:
            tmp_dir = os.environ.get("SIMAIBENCH_PROFILER_TMPDIR",
                                 os.path.join(os.getcwd(), "./.profiler_tmp"))
            self.profiler_server = ServerManager("profiler_server",config=server_registry.create_config("filesystem",server_address=tmp_dir))
            self.profiler_server.start_server()
            self.profiler_server_info = self.profiler_server.get_server_info()
            self.profiler_store = DataStore("profiler_store",self.profiler_server_info)
            self.dagstore = DagStoreProfiler(self.dagstore,self.profiler_server_info)
        self.executor = None
        self.dag_futures: Dict[str, DagFuture] = {}
    
    def _init_logger(self):
        log_level_str = os.environ.get("SIMAIBENCH_LOGLEVEL","INFO")
        if log_level_str == "INFO":
            log_level = logging.INFO
        elif log_level_str == "DEBUG":
            log_level = logging.DEBUG
        else:
            log_level = logging.INFO

        self.logger = logging.getLogger("Orchestrator")
        self.logger.setLevel(log_level)
        if not self.logger.handlers:
            log_dir = os.path.join(os.getcwd(), "logs")
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, f"orchestrator.log")
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

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
        while True:
            dag_futures = {}
            dag,last_update = self.dagstore.get_dag()
            if dag is not None:
                if self.config.profile:
                    for node in dag.graph.nodes():
                        node_obj = dag.graph.nodes[node]
                        node_obj["callable"] = CallableProfiler(node_obj["callable"],self.profiler_server_info)

                dag, futures_update = \
                    self.executor.submit_dag(self.cluster_resource, dag)
                futures.update(futures_update)
                for node in futures_update:
                    dag_futures[node] = DagFuture(self.dagstore, node)
                
                if len(futures_update) > 0 :
                    self.logger.info(f"Submitted {len(futures_update)} nodes for execution")
                    self.dagstore.put_dag(dag)
                    result_queue.put(dag_futures)

            ##check the done futures and update the dag store
            ndone = 0
            for node, future in futures.items():
                if dag.graph.nodes[node]["status"] == NodeStatus.COMPLETED or \
                    dag.graph.nodes[node]["status"] == NodeStatus.FAILED:
                    continue
                if future.done():
                    ndone += 1
                    try:
                        result = future.result()
                        dag.graph.nodes[node]["status"] = NodeStatus.COMPLETED
                        self.logger.info(f"{node} completed successfully!")
                    except Exception as e:
                        dag.graph.nodes[node]["status"] = NodeStatus.FAILED
                        self.logger.info(f"{node} failed with exception {e}")
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

    def start(self):
        """Start the server submit loop using a thread"""
        self.logger.info("Starting the server submit loop (threaded)")
        if self.config.name == "dragon":
            self.executor = DragonExecutor(self.config,self.sys_info)
        else:
            self.executor = TapsExecutor(self.config,self.sys_info)
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
        self.dagstore.cleanup()
        self.executor.cleanup()
        self.cluster_resource.cleanup()
        if self.config.profile:
            ##dump server-info
            self.profiler_store.dump()
            self.profiler_server.stop_server()

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
            if (len(self.dag_futures) > 0 and (ndone >= len(self.dag_futures))) or \
                (timeout is not None and time.time() > start + timeout):
                break

    def get_profile_store(self):
        if self.config.profile:
            return self.profiler_server_info
        return None

