from SimAIBench.datastore import DataStore, ServerManager
from SimAIBench.config import ServerConfig
from SimAIBench.component import WorkflowComponent
from typing import Literal
import networkx as nx
from .dag import DAG, NodeStatus
import time
import logging
import copy

logger = logging.getLogger(__name__)

class DagStore:
    def __init__(self,config:ServerConfig, start_store:bool = True):
        self._server_config = config
        self._store_started = False
        self.server_info = None
        self.client = None
        ##start the store at the first instantiation
        if start_store:
            self._start_store()
        
    def _start_store(self):
        self.manager = ServerManager("dagstore_server",self._server_config)
        try:
            self.manager.start_server()
            self._store_started = True
            self.server_info = self.manager.get_server_info()
        except Exception as e:
            logger.error(f"Dagstore server failed to start with exception {e}")
    
    def _create_client(self):
        if self.server_info is None:
            self._start_store()
        self.client = DataStore("datastore_client",server_info=self.server_info)

    def get_dag(self):
        if self.client is None:
            self._create_client()
        with self.client.acquire_lock("dagstore"):
            if not self.client.poll_staged_data("dag"):
                return (None,None)
            dag = self.client.stage_read("dag")
            last_updated = self.client.stage_read("last_updated")
        return (dag,last_updated)

    def put_dag(self, dag:DAG):
        if self.client is None:
            self._create_client()
        with self.client.acquire_lock("dagstore"):
            self.client.stage_write("dag",dag)
            self.client.stage_write("last_updated",time.time())

    def update_component(self, workflow_component: WorkflowComponent):
        if self.client is None:
            self._create_client()
        with self.client.acquire_lock("dagstore"):
            if self.client.poll_staged_data("dag"):
                dag:DAG = self.client.stage_read("dag")
                dag.update(workflow_component)
            else:
                dag = DAG({workflow_component.name:workflow_component})
            self.client.stage_write("dag",dag)
            self.client.stage_write("last_updated",time.time())
    
    def copy(self):
        dagstore = DagStore(copy.deepcopy(self._server_config),start_store=False)
        dagstore._store_started = self._store_started
        dagstore.server_info = self.server_info
        logger.info(f"Created a copy of dagstore")
        return dagstore
    
    def cleanup(self):
        logger.info("Cleaning up dagstore")
        if self.manager:
            self.manager.stop_server()