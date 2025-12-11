import logging

from SimAIBench.component import WorkflowComponent
from SimAIBench.resources import *
from SimAIBench.dag import DagFuture, DagStore
from typing import Dict
from SimAIBench.dag import DAG
import copy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .server import Orchestrator

class OrchetratorClient:
    """This is the client of Orchestrator server. This will enable any task to modify the orchestrator DAG"""
    def __init__(self,orchestrator: 'Orchestrator'):
        self.dagstore: DagStore = orchestrator.dagstore.copy()

    def submit(self,wokflow_component:WorkflowComponent):
        """the client simply updates the dag"""
        self.dagstore.update_component(wokflow_component)
        return DagFuture(self.dagstore, wokflow_component.name)
    
    def build_dag(self,workflow_components: Dict[str, WorkflowComponent]):
        """
            Build the dag
        """
        self.dag = DAG(workflow_components)
        self.dagstore.put_dag(self.dag)
    
    def get_status(self):
        raise NotImplementedError
        
    
