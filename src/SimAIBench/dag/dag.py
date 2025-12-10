import networkx as nx
from typing import Dict
from SimAIBench.component import WorkflowComponent
from SimAIBench.resources import ClusterResource, NodeResourceList, NodeResourceCount, JobResource
from SimAIBench.config import OchestratorConfig
import time
import logging
from functools import partial
from ._callables import RegularCallable, ResourceAwareCallable, MPICallable
from SimAIBench.utils import create_logger
import enum



class NodeStatus(enum.Enum):
    NOT_SUBMITTED = "not_submitted"
    SUBMITTED = "submitted"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

    def __next__(self):
        members = list(self.__class__)
        idx = members.index(self)
        if idx + 1 < len(members):
            return members[idx + 1]
        return self

    

class DAG:
    """
        Class to build implicit DAGs from explitic DAGs give by the user.
    """
    def __init__(self, workflow_components: Dict[str, WorkflowComponent], manage_resources = True):
        self._wcs = workflow_components
        self.manage_resources = manage_resources
        self.logger = create_logger("DAG", subdir="dag")
        self.logger.info(f"Initializing DAG with {len(workflow_components)} workflow components")
        self.graph = self._build_dag(workflow_components=workflow_components)
        self._prepare_callables()
        self.logger.info(f"DAG initialization complete. Graph has {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        

    def _build_dag(self,workflow_components: Dict[str, WorkflowComponent]):
        self.logger.debug("Building DAG from workflow components")
        graph = nx.DiGraph()
        
        ##add all the nodes
        for c,wc in workflow_components.items():
            self.logger.debug(f"Adding node '{c}' with {wc.nnodes} nodes")
            graph.add_node(c,component=wc,status=NodeStatus.NOT_SUBMITTED)
            if wc.nnodes*wc.ppn > 1 and self.manage_resources:
                resource_node = c+"_resource"
                self.logger.debug(f"Adding resource node '{resource_node}' for multi-node component '{c}'")
                graph.add_node(resource_node,component=WorkflowComponent(wc.name+"_resource",lambda x: x,"local"),status=NodeStatus.NOT_SUBMITTED)
            self.logger.debug(f"Node '{c}': {graph.nodes[c]}")
            
        
        ##add all the edges
        for c,wc in workflow_components.items():
            if wc.nnodes*wc.ppn > 1 and self.manage_resources:
                for d in workflow_components[c].dependencies:
                    self.logger.debug(f"Adding edge: {d} -> {c}_resource")
                    graph.add_edge(d,c+"_resource")
                self.logger.debug(f"Adding edge: {c}_resource -> {c}")
                graph.add_edge(c+"_resource",c)
            else:
                for d in workflow_components[c].dependencies:
                    self.logger.debug(f"Adding edge: {d} -> {c}")
                    graph.add_edge(d,c)
        
        return graph

    def _prepare_callables(self):
        """Create standardized callables for all nodes"""
        self.logger.debug("Preparing callables for all nodes")
        for node in self.graph.nodes:
            self._prepare_callables_for_node(node)
        self.logger.debug("Callable preparation complete")
    
    def _prepare_callables_for_node(self, node_name: str):
        """Create callable for a specific node"""
        self.logger.debug(f"Preparing callable for node '{node_name}'")
        wc:WorkflowComponent = self.graph.nodes[node_name]["component"]
        if wc.nnodes*wc.ppn > 1 and self.manage_resources:
            self.logger.debug(f"Creating MPI callable for multi-node component '{node_name}'")
            self.graph.nodes[node_name]["callable"] = self._get_mpi_callable(wc)
        elif "_resource" in node_name:
            self.logger.debug(f"Creating resource-aware callable for resource node '{node_name}'")
            original_node_name = node_name.replace("_resource","")
            original_wc = self.graph.nodes[original_node_name]["component"]
            self.graph.nodes[node_name]["callable"] = self._get_resource_aware_callable(original_wc)
        else:
            self.logger.debug(f"Creating regular callable for node '{node_name}'")
            self.graph.nodes[node_name]["callable"] = self._get_regular_callable(wc)


    def update(self, workflow_component: WorkflowComponent):
        """Update the dag with a new workflow component"""
        self.logger.info(f"Updating DAG with new component '{workflow_component.name}'")
        
        # Check if all dependencies exist in the graph
        for dep in workflow_component.dependencies:
            if dep not in self.graph.nodes():
                self.logger.error(f"Dependency '{dep}' for component '{workflow_component.name}' does not exist in the graph")
                raise ValueError(f"Dependency '{dep}' for component '{workflow_component.name}' does not exist in the graph")
        
        self.logger.debug(f"All dependencies verified for component '{workflow_component.name}'")
        
        # Add the new node to the graph
        self.graph.add_node(workflow_component.name, component=workflow_component, status=NodeStatus.NOT_SUBMITTED)
        self.logger.debug(f"Added node '{workflow_component.name}' to graph")
        
        # If it's a multi-node component, add resource node
        if workflow_component.nnodes*workflow_component.ppn > 1 and self.manage_resources:
            resource_node = workflow_component.name + "_resource"
            resource_wc = WorkflowComponent(workflow_component.name + "_resource", lambda x: x, "local")
            self.graph.add_node(resource_node, 
                               component=resource_wc, 
                               status=NodeStatus.NOT_SUBMITTED)
            self.logger.debug(f"Added resource node '{resource_node}' for multi-node component")
        
        # Add edges for dependencies
        if workflow_component.nnodes*workflow_component.ppn > 1 and self.manage_resources:
            for dep in workflow_component.dependencies:
                self.graph.add_edge(dep, workflow_component.name + "_resource")
                self.logger.debug(f"Added edge: {dep} -> {workflow_component.name}_resource")
            self.graph.add_edge(workflow_component.name + "_resource", workflow_component.name)
            self.logger.debug(f"Added edge: {workflow_component.name}_resource -> {workflow_component.name}")
        else:
            for dep in workflow_component.dependencies:
                self.graph.add_edge(dep, workflow_component.name)
                self.logger.debug(f"Added edge: {dep} -> {workflow_component.name}")
        
        # Create callables for the new nodes
        self._prepare_callables_for_node(workflow_component.name)
        if workflow_component.nnodes*workflow_component.ppn > 1 and self.manage_resources:
            self._prepare_callables_for_node(workflow_component.name + "_resource")
        
        self.logger.info(f"Successfully updated DAG with component '{workflow_component.name}'")

    def _get_resource_aware_callable(self, workflow_component: WorkflowComponent):
        return ResourceAwareCallable(workflow_component)
    
    def _get_mpi_callable(self,workflow_component: WorkflowComponent):
        return MPICallable(workflow_component)
    
    def _get_regular_callable(self, workflow_component: WorkflowComponent):
        return RegularCallable(workflow_component)