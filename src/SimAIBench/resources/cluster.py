from typing import List, Dict, Optional
from abc import ABC, abstractmethod
from SimAIBench.datastore import DataStore, ServerManager
import os
import contextlib
import uuid
import time
import logging
from SimAIBench.resources import NodeResource, JobResource, NodeResourceCount, NodeResourceList
from SimAIBench.config import server_registry

SUPPORTED_BACKENDS = ["redis","filesystem"]
# Configure logging
logger = logging.getLogger(__name__)

class ClusterResource(ABC):
    """
    Abstract base class for managing cluster resources and job allocation.
    Note:
        This is an abstract base class and cannot be instantiated directly. Concrete
        implementations must provide the allocate() and deallocate() methods.
    """

    def __init__(self, nodes: List[str], system_info: NodeResource):
        logger.info(f"Initializing {self.__class__.__name__} with {len(nodes)} nodes with each node config {system_info.__repr__()}")
        self._system_info = system_info
        self._nodes = {node: system_info for node in nodes}
        logger.debug(f"Node configuration: {list(self._nodes.keys())}")

    @abstractmethod
    def allocate(self, job_resource: JobResource):
        pass

    @abstractmethod
    def deallocate(self, job_resource: JobResource):
        pass

    def _can_allocate(self, job_resource: JobResource) -> bool | List[str]:
        """Check if the job resource can be allocated."""
        logger.debug(f"Checking allocation feasibility for job with {len(job_resource.resources)} resources")
        
        if not job_resource.nodes:
            # Need to find at least len(resources) nodes to allocate
            job_counter = 0
            cluster_counter = 0
            allocated_nodes = []
            node_names = list(self._nodes.keys())
            
            logger.debug("Auto-selecting nodes for allocation")
            
            while True:
                if job_counter >= len(job_resource.resources):
                    logger.debug(f"Successfully found {len(allocated_nodes)} suitable nodes: {allocated_nodes}")
                    return allocated_nodes
                
                if cluster_counter >= len(self._nodes):
                    logger.debug(f"Insufficient resources: only found {len(allocated_nodes)} nodes, need {len(job_resource.resources)}")
                    return []  
                
                resource_req = job_resource.resources[job_counter]
                node_name = node_names[cluster_counter]
                
                if resource_req in self._nodes[node_name]:
                    allocated_nodes.append(node_name)
                    job_counter += 1
                    logger.debug(f"Node {node_name} can satisfy resource requirement {job_counter}")
                
                cluster_counter += 1
        else:
            logger.debug(f"Checking specific nodes: {job_resource.nodes}")
            for node_id, node_name in enumerate(job_resource.nodes):
                if node_name not in self._nodes:
                    logger.error(f"Node {node_name} not found in cluster")
                    return False
                
                available = self._nodes[node_name]
                resource_req = job_resource.resources[node_id]
                
                if resource_req not in available:
                    logger.warning(f"Node {node_name} cannot satisfy resource requirement: need {resource_req}, available {available}")
                    return False
            
            logger.debug("All specified nodes can satisfy requirements")
            return True
    
    def __repr__(self) -> str:
        """Return string representation of the cluster."""
        node_info = []
        for node_name, resource in self._nodes.items():
            node_info.append(f"{node_name}: {resource}")
        
        nodes_str = "\n  ".join(node_info)
        return f"{self.__class__.__name__}(\n  {nodes_str}\n)"

class LocalClusterResource(ClusterResource):
    """
    Manages resource allocation and deallocation for a cluster of nodes.
    Attributes:
        _system_info (NodeResource): The system information template for nodes.
        _nodes (Dict[str, NodeResource]): Mapping of node names to their available resources.
    Args:
        nodes (List[str]): List of node names in the cluster.
        system_info (NodeResource): Resource information template applied to all nodes.
    """

    def allocate(self, job_resource: JobResource) -> tuple[bool, JobResource]:
        """Allocate specific resource IDs."""
        logger.info(f"Starting allocation for job with {len(job_resource.resources)} resource requirements")

        allocation_result = self._can_allocate(job_resource)
        if not allocation_result:
            logger.error("Allocation failed: insufficient resources")
            return False, job_resource
        
        # Track original state before allocation
        original_state = {}
        allocated_resources = []
        
        if not job_resource.nodes:
            allocated_nodes = allocation_result
            logger.info(f"Allocating resources on auto-selected nodes: {allocated_nodes}")
            
            # Capture original state and perform allocation
            for node_id, node_name in enumerate(allocated_nodes):
                resource_req = job_resource.resources[node_id]
                original_state[node_name] = self._nodes[node_name]
                logger.debug(f"Requesting {resource_req} from node {node_name}")
                self._nodes[node_name] = self._nodes[node_name] - resource_req
                
                # Calculate what was actually allocated
                allocated_resource = original_state[node_name] - self._nodes[node_name]
                allocated_resources.append(allocated_resource)
                logger.debug(f"Allocated {allocated_resource} on node {node_name}")
                logger.debug(f"Remaining resources on node {node_name} {self._nodes[node_name]}")
            
            # Return JobResource with actual allocated resources
            logger.info(f"Allocation successful.")
            return True, JobResource(resources=allocated_resources, nodes=allocated_nodes)
        else:
            logger.info(f"Allocating resources on specified nodes: {job_resource.nodes}")
            
            # Handle specified nodes case
            for node_id, node_name in enumerate(job_resource.nodes):
                resource_req = job_resource.resources[node_id]
                original_state[node_name] = self._nodes[node_name]
                self._nodes[node_name] = self._nodes[node_name] - resource_req
                
                # Calculate what was actually allocated
                allocated_resource = original_state[node_name] - self._nodes[node_name]
                allocated_resources.append(allocated_resource)
                logger.debug(f"Allocated {allocated_resource} on node {node_name}")
            
            logger.info("Allocation successful")
            return True, JobResource(resources=allocated_resources, nodes=job_resource.nodes)
    
    def deallocate(self, job_resource: JobResource) -> bool:
        """Deallocate the resources"""
        if not job_resource.nodes:
            logger.error("Deallocation failed: JobResource must have nodes specified")
            raise ValueError("JobResource must have nodes specified for deallocation")
        
        logger.info(f"Starting deallocation for {len(job_resource.nodes)} nodes: {job_resource.nodes}")
        
        for node_id, node_name in enumerate(job_resource.nodes):
            resource_req = job_resource.resources[node_id]
            self._nodes[node_name] += resource_req
            logger.debug(f"Deallocated {resource_req} from node {node_name}")
        
        logger.info("Deallocation successful")
        return True

    
class DistributedClusterResource(ClusterResource):
    """
    Distributed cluster resource manager. It uses SimAI-Bench's DataStore
    """
    def __init__(self, nodes: List[str], system_info: NodeResource, backend:str = "filesystem"):
        super().__init__(nodes, system_info)
        self._server_manager = None
        self._data_store = None
        logger.info("Initializing distributed cluster resource manager")
        if backend not in SUPPORTED_BACKENDS:
            logger.error(f"Backend {backend} is not one of supported backend {SUPPORTED_BACKENDS}")
            raise
        self.ds_backend = backend
        
        self._start_server()

    def _start_server(self) -> bool:
        logger.info("Starting data store server")
        
        if self.ds_backend == "redis":
            # Hardcode the server config to use a redis cluster
            default_redis_server = os.path.join(os.getenv("HOME"), "redis/src/redis-cli")
            server_config = {
            "type": "redis",
            "is_clustered": True,
            "server-address": ",".join([f"{node}:7257" for node in self._nodes.keys()]),
            "redis-server-exe": f"{os.environ.get('SIMAIBENCH_REDIS_CLI', default_redis_server)}"
            }
            server_config = server_registry.create_config(**server_config)
        else:
            server_config = {
                "type": self.ds_backend,
            }
            server_config = server_registry.create_config(**server_config)
        
        logger.debug(f"Server config: {server_config}")
        
        try:      
            self._server_manager = ServerManager("resource_server", server_config)
            self._server_manager.start_server()
            logger.info("Resource server started successfully")
            
            ds = self._start_data_store()
            
            if not ds:
                logger.error("Failed to start data store")
                return False
            
            # Put all the info related to the node (as serializable data)
            logger.debug("Initializing node data in DataStore")
            for node, resource in self._nodes.items():
                ds.stage_write(node, resource)
            
            logger.info("Server initialization complete")
            return True
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            return False
    
    def _start_data_store(self) -> Optional[DataStore]:
        if not self._server_manager:
            logger.debug("Server manager not available, starting server")
            self._start_server()
        
        try:
            logger.debug("Creating DataStore connection")
            ds = DataStore("resource_store", server_info=self._server_manager.get_server_info())
            logger.info("DataStore connection established")
            return ds
        except Exception as e:
            logger.error(f"Failed to start data store: {e}")
            return None
    
    def _update_from_data_store(self):
        """Sync local state from DataStore."""
        logger.debug("Syncing state from DataStore")
        
        if not self._data_store:
            self._data_store = self._start_data_store()
        
        for node_name in self._nodes.keys():
            try:
                self._nodes[node_name] = self._data_store.stage_read(node_name)
                logger.debug(f"Updated node {node_name} from DataStore")
            except Exception as e:
                logger.error(f"Failed to read node {node_name} from DataStore: {e}")
    
    def _update_to_data_store(self):
        """Sync local state to DataStore."""
        logger.debug("Syncing state to DataStore")
        
        if not self._data_store:
            self._data_store = self._start_data_store()
        
        for node_name, resource in self._nodes.items():
            try:
                self._data_store.stage_write(node_name, resource)
                logger.debug(f"Updated DataStore for node {node_name}")
            except Exception as e:
                logger.error(f"Failed to stage write for node {node_name}: {e}")
    
    def allocate(self, job_resource: JobResource) -> tuple[bool, JobResource]:
        """Allocate specific resource IDs with distributed locking."""
        logger.info(f"Starting distributed allocation for job with {len(job_resource.resources)} resource requirements")
        if not self._data_store:
            self._data_store = self._start_data_store()
        with self._data_store.acquire_lock("cluster_allocation"):
            logger.debug("Acquired allocation lock")
            self._update_from_data_store()
            
            allocation_result = self._can_allocate(job_resource)
            if not allocation_result:
                logger.warning("Distributed allocation failed: insufficient resources")
                return False, job_resource
            
            # Track original state before allocation
            original_state = {}
            allocated_resources = []
            
            if not job_resource.nodes:
                allocated_nodes = allocation_result
                logger.info(f"Allocating resources on auto-selected nodes: {allocated_nodes}")
                
                # Capture original state and perform allocation
                for node_id, node_name in enumerate(allocated_nodes):
                    resource_req = job_resource.resources[node_id]
                    original_state[node_name] = self._nodes[node_name]
                    self._nodes[node_name] = self._nodes[node_name] - resource_req
                    
                    # Calculate what was actually allocated
                    allocated_resource = original_state[node_name] - self._nodes[node_name]
                    allocated_resources.append(allocated_resource)
                    logger.debug(f"Allocated {allocated_resource} on node {node_name}")
                
                # Return JobResource with actual allocated resources
                new_job_resource = JobResource(resources=allocated_resources, nodes=allocated_nodes)
            else:
                logger.info(f"Allocating resources on specified nodes: {job_resource.nodes}")
                
                # Handle specified nodes case
                for node_id, node_name in enumerate(job_resource.nodes):
                    resource_req = job_resource.resources[node_id]
                    original_state[node_name] = self._nodes[node_name]
                    self._nodes[node_name] = self._nodes[node_name] - resource_req
                    
                    # Calculate what was actually allocated
                    allocated_resource = original_state[node_name] - self._nodes[node_name]
                    allocated_resources.append(allocated_resource)
                    logger.debug(f"Allocated {allocated_resource} on node {node_name}")
                
                new_job_resource = JobResource(resources=allocated_resources, nodes=job_resource.nodes)
            
            self._update_to_data_store()
            logger.info("Distributed allocation successful")
            return True, new_job_resource

    def deallocate(self, job_resource: JobResource) -> bool:
        """Deallocate the resources with distributed locking."""
        if not job_resource.nodes:
            logger.error("Distributed deallocation failed: JobResource must have nodes specified")
            raise ValueError("JobResource must have nodes specified for deallocation")
        
        logger.info(f"Starting distributed deallocation for {len(job_resource.nodes)} nodes: {job_resource.nodes}")

        if not self._data_store:
            self._data_store = self._start_data_store()
        
        with self._data_store.acquire_lock("cluster_allocation"):
            logger.debug("Acquired allocation lock for deallocation")
            self._update_from_data_store()
            
            for node_id, node_name in enumerate(job_resource.nodes):
                resource_req = job_resource.resources[node_id]
                self._nodes[node_name] += resource_req
                logger.debug(f"Deallocated {resource_req} from node {node_name}")
            
            self._update_to_data_store()
            logger.info("Distributed deallocation successful")
            return True
    
    def get_cluster_status(self) -> Dict[str, any]:
        """Get current cluster status."""
        logger.debug("Retrieving cluster status")
        
        with self._data_store.acquire_lock("cluster_status", acquire_timeout=5):
            self._update_from_data_store()
            
            status = {
                "nodes": {name: resource.counts for name, resource in self._nodes.items()},
                "timestamp": time.time()
            }
            
            logger.debug(f"Cluster status retrieved for {len(status['nodes'])} nodes")
            return status

    def cleanup(self):
        """Cleanup when object is destroyed."""
        logger.info("Cleaning up DistributedClusterResource")
        if self._server_manager:
            try:
                self._server_manager.stop_server()
                logger.info("Server stopped successfully")
            except Exception as e:
                logger.error(f"Error stopping server: {e}")

if __name__ == "__main__":
    # Configure logging for the main execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    sys_info = NodeResourceList(cpus=list(range(104)),gpus=list(range(12)))
    cluster = LocalClusterResource(nodes=[f"node:{str(i)}" for i in range(10)],system_info=sys_info)

    print("*"*100)
    print(cluster)

    def print_allocated(allocated_job):
        for node_id,node_name in enumerate(allocated_job.nodes):
            print(node_name)
            print(allocated_job.resources[node_id])

    resources = []
    resources.append(NodeResourceCount(ncpus=10,ngpus=6))
    job = JobResource(resources=resources)
    allocated,allocated_job = cluster.allocate(job)
    if allocated:
        print("*"*100)
        print(cluster)
        # print_allocated(allocated_job)
        cluster.deallocate(allocated_job)
        print("*"*100)
        print(cluster)
    else:
        print("Not allocated")