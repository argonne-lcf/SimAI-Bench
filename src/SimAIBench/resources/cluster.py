from typing import List, Dict, Optional
from abc import ABC, abstractmethod
from SimAIBench.datastore import DataStore, ServerManager
import os
import contextlib
import uuid
import time
from SimAIBench.resources import NodeResource, JobResource, NodeResourceCount, NodeResourceList

class ClusterResource(ABC):
    """
    Abstract base class for managing cluster resources and job allocation.
    Note:
        This is an abstract base class and cannot be instantiated directly. Concrete
        implementations must provide the allocate() and deallocate() methods.
    """

    def __init__(self, nodes: List[str], system_info: NodeResource):
        self._system_info = system_info
        self._nodes = {node: system_info for node in nodes}

    @abstractmethod
    def allocate(self, job_resource: JobResource):
        pass

    @abstractmethod
    def deallocate(self, job_resource: JobResource):
        pass

    def _can_allocate(self, job_resource: JobResource) -> bool | List[str]:
        """Check if the job resource can be allocated."""
        if not job_resource.nodes:
            # Need to find at least len(resources) nodes to allocate
            job_counter = 0
            cluster_counter = 0
            allocated_nodes = []
            node_names = list(self._nodes.keys())
            
            while True:
                if job_counter >= len(job_resource.resources):
                    return allocated_nodes
                
                if cluster_counter >= len(self._nodes):
                    return []  
                
                resource_req = job_resource.resources[job_counter]
                node_name = node_names[cluster_counter]
                
                if resource_req in self._nodes[node_name]:
                    allocated_nodes.append(node_name)
                    job_counter += 1
                
                cluster_counter += 1
        else:
            for node_id, node_name in enumerate(job_resource.nodes):
                if node_name not in self._nodes:
                    return False
                
                available = self._nodes[node_name]
                resource_req = job_resource.resources[node_id]
                
                if resource_req not in available:
                    return False
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

        allocation_result = self._can_allocate(job_resource)
        if not allocation_result:
            return False, job_resource
        
        # Track original state before allocation
        original_state = {}
        allocated_resources = []
        
        if not job_resource.nodes:
            allocated_nodes = allocation_result
            # Capture original state and perform allocation
            for node_id, node_name in enumerate(allocated_nodes):
                resource_req = job_resource.resources[node_id]
                original_state[node_name] = self._nodes[node_name]
                self._nodes[node_name] = self._nodes[node_name] - resource_req
                
                # Calculate what was actually allocated
                allocated_resource = original_state[node_name] - self._nodes[node_name]
                allocated_resources.append(allocated_resource)
            
            # Return JobResource with actual allocated resources
            return True, JobResource(resources=allocated_resources, nodes=allocated_nodes)
        else:
            # Handle specified nodes case
            for node_id, node_name in enumerate(job_resource.nodes):
                resource_req = job_resource.resources[node_id]
                original_state[node_name] = self._nodes[node_name]
                self._nodes[node_name] = self._nodes[node_name] - resource_req
                
                # Calculate what was actually allocated
                allocated_resource = original_state[node_name] - self._nodes[node_name]
                allocated_resources.append(allocated_resource)
            
            return True, JobResource(resources=allocated_resources, nodes=job_resource.nodes)
    
    def deallocate(self, job_resource: JobResource) -> bool:
        """Deallocate the resources"""
        if not job_resource.nodes:
            raise ValueError("JobResource must have nodes specified for deallocation")
        
        for node_id, node_name in enumerate(job_resource.nodes):
            resource_req = job_resource.resources[node_id]
            self._nodes[node_name] += resource_req
        return True

    
class DistributedClusterResource(ClusterResource):
    """
    Distributed cluster resource manager. It uses SimAI-Bench's DataStore
    """
    def __init__(self, nodes: List[str], system_info: NodeResource):
        super().__init__(nodes, system_info)
        self._server_manager = None
        self._data_store = None
        self._start_server()

    def _start_server(self) -> bool:
        # Hardcode the server config to use a redis cluster
        default_redis_server = os.path.join(os.getenv("HOME"), "redis/src/redis-cli")
        server_config = {
            "type": "redis",
            "is_clustered": True,
            "server-address": ",".join([f"{node}:7257" for node in self._nodes.keys()]),
            "redis-server-exe": f"{os.environ.get('SIMAIBENCH_REDIS_CLI', default_redis_server)}"
        }  
        try:      
            self._server_manager = ServerManager("resource_server", server_config)
            self._server_manager.start_server()
            ds = self._start_data_store()
            
            if not ds:
                return False
            
            # Put all the info related to the node (as serializable data)
            for node, resource in self._nodes.items():
                ds.stage_write(node, resource)
            return True
        except Exception as e:
            print(f"Failed to start server: {e}")
            return False
    
    def _start_data_store(self) -> Optional[DataStore]:
        if not self._server_manager:
            self._start_server()
        
        try:
            ds = DataStore("resource_store", server_info=self._server_manager.get_server_info())
            return ds
        except Exception as e:
            print(f"Failed to start data store: {e}")
            return None
    
    @contextlib.contextmanager
    def _redis_lock(self, lock_name: str, acquire_timeout: int = 100, lock_timeout: int = 300):
        """
        Acquire a redis lock.
        adapted from https://redis.io/glossary/redis-lock/
        """
        lock_key = f"lock:{lock_name}"
        lock_identifier = str(uuid.uuid4()) + ":" + str(time.time())
        
        if not self._data_store:
            self._data_store = self._start_data_store()
        
        if not self._data_store:
            raise RuntimeError("Cannot acquire lock: DataStore unavailable")
        
        redis_conn = self._data_store.redis_client[0]
        end = time.time() + acquire_timeout

        try:
            acquired = False
            while time.time() < end:
                if redis_conn.set(lock_key, lock_identifier, nx=True, ex=lock_timeout):
                    acquired = True
                    break
                time.sleep(0.01)
            
            if not acquired:
                raise TimeoutError("Lock not acquired!")
            
            yield
            
        finally:
            if acquired:
                current_value = redis_conn.get(lock_key)
                if current_value and current_value.decode() == lock_identifier:
                    redis_conn.delete(lock_key)
    
    def _update_from_data_store(self):
        """Sync local state from DataStore."""
        if not self._data_store:
            self._data_store = self._start_data_store()
        
        for node_name in self._nodes.keys():
            try:
                self._nodes[node_name] = self._data_store.stage_read(node_name)
            except Exception as e:
                print(f"Failed to read node {node_name} from DataStore: {e}")
    
    def _update_to_data_store(self):
        """Sync local state to DataStore."""
        if not self._data_store:
            self._data_store = self._start_data_store()
        
        for node_name, resource in self._nodes.items():
            try:
                self._data_store.stage_write(node_name, resource)
            except Exception as e:
                print(f"Failed to stage write for node {node_name}: {e}")
    
    def allocate(self, job_resource: JobResource) -> tuple[bool, JobResource]:
        """Allocate specific resource IDs with distributed locking."""
        with self._redis_lock("cluster_allocation"):
            self._update_from_data_store()
            
            allocation_result = self._can_allocate(job_resource)
            if not allocation_result:
                return False, job_resource
            
            # Track original state before allocation
            original_state = {}
            allocated_resources = []
            
            if not job_resource.nodes:
                allocated_nodes = allocation_result
                # Capture original state and perform allocation
                for node_id, node_name in enumerate(allocated_nodes):
                    resource_req = job_resource.resources[node_id]
                    original_state[node_name] = self._nodes[node_name]
                    self._nodes[node_name] = self._nodes[node_name] - resource_req
                    
                    # Calculate what was actually allocated
                    allocated_resource = original_state[node_name] - self._nodes[node_name]
                    allocated_resources.append(allocated_resource)
                
                # Return JobResource with actual allocated resources
                new_job_resource = JobResource(resources=allocated_resources, nodes=allocated_nodes)
            else:
                # Handle specified nodes case
                for node_id, node_name in enumerate(job_resource.nodes):
                    resource_req = job_resource.resources[node_id]
                    original_state[node_name] = self._nodes[node_name]
                    self._nodes[node_name] = self._nodes[node_name] - resource_req
                    
                    # Calculate what was actually allocated
                    allocated_resource = original_state[node_name] - self._nodes[node_name]
                    allocated_resources.append(allocated_resource)
                
                new_job_resource = JobResource(resources=allocated_resources, nodes=job_resource.nodes)
            
            self._update_to_data_store()
            return True, new_job_resource

    def deallocate(self, job_resource: JobResource) -> bool:
        """Deallocate the resources with distributed locking."""
        if not job_resource.nodes:
            raise ValueError("JobResource must have nodes specified for deallocation")
        
        with self._redis_lock("cluster_allocation"):
            self._update_from_data_store()
            
            for node_id, node_name in enumerate(job_resource.nodes):
                resource_req = job_resource.resources[node_id]
                self._nodes[node_name] += resource_req
            
            self._update_to_data_store()
            return True
    
    def get_cluster_status(self) -> Dict[str, any]:
        """Get current cluster status."""
        with self._redis_lock("cluster_status", acquire_timeout=5):
            self._update_from_data_store()
            
            return {
                "nodes": {name: resource.counts for name, resource in self._nodes.items()},
                "timestamp": time.time()
            }

    def __del__(self):
        """Cleanup when object is destroyed."""
        if self._server_manager:
            try:
                self._server_manager.stop_server()
            except:
                pass

if __name__ == "__main__":
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
    
    