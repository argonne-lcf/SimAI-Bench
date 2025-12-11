from abc import ABC, abstractmethod
from SimAIBench.component import WorkflowComponent
import logging
import subprocess
import base64
import cloudpickle
from SimAIBench.resources import ClusterResource, JobResource, NodeResourceCount, NodeResourceList
from SimAIBench.utils import create_logger
import time
from functools import partial
import numpy as np
from ._utils import *
import os
import stat
import copy

# Create logger name as module-level constant (serializable)
LOGGER_NAME = __name__

class Callable(ABC):

    def __init__(self,workflow_component: WorkflowComponent):
        self.__name__ = workflow_component.name
        self.logger = None
    
    def _init_logger(self):
        if self.logger is not None:
            return
        self.logger = create_logger(f"Callable_{self.__name__}", subdir="callables")

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

class RegularCallable(Callable):
    """Serializable callable for regular workflow components."""

    def __init__(self, workflow_component: WorkflowComponent):
        super().__init__(workflow_component)
        # EXPLICITLY capture only the data you need (not the entire object)
        self.component_name = workflow_component.name
        self.component_type = getattr(workflow_component, 'type', 'unknown')
        self.return_array = np.empty(workflow_component.return_dim) if len(workflow_component.return_dim) != 0 else None
        
        # Handle executable - store it appropriately based on type
        if callable(workflow_component.executable):
            # For callable executables, serialize them
            try:
                self.executable_type = 'callable'
                self.serialized_executable = base64.b64encode(
                    cloudpickle.dumps(workflow_component.executable)
                ).decode('ascii')
            except Exception as e:
                raise ValueError(f"Cannot serialize executable for {workflow_component.name}: {e}")
        else:
            # For string executables, store as-is
            self.executable_type = 'string'
            self.executable_string = str(workflow_component.executable)
        
        # Handle args - ensure they're serializable
        component_args = getattr(workflow_component, "args", None)
        if component_args:
            try:
                # Test if args are serializable
                cloudpickle.dumps(component_args)
                self.args = component_args
                self.has_args = True
            except Exception as e:
                raise ValueError(f"Cannot serialize args for {workflow_component.name}: {e}")
        else:
            self.args = {}
            self.has_args = False
    
    def __call__(self, cluster_resource: ClusterResource, *results):
        # Create fresh logger in the worker process
        self._init_logger()
        logger = self.logger
        logger.info(f"Executing regular component '{self.component_name}'")
        
        try:
            if self.executable_type == 'callable':
                # Deserialize and execute the callable
                logger.info(f"Executing callable for '{self.component_name}'")
                
                # Deserialize the function
                executable = cloudpickle.loads(
                    base64.b64decode(self.serialized_executable.encode('ascii'))
                )
                
                # Execute with or without args
                if self.has_args:
                    result = executable(**self.args)
                else:
                    result = executable()
                    
                logger.info(f"Callable execution completed for '{self.component_name}'")
                return result
                
            else:  # string executable
                logger.debug(f"Executing shell command for '{self.component_name}': {self.executable_string}")
                
                result = subprocess.run(
                    self.executable_string, 
                    shell=True, 
                    capture_output=True, 
                    text=True
                )
                
                logger.info(f"Shell command completed for '{self.component_name}' with return code: {result.returncode}")
                
                if result.returncode != 0:
                    logger.error(f"Shell command failed for '{self.component_name}': {result.stderr}")
                
                return self.return_array
                
        except Exception as e:
            logger.error(f"Exception during execution of '{self.component_name}': {str(e)}")
            raise

    def __repr__(self):
        return f"RegularCallable(component='{self.component_name}', type='{self.executable_type}')"
    
class ResourceAwareCallable(Callable):
    """Callable that allocates cluster resources for multi-node components."""
    
    def __init__(self, original_workflow_component: WorkflowComponent):
        super().__init__(original_workflow_component)
        self.__name__ = original_workflow_component.name + "_resource"
        # Store resource requirements (don't create JobResource yet)
        self.component_name = self.__name__
        self.nnodes = original_workflow_component.nnodes
        self.nodes = original_workflow_component.nodes
        self.ppn = original_workflow_component.ppn
        self.num_gpus_per_process = getattr(original_workflow_component, 'num_gpus_per_process', 0)
        
        # Calculate total resources needed
        self.total_gpus = self.num_gpus_per_process * self.ppn
        
        # Resource allocation settings
        self.max_allocation_attempts = 1000  # Prevent infinite loops
        self.allocation_retry_delay = 0.5    # Seconds between attempts
        self.allocation_timeout = 300        # Total timeout in seconds
        self.cpu_affinity = []
        self.gpu_affinity = []
        if original_workflow_component.cpu_affinity is not None and len(original_workflow_component.cpu_affinity) > 0:
            self.cpu_affinity = copy.deepcopy(original_workflow_component.cpu_affinity)
        if original_workflow_component.gpu_affinity is not None and len(original_workflow_component.gpu_affinity) > 0:
            self.gpu_affinity = copy.deepcopy(original_workflow_component.gpu_affinity)

    def __call__(self, cluster_resource: ClusterResource, *results):
        self._init_logger()
        logger = self.logger
        logger.info(f"Requesting resources for component '{self.__name__}'")
        logger.debug(f"Resource requirements: {self.nnodes} nodes, {self.ppn} processes per node, {self.total_gpus} GPUs per node")
        
        if self.cpu_affinity:
            # Create JobResource at execution time (not init time)
            job_resource = JobResource(
                resources=[
                    NodeResourceList(cpus=self.cpu_affinity, gpus=self.gpu_affinity) 
                    for _ in range(self.nnodes)
                ],
                nodes=self.nodes
            )
        else:
            # Create JobResource at execution time (not init time)
            job_resource = JobResource(
                resources=[
                    NodeResourceCount(ncpus=self.ppn, ngpus=self.total_gpus) 
                    for _ in range(self.nnodes)
                ],
                nodes=self.nodes
            )
        
        # Try to allocate resources with timeout protection
        start_time = time.time()
        allocation_attempts = 0
        
        while allocation_attempts < self.max_allocation_attempts:
            allocation_attempts += 1
            
            # Check timeout
            elapsed_time = time.time() - start_time
            if elapsed_time > self.allocation_timeout:
                logger.error(f"Resource allocation timeout ({self.allocation_timeout}s) for '{self.__name__}'")
                raise TimeoutError(f"Could not allocate resources for '{self.__name__}' within {self.allocation_timeout} seconds")
            
            # Log progress periodically
            if allocation_attempts % 100 == 0:
                logger.debug(f"Resource allocation attempt #{allocation_attempts} for '{self.__name__}' (elapsed: {elapsed_time:.1f}s)")
            
            # Try allocation
            try:
                allocated, allocated_resource = cluster_resource.allocate(job_resource)
                if allocated:
                    logger.info(f"Successfully allocated resources for '{self.__name__}' after {allocation_attempts} attempts ({elapsed_time:.1f}s)")
                    logger.debug(f"Allocated resource: {allocated_resource}")
                    return allocated_resource
                    
            except Exception as e:
                logger.error(f"Error during resource allocation for '{self.__name__}': {e}")
                raise
            
            # Wait before retry
            time.sleep(self.allocation_retry_delay)
        
        # If we get here, we've exhausted all attempts
        logger.error(f"Resource allocation failed for '{self.__name__}' after {allocation_attempts} attempts")
        raise RuntimeError(f"Could not allocate resources for '{self.__name__}' after {allocation_attempts} attempts")

    def __repr__(self):
        return f"ResourceAwareCallable(component='{self.__name__}', nodes={self.nnodes}, ppn={self.ppn}, gpus={self.total_gpus})"


class MPICallable(Callable):
    """Serializable callable for MPI workflow components."""
    
    def __init__(self, workflow_component: WorkflowComponent):
        super().__init__(workflow_component)
        
        self.component_name = workflow_component.name
        self.nnodes = workflow_component.nnodes
        self.ppn = workflow_component.ppn
        self.num_gpus_per_process = workflow_component.num_gpus_per_process
        self.return_array = np.empty(workflow_component.return_dim) if len(workflow_component.return_dim) != 0 else None
        self.tmp_dir = os.path.join(os.getcwd(),".callable_tmp")
        self.gpu_selector = os.environ.get("SIMAIBENCH_GPUSELECTOR","ZE_AFFINITY_MASK")
        self.env = dict(os.environ)
        os.makedirs(self.tmp_dir,exist_ok=True)
        
        # Handle executable serialization
        if not isinstance(workflow_component.executable, str):
            try:
                # Get args for the callable
                component_args = getattr(workflow_component, 'args', None)

                if component_args:
                    bound_callable = partial(workflow_component.executable, **component_args)
                else:
                    bound_callable = workflow_component.executable

                # Serialize the callable
                serialized_func = cloudpickle.dumps(bound_callable)
                encoded_func = base64.b64encode(serialized_func).decode('ascii')

                # Create command that deserializes and executes the function
                self.cmd = (
                    f"python3 -c \""
                    f"import base64; import cloudpickle; "
                    f"func = cloudpickle.loads(base64.b64decode('{encoded_func}')); "
                    "func()\""
                )
                self.executable_type = 'callable'
                
            except Exception as e:
                raise ValueError(f"Cannot serialize executable for {workflow_component.name}: {e}")
        else:
            self.cmd = workflow_component.executable
            self.executable_type = 'string'
    
    def _buildcmd(self, job_resource: JobResource):
        """Function to build the mpi cmd from the job resources"""
        self._init_logger()
        logger = self.logger
        env = {}
        launcher_cmd = ""
        common_cpus = set.intersection(*[set(node_resource.cpus) for node_resource in job_resource.resources])
        use_common_cpus = common_cpus == set(job_resource.resources[0].cpus)
        if use_common_cpus:
            cores = ":".join(map(str, job_resource.resources[0].cpus))
            launcher_cmd += f"--cpu-bind list:{cores} "
        else:
            ##TODO: implement host file option
            logger.warning(f"Can't use same CPUs on all the nodes. Over subscribing cores")
            cores = ":".join(map(str, job_resource.resources[0].cpus))
            launcher_cmd += f"--cpu-bind list:{cores} "
        
        ##defaults to Aurora (Level zero)
        logger.info(f"Using {self.gpu_selector} for pinning GPUs")
        common_gpus = set.intersection(*[set(node_resource.gpus) for node_resource in job_resource.resources])
        use_common_gpus = common_gpus == set(job_resource.resources[0].gpus)
        if use_common_gpus:
            if self.nnodes == 1 and self.ppn == 1:
                env.update({"ZE_AFFINITY_MASK": ",".join([str(i) for i in job_resource.resources[0].gpus])})
            else:
                bash_script = gen_affinity_bash_script_1(self.num_gpus_per_process,self.gpu_selector)
                fname = os.path.join(self.tmp_dir,f"gpu_affinity_file_{self.component_name}.sh")
                if not os.path.exists(fname):
                    with open(fname, "w") as f:
                        f.write(bash_script)
                    st = os.stat(fname)
                    os.chmod(fname,st.st_mode | stat.S_IEXEC)
                launcher_cmd += f"{fname} "
                ##set environment variables
                env.update({"AVAILABLE_GPUS": ",".join([str(i) for i in job_resource.resources[0].gpus])})
        else:
            bash_script = gen_affinity_bash_script_2(self.num_gpus_per_process,self.gpu_selector)
            fname = os.path.join(self.tmp_dir,f"gpu_affinity_file_{self.component_name}.sh")
            if not os.path.exists(fname):
                with open(fname, "w") as f:
                    f.write(bash_script)
                st = os.stat(fname)
                os.chmod(fname,st.st_mode | stat.S_IEXEC)
            launcher_cmd += f"{fname} "
            ##Here you need to set the environment variables for each node
            for nid,node in enumerate(job_resource.nodes):
                env.update({f"AVAILABLE_GPUS_{node}": ",".join([str(i) for i in job_resource.resources[nid].gpus])})

        return launcher_cmd, env

    def __call__(self, cluster_resource: ClusterResource, *results):
        # Create fresh logger in worker process
        self._init_logger()
        logger = self.logger
        logger.info(f"Executing MPI component '{self.component_name}'")
        
        # Extract job_resource from results
        job_resource = results[0] if results else None
        if job_resource is None:
            logger.error(f"No job resource provided for MPI component '{self.component_name}'")
            raise ValueError("No job resource provided from resource allocation step")
        
        logger.debug(f"Using job resource with nodes: {job_resource.nodes} cpus: {job_resource.resources[0].cpus} gpus:{job_resource.resources[0].gpus}")
        logger.debug(f"Executable type: {self.executable_type}")
        
        laucher_opts,env = self._buildcmd(job_resource)

        try:
            # Construct MPI command
            nodes_str = ",".join(job_resource.nodes)
            if self.nnodes >= 1:
                full_cmd = f"mpirun -np {self.ppn * self.nnodes} -ppn {self.ppn} --hosts {nodes_str} {laucher_opts} {self.cmd}"
            else:
                full_cmd = f"mpirun -np {self.ppn * self.nnodes} {laucher_opts} {self.cmd}"
            
            logger.info(f"Executing MPI command for '{self.component_name}'")
            logger.info(f"Full commad: {full_cmd}")
            
            # Execute MPI command
            result = subprocess.run(full_cmd, shell=True, capture_output=True, text=True, env=self.env.update(env))
            
            logger.info(f"MPI execution completed for '{self.component_name}' with return code: {result.returncode}")
            
            if result.returncode != 0:
                logger.error(f"MPI execution failed for '{self.component_name}': {result.stderr}")
                raise
            else:
                logger.debug(f"MPI execution output for '{self.component_name}': {result.stdout[:200]}...")
            
        except Exception as e:
            logger.error(f"Exception during MPI execution for '{self.component_name}': {str(e)}")
            raise
        finally:
            # Always deallocate resources
            try:
                cluster_resource.deallocate(job_resource)
                logger.info(f"Deallocated resources for '{self.component_name}'")
            except Exception as e:
                logger.error(f"Failed to deallocate resources for '{self.component_name}': {e}")
        
        return self.return_array

    def __repr__(self):
        return f"MPICallable(component='{self.component_name}', nodes={self.nnodes}, ppn={self.ppn}, type='{self.executable_type}')"
    
if __name__ == "__main__":
    ##all cpus same
    job_resource = JobResource(resources=[NodeResourceList(cpus=(1,2,3,4),gpus=(1,2,3,4))],nodes=["node1"])
    callmpi = MPICallable(WorkflowComponent("hello",executable=lambda x: x**2, type="local",ppn=4,num_gpus_per_process=1))
    opts,env = callmpi._buildcmd(job_resource)
    print(opts,env)

    ##
    job_resource = JobResource(resources=[NodeResourceList(cpus=(1,2,3,4),gpus=(1,2,3,4)),
                                          NodeResourceList(cpus=(1,2,3,4),gpus=(1,2,3,4))],nodes=["node1",
                                                                                                  "node2"])
    callmpi = MPICallable(WorkflowComponent("hello_1",executable=lambda x: x**2, type="local",ppn=4,nnodes=2,num_gpus_per_process=1))
    opts,env = callmpi._buildcmd(job_resource)
    print(opts,env)

    ##
    job_resource = JobResource(resources=[NodeResourceList(cpus=(1,2,3,4),gpus=(1,2,3,4)),
                                          NodeResourceList(cpus=(1,2,3,4),gpus=(5,6,7,8))],nodes=["node1",
                                                                                                  "node2"])
    callmpi = MPICallable(WorkflowComponent("hello_2",executable=lambda x: x**2, type="local",ppn=4,nnodes=2,num_gpus_per_process=1))
    opts,env = callmpi._buildcmd(job_resource)
    print(opts,env)
