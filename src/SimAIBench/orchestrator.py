import os
import multiprocessing
import subprocess
import networkx as nx
import time

from typing import List, Union, Dict, Any
from SimAIBench import WorkflowComponent
from SimAIBench.resources import ClusterResource, JobResource, NodeResourceCount, NodeResourceList

def wrap_with_mpirun(workflow_component: WorkflowComponent, 
                     sys_info: NodeResourceList, 
                     cluster_resource: ClusterResource):
    import subprocess
    import base64
    import cloudpickle

    nnodes = workflow_component.nnodes
    ppn = workflow_component.ppn
    ngpus = workflow_component.num_gpus_per_process*ppn
    job_resource = JobResource(resources=[NodeResourceCount(ncpus=ppn,ngpus=ngpus) for i in range(nnodes)])

    while True:
        allocated,allocated_resource = cluster_resource.allocate(job_resource)
        if allocated:
            break
        time.sleep(0.1)

    if not isinstance(workflow_component.executable,str):
        # Get args for the callable
        component_args = getattr(workflow_component, 'args', None)
        serialized_func = cloudpickle.dumps(workflow_component.executable)
        
        encoded_func = base64.b64encode(serialized_func).decode('ascii')
        
        # Serialize args if present
        encoded_args = ""
        if component_args:
            serialized_args = cloudpickle.dumps(component_args)
            encoded_args = base64.b64encode(serialized_args).decode('ascii')
        
        # Create command that deserializes and executes the function with args
        cmd = (
            f"python3 -c \""
            f"import base64; import cloudpickle; "
            f"func = cloudpickle.loads(base64.b64decode('{encoded_func}')); "
        )
        
        if encoded_args:
            cmd += f"args = cloudpickle.loads(base64.b64decode('{encoded_args}')); "
            # Args will always be a dict, so use **args for keyword arguments
            cmd += "func(**args)\""
        else:
            cmd += "func()\""
    else:
        cmd = workflow_component.executable
    
    nnodes = len(workflow_component.nodes)
    ppn = workflow_component.ppn
    nodes_str = ",".join(workflow_component.nodes)
    cmd = f"mpirun -np {ppn*nnodes} -ppn {ppn} --hosts {nodes_str} {cmd}"

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    cluster_resource.deallocate(job_resource)

    return result.stdout

class DAG:
    """
        Class to build implicit DAGs from explitic DAGs give by the user.
        This is to make sure to be compatible with TAPS
    """
    def __init__(self, workflow_components: Dict[str, WorkflowComponent]):
        self._graph = self._build_dag(workflow_components=workflow_components)
        

    def _build_dag(self,workflow_components: Dict[str, WorkflowComponent]):
        graph = nx.DiGraph()
        ##add all the nodes
        for c,wc in workflow_components.items():
            graph.add_node(c,component=wc,status="not_ready")
            if wc.nnodes > 1:
                graph.add_node(c+"_resource",component=DAG._get_resource_aware_node(wc),status="not_ready")
        
        ##add all the edges
        for c,wc in workflow_components.items():
            if wc.nnodes > 1:
                for d in workflow_components[c].dependencies:
                    graph.add_edge(d,c+"_resource")
                graph.add_edge(c+"_resource",c)
            else:
                for d in workflow_components[c].dependencies:
                    graph.add_edge(d,c)
        
        # Set nodes with no dependencies as ready
        for node in graph.nodes():
            if graph.in_degree(node) == 0:
                graph.nodes[node]['status'] = 'ready'
        
        return graph

    def update(self,workflow_component: WorkflowComponent):
        """Update the dag with a new workflow component"""
        # Check if all dependencies exist in the graph
        for dep in workflow_component.dependencies:
            if dep not in self._graph.nodes():
                raise ValueError(f"Dependency '{dep}' for component '{workflow_component.name}' does not exist in the graph")
            
        # Add the new node to the graph
        self._graph.add_node(workflow_component.name, component=workflow_component, status="not_ready")
        
        # If it's a multi-node component, add resource node
        if workflow_component.nnodes > 1:
            self._graph.add_node(workflow_component.name + "_resource", 
                               component=DAG._get_resource_aware_node(workflow_component), 
                               status="not_ready")
        
        # Add edges for dependencies
        if workflow_component.nnodes > 1:
            for dep in workflow_component.dependencies:
                self._graph.add_edge(dep, workflow_component.name + "_resource")
            self._graph.add_edge(workflow_component.name + "_resource", workflow_component.name)
        else:
            for dep in workflow_component.dependencies:
                self._graph.add_edge(dep, workflow_component.name)
        
        # Set status to ready if no dependencies
        if self._graph.in_degree(workflow_component.name) == 0:
            self._graph.nodes[workflow_component.name]['status'] = 'ready'
        if workflow_component.nnodes > 1 and self._graph.in_degree(workflow_component.name + "_resource") == 0:
            self._graph.nodes[workflow_component.name + "_resource"]["status"] = 'ready'
        
    @classmethod
    def _get_resource_aware_node(cls, workflow_component: WorkflowComponent):
        """
        This function takes in the workflow components object and 
        create a node that blocks until resources required by the workflow components are available
        """
        def resource_node(cluster_resource: ClusterResource):
            from SimAIBench.resources import JobResource, NodeResourceCount
            import time
            
            nnodes = workflow_component.nnodes
            ppn = workflow_component.ppn
            ngpus = workflow_component.num_gpus_per_process * workflow_component.ppn
            
            job_resource = JobResource(resources=[NodeResourceCount(ncpus=ppn, ngpus=ngpus) for i in range(nnodes)])
            
            while True:
                allocated, allocated_resource = cluster_resource.allocate(job_resource)
                if allocated:
                    break
                time.sleep(0.1)
            
            # Return the allocated resource for potential use by downstream tasks
            # Note: The caller is responsible for deallocation
            return allocated_resource
        
        wc = WorkflowComponent(
            workflow_component.name+"_resource",
            resource_node,
            "local"
        )
        
        return wc

class Orchestrator:
    """
    """

    def __init__(self, workflow_components: Dict[str, WorkflowComponent], sys_info: dict = {"name": "local"}):
        """
        Initialize the BasicLauncher.
        
        Args:
            system: System name (e.g., "local", "aurora", "polaris")
            launcher_config: Configuration for the launcher
        """
        self.sys_info = sys_info
        self.dag = DAG(workflow_components)
        self.executor = None
    
    def launch(self):
        pass

    def update(self,workflow_component: WorkflowComponent):
        """Update the dag"""
        self.dag.update(workflow_component)

    # @staticmethod
    # def _prepare_environment(base_env: Dict[str, str] = None, 
    #                        additional_env: Dict[str, str] = None) -> Dict[str, str]:
    #     """
    #     Prepare environment variables for process execution.
        
    #     Args:
    #         base_env: Base environment (defaults to os.environ)
    #         additional_env: Additional environment variables
        
    #     Returns:
    #         Environment dictionary
    #     """
    #     env = (base_env or os.environ).copy()
        
    #     if additional_env:
    #         env.update(additional_env)
            
    #     return env

    # def launch_component(self, workflow_component) -> Union[multiprocessing.Process, Any]:
    #     """
    #     Launch a single workflow component using ensemble_launcher or Dragon.
        
    #     Args:
    #         workflow_component: WorkflowComponent object to launch
            
    #     Returns:
    #         multiprocessing.Process object for ensemble launcher or ProcessGroup for Dragon
    #     """
    #     component_type = getattr(workflow_component, 'type', 'ensemble')
        
    #     if component_type == "dragon":
    #         return self._launch_dragon_component(workflow_component)
    #     elif component_type == "local":
    #         return self._launch_local_component(workflow_component)
    #     elif component_type == "remote":
    #         # Use ensemble launcher for all other types (local, remote, ensemble)
    #         return self._launch_component_with_ensemble(workflow_component)
    #     else:
    #         if TAPS_AVAILABLE:
    #             available_executors = get_executor_configs()
    #             if component_type in available_executors:
    #                 self.executor_config = available_executors[component_type]()
    #                 if component_type == "dask":
    #                     self.executor_config.workers = 1
    #                 self.executor = self.executor_config.get_executor()
    #                 if workflow_component.ppn * (max(len(workflow_component.nodes),1)) > 1:
    #                     return self.executor.submit(wrap_with_mpirun,workflow_component)
    #                 else:
    #                     return self.executor.submit(workflow_component.executable)

    #         raise ValueError(f"Unknown component type: {component_type}. Expected 'local', 'remote', 'dragon', or 'ensemble'.")
    
    # def wait_for_component(self, launched_process, timeout: int = None) -> Union[int, List[int]]:
    #     """
    #     Wait for a launched workflow component to complete.
        
    #     Args:
    #         launched_process: The process or process group to wait for
    #         timeout: Timeout in seconds (None for no timeout)
            
    #     Returns:
    #         Exit code or list of exit codes
    #     """
    #     if isinstance(launched_process, subprocess.Popen):
    #         # This is a subprocess.Popen from local string executable
    #         try:
    #             launched_process.wait(timeout=timeout)
    #             return launched_process.returncode if launched_process.returncode is not None else 0
    #         except subprocess.TimeoutExpired:
    #             # If wait times out, terminate the process
    #             launched_process.terminate()
    #             try:
    #                 launched_process.wait(timeout=5)  # Give 5 seconds for graceful termination
    #             except subprocess.TimeoutExpired:
    #                 launched_process.kill()  # Force kill if still alive
    #             return launched_process.returncode if launched_process.returncode is not None else 124
    #     elif isinstance(launched_process, multiprocessing.Process):
    #         # This is a multiprocessing.Process from ensemble launcher or callable
    #         try:
    #             launched_process.join(timeout=timeout)
    #             return launched_process.exitcode if launched_process.exitcode is not None else 0
    #         except Exception as e:
    #             # If join times out or fails, terminate the process
    #             if launched_process.is_alive():
    #                 launched_process.terminate()
    #                 launched_process.join(1)  # Give it 1 second to terminate gracefully
    #                 if launched_process.is_alive():
    #                     launched_process.kill()  # Force kill if still alive
    #             return launched_process.exitcode if launched_process.exitcode is not None else 1
    #     elif DRAGON_AVAILABLE and ProcessGroup is not None and isinstance(launched_process, ProcessGroup):
    #         try:
    #             launched_process.join(timeout)
    #             exit_code = 1 if any(p[1]!=0 for p in launched_process.inactive_puids) else 0
    #             launched_process.stop()  # Stop the process group after waiting
    #         except Exception as e:
    #             exit_code = 1
    #             launched_process.stop()
    #         return exit_code
    #     else:
    #         try:
    #             result = launched_process.result()
    #             print(f"Task returned {result}")
    #             self.executor.shutdown()
    #             return 0
    #         except:
    #             raise ValueError(f"Unknown launched process type: {type(launched_process)}. Expected subprocess.Popen, multiprocessing.Process, or Dragon ProcessGroup.")