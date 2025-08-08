import os
import multiprocessing
import socket
import subprocess
import shlex
from typing import List, Union, Dict, Any, Callable, Tuple, Optional

# Handle Dragon imports and type hints
try:
    import dragon
    from dragon.native.process_group import ProcessGroup
    from dragon.native.process import ProcessTemplate, Process as DragonProcess, Popen
    from dragon.infrastructure.policy import Policy
    DRAGON_AVAILABLE = True
except ImportError:
    DRAGON_AVAILABLE = False
    # When Dragon is not available, we'll use Any for type hints
    ProcessGroup = Any

# Try to import ensemble_launcher components
try:
    from ensemble_launcher.helper_functions import create_task_info
    from ensemble_launcher.worker import worker
    ENSEMBLE_LAUNCHER_AVAILABLE = True
except ImportError:
    raise ImportError("ensemble_launcher is required but not available. Please install ensemble_launcher to use this launcher.")


class BasicLauncher:
    """
    Ensemble launcher-based workflow component launcher.
    
    This launcher uses ensemble_launcher as the primary execution engine for all
    workflow components. It provides unified execution for local and remote tasks
    with advanced resource management and system-specific optimizations.
    
    Requires ensemble_launcher to be installed.
    """

    def __init__(self, sys_info: dict = {"name": "local"}, launcher_config: Dict[str, Any] = None):
        """
        Initialize the BasicLauncher.
        
        Args:
            system: System name (e.g., "local", "aurora", "polaris")
            launcher_config: Configuration for the launcher
        """
        self.sys_info = sys_info
        self.launcher_config = launcher_config or {"mode": "mpi"}
        
    @staticmethod
    def _prepare_environment(base_env: Dict[str, str] = None, 
                           additional_env: Dict[str, str] = None) -> Dict[str, str]:
        """
        Prepare environment variables for process execution.
        
        Args:
            base_env: Base environment (defaults to os.environ)
            additional_env: Additional environment variables
        
        Returns:
            Environment dictionary
        """
        env = (base_env or os.environ).copy()
        
        if additional_env:
            env.update(additional_env)
            
        return env
    
    def _workflow_component_to_task_info(self, workflow_component, task_id: str = None) -> Dict[str, Any]:
        """
        Convert a workflow component to ensemble_launcher task_info format.
        
        Args:
            workflow_component: WorkflowComponent object
            task_id: Optional task ID (defaults to component name)
            
        Returns:
            Dictionary in task_info format
        """
        if task_id is None:
            task_id = getattr(workflow_component, 'name', f"task_{id(workflow_component)}")
        
        # Handle both string executables and Python callables
        if isinstance(workflow_component.executable, str):
            cmd_template = workflow_component.executable
            
            # Add args if provided - format as "key value" pairs
            args = getattr(workflow_component, 'args', None)
            if args:
                # Convert dict to "key value" format
                for key, value in args.items():
                    cmd_template += f" {key} {value}"
        else:
            # For Python callables, we need to create a shell command that can execute them
            # since ensemble_launcher uses cmd_template for actual task execution
            import base64
            import pickle
            
            # Get args for the callable
            component_args = getattr(workflow_component, 'args', None)
            
            # Try to use cloudpickle for better serialization, fallback to pickle
            try:
                import cloudpickle
                serialized_func = cloudpickle.dumps(workflow_component.executable)
                pickle_module = "cloudpickle"
            except ImportError:
                serialized_func = pickle.dumps(workflow_component.executable)
                pickle_module = "pickle"
            
            try:
                encoded_func = base64.b64encode(serialized_func).decode('ascii')
                
                # Serialize args if present
                encoded_args = ""
                if component_args:
                    try:
                        if pickle_module == "cloudpickle":
                            serialized_args = cloudpickle.dumps(component_args)
                        else:
                            serialized_args = pickle.dumps(component_args)
                        encoded_args = base64.b64encode(serialized_args).decode('ascii')
                    except:
                        encoded_args = ""
                
                # Create command that deserializes and executes the function with args
                cmd_template = (
                    f"python3 -c \""
                    f"import base64; import {pickle_module}; "
                    f"func = {pickle_module}.loads(base64.b64decode('{encoded_func}')); "
                )
                
                if encoded_args:
                    cmd_template += f"args = {pickle_module}.loads(base64.b64decode('{encoded_args}')); "
                    # Args will always be a dict, so use **args for keyword arguments
                    cmd_template += "func(**args)\""
                else:
                    cmd_template += "func()\""
                    
            except Exception:
                # Final fallback - assume it's a simple function call
                func_name = getattr(workflow_component.executable, '__name__', 'unknown_function')
                module_name = getattr(workflow_component.executable, '__module__', '__main__')
                if module_name == '__main__':
                    # If function is from __main__, we can't import it this way
                    cmd_template = f"python3 -c \"print('Error: Cannot execute function {func_name} from __main__ module')\""
                else:
                    cmd_template = f"python3 -c \"from {module_name} import {func_name}; {func_name}()\""
        
        # Calculate nodes list
        nodes = workflow_component.nodes if hasattr(workflow_component, 'nodes') and workflow_component.nodes else [socket.gethostname()]
        num_nodes = len(nodes)
        
        # Get processes per node
        ppn = getattr(workflow_component, 'ppn', 1)
        
        # Get GPU information
        num_gpus_per_process = getattr(workflow_component, 'num_gpus_per_process', 0)
        gpu_affinity = None
        if hasattr(workflow_component, 'gpu_affinity') and workflow_component.gpu_affinity:
            gpu_affinity = workflow_component.gpu_affinity
        
        # Get CPU affinity
        cpu_affinity = getattr(workflow_component, 'cpu_affinity', None)
        
        # Get environment variables
        env_vars = getattr(workflow_component, 'env_vars', {})
        
        run_dir = getattr(workflow_component, 'run_dir', os.path.join(os.getcwd(), "run_dir"))
        io = False
        log_file = getattr(workflow_component, 'log_file', os.path.join(run_dir, f"{getattr(workflow_component, 'name')}.log"))
        err_file = getattr(workflow_component, 'err_file', os.path.join(run_dir, f"{getattr(workflow_component, 'name')}.err"))
        # Create task_info using the helper function
        task_info = create_task_info(
            task_id=task_id,
            cmd_template=cmd_template,
            system=self.sys_info["name"],
            num_nodes=num_nodes,
            num_processes_per_node=ppn,
            num_gpus_per_process=num_gpus_per_process,
            gpu_affinity=gpu_affinity,
            cpu_affinity=cpu_affinity,
            env=env_vars,
            io=io,
            log_file=log_file,
            err_file=err_file,
            run_dir=run_dir,
            timeout=getattr(workflow_component, 'timeout', None)
        )
        
        # Add assigned nodes for the worker
        task_info['assigned_nodes'] = nodes
        task_info['assigned_cores'] = {node: list(range(ppn)) for node in nodes}
        task_info['assigned_gpus'] = {node: gpu_affinity[:num_gpus_per_process] if gpu_affinity else [] for node in nodes}
        
        return task_info
    
    def _launch_component_with_ensemble(self, workflow_component) -> multiprocessing.Process:
        """
        Launch a single component using ensemble_launcher worker in a multiprocessing.Process.
        Handles both local and remote execution types through ensemble launcher.
        
        Args:
            workflow_component: WorkflowComponent object to launch
            
        Returns:
            multiprocessing.Process object representing the launched worker process
        """
        # Convert workflow component to task_info
        task_id = getattr(workflow_component, 'name', f"task_{id(workflow_component)}")
        task_info = self._workflow_component_to_task_info(workflow_component, task_id)
        
        # Create tasks dictionary
        my_tasks = {task_id: task_info}

        # Get nodes list
        nodes = workflow_component.nodes if hasattr(workflow_component, 'nodes') and workflow_component.nodes else [socket.gethostname()]
        
        if self.sys_info["name"] == "local":
            self.sys_info["ncores_per_node"] = getattr(workflow_component, 'ncores_per_node', multiprocessing.cpu_count())
            self.sys_info["ngpus_per_node"] = getattr(workflow_component, 'ngpus_per_node', 0)
        
        # Create worker instance
        worker_id = f"worker_{task_id}"
        comm_config = {"comm_layer": "multiprocessing"}
        
        worker_instance = worker(
            worker_id=worker_id,
            my_tasks=my_tasks,
            my_nodes=nodes,
            sys_info=self.sys_info,
            comm_config=comm_config,
            launcher_config=self.launcher_config
        )
        
        # Create and start the multiprocessing.Process directly with worker.run_tasks
        process = multiprocessing.Process(
            target=worker_instance.run_tasks,
            kwargs={'logger': False},
            name=f"worker_{task_id}"
        )
        
        # Store reference to task info for later retrieval
        process.task_id = task_id
        process.worker_instance = worker_instance
        
        # Start the process
        process.start()
        
        return process
    
    @staticmethod
    def _launch_dragon_component(workflow_component,size_kwarg="size",rank_kwarg="rank",mpi_kwarg="init_MPI") -> Any:
        """
        Launch a single component using Dragon.
        
        Args:
            workflow_component: WorkflowComponent object to launch
            size_kwarg: Keyword argument for total processes (default: "size")
            rank_kwarg: Keyword argument for process rank (default: "rank")
            mpi_kwarg: Keyword argument for MPI initialization (default: "init_MPI")
            
        Returns:
            Launched process group
        """
        if not DRAGON_AVAILABLE:
            raise RuntimeError("Dragon is not available")
        
        if not callable(workflow_component.executable):
            raise ValueError("Dragon launcher requires Python callable, not string executable")
        
        # Calculate total processes
        total_processes = workflow_component.ppn
        if workflow_component.nodes:
            total_processes = workflow_component.ppn * len(workflow_component.nodes)
        
        # Create process group
        policy = Policy(distribution=Policy.Distribution.BLOCK)
        pg = ProcessGroup(restart=False,policy=policy)
        
        # Add processes based on nodes and ppn
        process_count = 0
        for nid, node in enumerate(workflow_component.nodes or ["localhost"]):
            for local_rank in range(workflow_component.ppn):
                if process_count >= total_processes:
                    break
                
                # Create placement policy
                placement_policy = Policy(
                    placement=Policy.Placement.HOST_NAME,
                    host_name=node
                )
                

                # Prepare environment
                env = BasicLauncher._prepare_environment(
                    additional_env=workflow_component.env_vars
                )

                # Set CPU affinity
                if workflow_component.cpu_affinity and local_rank < len(workflow_component.cpu_affinity):
                    placement_policy.cpu_affinity = [workflow_component.cpu_affinity[local_rank]]
                
                # Set GPU affinity
                if workflow_component.gpu_affinity and local_rank < len(workflow_component.gpu_affinity):
                    placement_policy.gpu_affinity = [workflow_component.gpu_affinity[local_rank]]
                    env["ZE_AFFINITY_MASK"] = workflow_component.gpu_affinity[local_rank]
                    env["CUDA_VISIBLE_DEVICES"] = workflow_component.gpu_affinity[local_rank]
                
                
                # Create process template with args
                template_args = []

                # Add workflow component args
                component_args = getattr(workflow_component, 'args', None)
                if component_args:
                    # Convert dict to tuple of values
                    template_args.extend(component_args.values())
                
                pg.add_process(
                    nproc=1,
                    template=ProcessTemplate(
                        target=workflow_component.executable,
                        args=tuple(template_args),
                        kwargs={size_kwarg: total_processes, rank_kwarg: process_count, mpi_kwarg: False},
                        cwd=os.getcwd(),
                        policy=placement_policy,
                        stdout=Popen.DEVNULL,
                        env=env
                    )
                )
                
                process_count += 1
            
            if process_count >= total_processes:
                break
        
        print(f"Launching Dragon process group with {process_count} processes on {len(workflow_component.nodes)} nodes")
        pg.init()
        pg.start()
        
        return pg
    
    def _launch_local_component(self, workflow_component) -> Union[subprocess.Popen, multiprocessing.Process]:
        """
        Launch a single component locally.
        
        Args:
            workflow_component: WorkflowComponent object to launch
            
        Returns:
            subprocess.Popen for string executables or multiprocessing.Process for callables
        """
        if isinstance(workflow_component.executable, str):
            # If executable is a string, use subprocess.Popen directly
            env = self._prepare_environment(additional_env=getattr(workflow_component, 'env_vars', {}))
            cmd_args = shlex.split(workflow_component.executable)
            
            # Add args if provided - format as "key value" pairs
            args = getattr(workflow_component, 'args', None)
            if args:
                # Convert dict to "key value" format
                for key, value in args.items():
                    cmd_args.extend([str(key), str(value)])
            
            process = subprocess.Popen(
                cmd_args,
                env=env,
                cwd=getattr(workflow_component, 'run_dir', None),
                stdout=None,
                stderr=None,
            )
            return process
            
        elif callable(workflow_component.executable):
            # If executable is a callable, run it in multiprocessing.Process
            args = getattr(workflow_component, 'args', None)
            
            process = multiprocessing.Process(
                target=workflow_component.executable,
                args=tuple(args.values()) if args is not None else (),
                name=getattr(workflow_component, 'name', f"local_{id(workflow_component)}")
            )
            process.start()
            return process
        else:
            raise ValueError(f"Unsupported executable type: {type(workflow_component.executable)}. Expected str or callable.")

    def launch_component(self, workflow_component) -> Union[multiprocessing.Process, Any]:
        """
        Launch a single workflow component using ensemble_launcher or Dragon.
        
        Args:
            workflow_component: WorkflowComponent object to launch
            
        Returns:
            multiprocessing.Process object for ensemble launcher or ProcessGroup for Dragon
        """
        component_type = getattr(workflow_component, 'type', 'ensemble')
        
        if component_type == "dragon":
            return self._launch_dragon_component(workflow_component)
        elif component_type == "local":
            return self._launch_local_component(workflow_component)
        elif component_type == "remote":
            # Use ensemble launcher for all other types (local, remote, ensemble)
            return self._launch_component_with_ensemble(workflow_component)
        else:
            raise ValueError(f"Unknown component type: {component_type}. Expected 'local', 'remote', 'dragon', or 'ensemble'.")
    
    def wait_for_component(self, launched_process, timeout: int = None) -> Union[int, List[int]]:
        """
        Wait for a launched workflow component to complete.
        
        Args:
            launched_process: The process or process group to wait for
            timeout: Timeout in seconds (None for no timeout)
            
        Returns:
            Exit code or list of exit codes
        """
        if isinstance(launched_process, subprocess.Popen):
            # This is a subprocess.Popen from local string executable
            try:
                launched_process.wait(timeout=timeout)
                return launched_process.returncode if launched_process.returncode is not None else 0
            except subprocess.TimeoutExpired:
                # If wait times out, terminate the process
                launched_process.terminate()
                try:
                    launched_process.wait(timeout=5)  # Give 5 seconds for graceful termination
                except subprocess.TimeoutExpired:
                    launched_process.kill()  # Force kill if still alive
                return launched_process.returncode if launched_process.returncode is not None else 124
        elif isinstance(launched_process, multiprocessing.Process):
            # This is a multiprocessing.Process from ensemble launcher or callable
            try:
                launched_process.join(timeout=timeout)
                return launched_process.exitcode if launched_process.exitcode is not None else 0
            except Exception as e:
                # If join times out or fails, terminate the process
                if launched_process.is_alive():
                    launched_process.terminate()
                    launched_process.join(1)  # Give it 1 second to terminate gracefully
                    if launched_process.is_alive():
                        launched_process.kill()  # Force kill if still alive
                return launched_process.exitcode if launched_process.exitcode is not None else 1
        elif DRAGON_AVAILABLE and ProcessGroup is not None and isinstance(launched_process, ProcessGroup):
            try:
                launched_process.join(timeout)
                exit_code = 1 if any(p[1]!=0 for p in launched_process.inactive_puids) else 0
                launched_process.stop()  # Stop the process group after waiting
            except Exception as e:
                exit_code = 1
                launched_process.stop()
            return exit_code
        else:
            raise ValueError(f"Unknown launched process type: {type(launched_process)}. Expected subprocess.Popen, multiprocessing.Process, or Dragon ProcessGroup.")