# Try to import ensemble_launcher components
try:
    from ensemble_launcher.helper_functions import create_task_info
    from ensemble_launcher.worker import worker
    ENSEMBLE_LAUNCHER_AVAILABLE = True
except ImportError:
    raise ImportError("ensemble_launcher is required but not available. Please install ensemble_launcher to use this launcher.")


class EnsembleExecutor:
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