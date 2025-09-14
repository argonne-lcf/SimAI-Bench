
import subprocess


class LocalExecutor:
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