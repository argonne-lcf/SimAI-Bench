

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

class DragonExecutor:
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