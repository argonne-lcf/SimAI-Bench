import subprocess
import os
import sys
import time
import argparse
import json
from typing import Dict, List, Optional, Union, Callable, Any
from dataclasses import dataclass, field
from SimAIBench.orchestrator import Orchestrator
import networkx as nx
from typing import Union, List, Dict, Any
from SimAIBench.component import WorkflowComponent
from concurrent.futures import Future
from SimAIBench.config import OchestratorConfig, SystemConfig


class Workflow:
    """
    Workflow orchestration class responsible for registering components and 
    submitting to launcher for execution.
    """
    
    def __init__(self,orchestrator_config: Orchestrator=OchestratorConfig(), 
                 system_config: SystemConfig = SystemConfig(name="local"),
                 **config_files):
        """
        Initialize workflow.
        
        Args:
            **config_files: Named configuration files (e.g., ai_config="path.json")
        """
        # Store configuration files and loaded configs
        self.config_files = {key: value for key, value in config_files.items() if isinstance(value, str)}
        self.configs = {key: value for key, value in config_files.items() if isinstance(value, dict)}
        
        # Load configurations if provided
        for config_name, config_path in self.config_files.items():
            if config_path:
                with open(config_path, 'r') as f:
                    self.configs[config_name] = json.load(f)
        
        # Registered workflow components
        self.components: Dict[str, WorkflowComponent] = {}
        
        # orchestrator instance
        self.orchestrator = None
        self.orchestrator_config = orchestrator_config
        self.sys_config = system_config

    def register_component(self, name: str, 
                          executable: Union[str, Callable], 
                          type: str,
                          args: Dict[str, Any] = None,
                          nodes: List[str] = None,
                          ppn: int = 1,
                          num_gpus_per_process: int = 0,
                          cpu_affinity: List[int] = None,
                          gpu_affinity: List[str] = None, 
                          env_vars: Dict[str, str] = None,
                          dependencies: List[str] = None) -> 'Workflow':
        """
        Register a component in the workflow.
        
        Args:
            name: Unique name for this component
            executable: Command string or Python function to execute
            type: Component type ("local", "remote", "dragon")
            args: Arguments dictionary for the component
            nodes: List of nodes to run on (optional)
            ppn: Processes per node (optional)
            num_gpus_per_process: Number of GPUs per process (optional)
            cpu_affinity: CPU cores to bind to (optional)
            gpu_affinity: GPU devices to bind to (optional)
            env_vars: Environment variables (optional)
            dependencies: List of component names this depends on (optional)
            
        Returns:
            Self for method chaining
        """
        component = WorkflowComponent(
            name=name,
            type=type,
            executable=executable,
            args=args or {},
            nodes=nodes or [],
            ppn=ppn,
            num_gpus_per_process=num_gpus_per_process,
            cpu_affinity=cpu_affinity,
            gpu_affinity=gpu_affinity,
            env_vars=env_vars or {},
            dependencies=dependencies or []
        )
        
        self.components[name] = component
        return self

    def get_component(self, name: str) -> Optional[WorkflowComponent]:
        """Get a registered component by name."""
        return self.components.get(name)
    
    def list_components(self) -> List[str]:
        """List all registered component names."""
        return list(self.components.keys())

    def component(self, func: Callable = None, *, 
                 name: str = None,
                 type: str = "local",
                 args: Dict[str, Any] = None,
                 nodes: List[str] = None,
                 ppn: int = 1,
                 num_gpus_per_process: int = 0,
                 cpu_affinity: List[int] = None,
                 gpu_affinity: List[str] = None, 
                 env_vars: Dict[str, str] = None,
                 dependencies: List[str] = None):
        """
        Decorator to register a component in the workflow.
        
        Can be used with or without parentheses:
        - @workflow.component
        - @workflow.component()
        - @workflow.component(name="my_task", dependencies=["setup"])
        
        Args:
            func: Function to register (when used without parentheses)
            name: Unique name for this component (defaults to function name)
            type: Component type ("local", "remote", "dragon")
            args: Arguments dictionary for the component
            nodes: List of nodes to run on (optional)
            ppn: Processes per node (optional)
            num_gpus_per_process: Number of GPUs per process (optional)
            cpu_affinity: CPU cores to bind to (optional)
            gpu_affinity: GPU devices to bind to (optional)
            env_vars: Environment variables (optional)
            dependencies: List of component names this depends on (optional)
            
        Returns:
            Decorated function or decorator function
            
        Examples:
            @workflow.component
            def my_function():
                return 0
                
            @workflow.component()
            def another_function():
                return 0
                
            @workflow.component(name="my_task", args={"--input": "file.txt"}, dependencies=["setup"])
            def third_function():
                return 0
        """
        def decorator(f: Callable):
            component_name = name if name is not None else f.__name__
            self.register_component(
                name=component_name,
                type=type,
                executable=f,
                args=args,
                nodes=nodes,
                ppn=ppn,
                num_gpus_per_process=num_gpus_per_process,
                cpu_affinity=cpu_affinity,
                gpu_affinity=gpu_affinity,
                env_vars=env_vars,
                dependencies=dependencies
            )
            return f
        
        # If func is provided, this was called without parentheses: @workflow.component
        if func is not None:
            return decorator(func)
        
        # Otherwise, this was called with parentheses: @workflow.component() or @workflow.component(args)
        return decorator
    
    def launch(self, **kwargs) -> Future:
        """
        Execute the complete workflow by launching all registered components
        in dependency order.
        
        Args:
            **kwargs: Additional arguments passed to component handlers
        
        Returns:
            0 for success, 1 for failure
        """
        if not self.components:
            print("No components registered in workflow")
            return 0
            
        self.orchestrator = Orchestrator(self.components, sys_info=self.sys_config, config=self.orchestrator_config)
        future = self.orchestrator.launch()
        return future