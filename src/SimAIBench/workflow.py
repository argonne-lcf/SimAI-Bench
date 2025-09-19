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

    def get_component(self, name: str) -> Optional[WorkflowComponent]:
        """Get a registered component by name."""
        return self.components.get(name)
    
    def list_components(self) -> List[str]:
        """List all registered component names."""
        return list(self.components.keys())

    def component(self, func: Callable = None, **kwargs):
        """
        Decorator to register a component in the workflow.
        
        Args:
            func: Function to register (when used without parentheses)
            **kwargs: All WorkflowComponent parameters (name, type, args, nodes, etc.)
        """
        def decorator(f: Callable):
            # Use function name as default if name not provided
            component_kwargs = kwargs.copy()
            if 'name' not in component_kwargs:
                component_kwargs['name'] = f.__name__
            
            # Set the executable to the function
            component_kwargs['executable'] = f
            
            # Create component with all kwargs
            component = WorkflowComponent(**component_kwargs)
            self.components[component.name] = component
            return f
        
        if func is not None:
            return decorator(func)
        return decorator
    
    def register_component(self, **kwargs) -> 'Workflow':
        """Register a component with keyword arguments."""
        component = WorkflowComponent(**kwargs)
        self.components[component.name] = component
        return self
    
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