from typing import Dict, List, Union, Callable, Any, Sequence
from dataclasses import dataclass, field


@dataclass
class WorkflowComponent:
    """
    Attributes:
        name (str): The name of the workflow component.
        executable (Union[str, Callable]): The executable to run for this component, either as a command string or a callable function.
        type (str): The type of executor to use for this component.
        args (Dict[str, Any]): Arguments to pass to the component during execution.
        nodes (List[str]): List of node names where the component should be executed.
        ppn (int): Number of processes per node to launch for this component. Defaults to 1.
        num_gpus_per_process (int): Number of GPUs to allocate per process. Defaults to 0.
        cpu_affinity (List[int]): List of CPU core indices to bind the process to. Optional.
        gpu_affinity (List[str]): List of GPU device IDs to bind the process to. Optional.
        env_vars (Dict[str, str]): Environment variables to set for the component execution.
        dependencies (List[Union[str, Dict[str, int]]]): List of dependencies for this component, either as component names or dictionaries specifying component names and data size in MB of the message size.
        return_dim: The array dimensions of the array to be returned by this component
    """
        
    # Required fields (no defaults) must come first
    name: str
    executable: Union[str, Callable]
    type: str  #["remote","local"].
    args: Dict[str, Any] = field(default_factory=dict)  # Arguments for the component
    nodes: List[str] = field(default_factory=list)
    nnodes: int = 1
    ppn: int = 1
    num_gpus_per_process: int = 0
    cpu_affinity: List[int] = None
    gpu_affinity: List[str] = None
    env_vars: Dict[str, str] = field(default_factory=dict)
    dependencies: List[Union[str, Dict[str, int]]] = field(default_factory=list)
    return_dim: Sequence[int] = field(default_factory=list)