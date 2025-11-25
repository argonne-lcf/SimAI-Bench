from dataclasses import dataclass, field
from typing import List, Dict, Any
from abc import ABC, abstractmethod
from SimAIBench.config import SystemConfig
import copy

@dataclass(frozen=True, eq=True)
class NodeResource(ABC):
    """Base class for node resources"""
    
    @property
    @abstractmethod
    def cpu_count(self) -> int:
        """Total number of CPUs."""
        pass
    
    @property
    @abstractmethod
    def gpu_count(self) -> int:
        """Total number of GPUs."""
        pass
    
    @property
    def counts(self) -> dict:
        """Counts of all resources"""
        return {"cpus": self.cpu_count, "gpus": self.gpu_count}
    
    def is_empty(self) -> bool:
        """Check if resource has no CPUs or GPUs."""
        return self.cpu_count == 0 and self.gpu_count == 0
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(cpus={self.cpu_count}, gpus={self.gpu_count})"
    
    def __add__(self, other):
        if isinstance(other, NodeResource):
            return self._add_impl(other)
        return NotImplemented
    
    def __sub__(self, other):
        if isinstance(other, NodeResource):
            return self._sub_impl(other)
        return NotImplemented

    def __radd__(self, other):
        if other == 0:  # Support sum() with start=0
            return self
        return self.__add__(other)
    
    @abstractmethod
    def _add_impl(self, other: 'NodeResource') -> 'NodeResource':
        """Implementation-specific addition"""
        pass
    
    @abstractmethod
    def _sub_impl(self, other: 'NodeResource') -> 'NodeResource':
        """Implementation-specific subtraction"""
        pass
    
    @abstractmethod
    def __contains__(self, other) -> bool:
        """Check if another resource is contained within this one."""
        pass


@dataclass(frozen=True, eq=True)
class NodeResourceCount(NodeResource):
    """Count-based node resource representation"""
    ncpus: int = 0
    ngpus: int = 0
    # Future: memory: int = 0
    
    @property
    def cpu_count(self) -> int:
        return self.ncpus
    
    @property
    def gpu_count(self) -> int:
        return self.ngpus
    
    def _add_impl(self, other: NodeResource) -> 'NodeResourceCount':
        return NodeResourceCount(
            ncpus=self.ncpus + other.cpu_count,
            ngpus=self.ngpus + other.gpu_count
        )
    
    def _sub_impl(self, other: NodeResource) -> 'NodeResourceCount':
        return NodeResourceCount(
            ncpus=max(0, self.ncpus - other.cpu_count),
            ngpus=max(0, self.ngpus - other.gpu_count)
        )
    
    def __contains__(self, other) -> bool:
        """Check if another resource can be satisfied by this count-based resource."""
        if isinstance(other, NodeResource):
            return (other.cpu_count <= self.ncpus and 
                    other.gpu_count <= self.ngpus)
        return False


@dataclass(frozen=True, eq=True)
class NodeResourceList(NodeResource):
    """List-based (specific IDs) node resource representation"""
    cpus: tuple[int, ...] = field(default_factory=tuple)
    gpus: tuple[int, ...] = field(default_factory=tuple)
    # Future: memory: int = 0
    
    @property
    def cpu_count(self) -> int:
        return len(self.cpus)
    
    @property
    def gpu_count(self) -> int:
        return len(self.gpus)
    
    def _add_impl(self, other: NodeResource) -> 'NodeResourceList':
        if isinstance(other, NodeResourceList):
            return NodeResourceList(
                cpus=tuple(set(self.cpus) | set(other.cpus)),
                gpus=tuple(set(self.gpus) | set(other.gpus))
            )
        elif isinstance(other, NodeResourceCount):
            # Convert count to consecutive IDs and add
            next_cpu_id = max(self.cpus) + 1 if self.cpus else 0
            next_gpu_id = max(self.gpus) + 1 if self.gpus else 0
            new_cpus = tuple(range(next_cpu_id, next_cpu_id + other.ncpus))
            new_gpus = tuple(range(next_gpu_id, next_gpu_id + other.ngpus))
            return NodeResourceList(
                cpus=self.cpus + new_cpus,
                gpus=self.gpus + new_gpus
            )
        return NotImplemented
    
    def _sub_impl(self, other: NodeResource) -> 'NodeResourceList':
        if isinstance(other, NodeResourceList):
            remaining_cpus = tuple(set(self.cpus) - set(other.cpus))
            remaining_gpus = tuple(set(self.gpus) - set(other.gpus))
            return NodeResourceList(
                cpus=remaining_cpus,
                gpus=remaining_gpus
            )
        elif isinstance(other, NodeResourceCount):
            # Remove first N CPUs and GPUs
            remaining_cpus = self.cpus[other.ncpus:]
            remaining_gpus = self.gpus[other.ngpus:]
            return NodeResourceList(
                cpus=remaining_cpus,
                gpus=remaining_gpus
            )
        return NotImplemented
    
    def __contains__(self, other) -> bool:
        """Check if another resource is contained within this list-based resource."""
        if isinstance(other, NodeResourceList):
            # Check if all CPUs and GPUs in 'other' are available in 'self'
            return (set(other.cpus).issubset(set(self.cpus)) and 
                    set(other.gpus).issubset(set(self.gpus)))
        elif isinstance(other, NodeResourceCount):
            # Check if we have enough resources
            return (other.ncpus <= self.cpu_count and 
                    other.ngpus <= self.gpu_count)
        return False
    
    @classmethod
    def from_config(self, info: SystemConfig):
        """creates a node resource list from a dict"""
        return NodeResourceList(
            cpus=list(range(info.ncpus)) if len(info.cpus) == 0 else copy.deepcopy(info.cpus),
            gpus = list(range(info.ngpus)) if len(info.gpus) == 0 else copy.deepcopy(info.gpus)
        )


@dataclass(frozen=True, eq=True)
class JobResource:
    """
    Represents the computational resources required for a job.

    This immutable dataclass encapsulates a collection of node resources
    that define the computational requirements for executing a job in a
    distributed computing environment.

    Attributes:
        resources (List[NodeResource]): A list of NodeResource objects defining
            the computational requirements for the job.
        nodes (List): A list of node identifiers where the job resources
            will be allocated. Defaults to an empty list.
    """
    resources: List[NodeResource]
    nodes: List =  field(default_factory=list)

    def __post_init__(self):
        if self.nodes:
            assert len(self.nodes) == len(self.resources), "number of nodes != number of job resources"
        
        # Validate that resources is not empty
        if not self.resources:
            raise ValueError("JobResource must have at least one resource")
        
        # Validate that all resources are NodeResource instances
        for i, resource in enumerate(self.resources):
            if not isinstance(resource, NodeResource):
                raise TypeError(f"Resource at index {i} must be a NodeResource instance")
    
    def __repr__(self) -> str:
        total_cpus = sum(r.cpu_count for r in self.resources)
        total_gpus = sum(r.gpu_count for r in self.resources)
        nodes_info = f", nodes={self.nodes}" if self.nodes else ""
        return f"JobResource({len(self.resources)} nodes, total_cpus={total_cpus}, total_gpus={total_gpus}{nodes_info})"