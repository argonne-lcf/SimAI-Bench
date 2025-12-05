
import os
from networkx import DiGraph, topological_sort
from typing import Tuple, Dict
# Handle Dragon imports and type hints
try:
    import dragon
    from dragon.native.process_group import ProcessGroup
    from dragon.native.process import ProcessTemplate, Process as DragonProcess, Popen
    from dragon.infrastructure.policy import Policy
    from dragon.workflows.batch import Batch
    from dragon.workflows.batch.batch import AsyncDict, Task
    DRAGON_AVAILABLE = True
    from typing import Any, Sequence
except ImportError:
    DRAGON_AVAILABLE = False
    # When Dragon is not available, we'll use Any for type hints
    ProcessGroup = any

from .base import BaseExecutor
from SimAIBench.component import WorkflowComponent
from SimAIBench.dag import NodeStatus, DAG
from SimAIBench.config import OchestratorConfig
from SimAIBench.resources import NodeResourceList


class DragonCompiledTaskFuture:
    """
    A simple wrapper around dragon task to provide methods like exception, done, cancel, result
    :task
    """
    def __init__(self, task: Task, dag_task: Task):
        self.task = task
        self.dag_task = dag_task
    
    def cancel(self):
        raise RuntimeError("Can't cancel the dragon task")

    def done(self):
        try:
            self.dag_task.wait(0.1)
            return True
        except TimeoutError:
            return False

    def exception(self):
        self.dag_task.wait()
        return self.task.stderr.get()

    def result(self):
        return self.task.result


class DragonTaskFuture:
    """
    A simple wrapper around dragon task to provide methods like exception, done, cancel, result
    :task
    """
    def __init__(self, task: Task):
        self.task = task
    
    def cancel(self):
        raise RuntimeError("Can't cancel the dragon task")

    def done(self):
        try:
            self.task.wait(0.1)
            return True
        except TimeoutError:
            return False

    def exception(self):
        self.task.wait()
        return self.task.stderr.get()

    def result(self):
        return self.task.result

class DragonExecutor(BaseExecutor):
    def __init__(self, config: OchestratorConfig, sys_info: NodeResourceList):
        super().__init__(config, sys_info)
        if not DRAGON_AVAILABLE:
            raise ModuleNotFoundError("Dragon is not available")
        self.batch = Batch(num_workers=1, disable_telem=True)
    

    def submit_dag(self, cluster_resource: Any, dag: DAG) -> Tuple[DAG, Dict]:
        """
        Submit a DAG for execution

        Note: 
        Return future tracks the whole compiled DAG task.
        
        :param self: Description
        :param cluster_resource: Description
        :type cluster_resource: Any
        :param dag: Description
        :type dag: DAG
        :return: Description
        :rtype: Tuple[DAG, Dict]
        """
        futures = {}
        dependencies = {}
        tasks = {}
        graph: DiGraph = dag.graph
        node_execution_order = list(topological_sort(graph))
        self.logger.info(f"node execution order: {node_execution_order}")
        for i, node in enumerate(node_execution_order):
            node_obj = graph.nodes[node]
            if node_obj['status'] == NodeStatus.NOT_SUBMITTED:
                try:
                    self.logger.info(f"Submitting {node} for execution")
                    args = [cluster_resource]
                    # Iterate through dependencies and collect their futures
                    predecessors = list(graph.predecessors(node))
                    if predecessors:
                        self.logger.debug(f"Node {node} has {len(predecessors)} dependencies: {predecessors}")
                        for predecessor in predecessors:
                            args.append(dependencies[predecessor])
                    else:
                        self.logger.debug(f"Node {node} has no dependencies")
                    
                    task = self._create_task(node_obj, args)
                    tasks[node] = task
                    dependencies[node] = task.result
                    node_obj["status"] = next(node_obj["status"])
                except Exception as e:
                    self.logger.error(f"Submitting {node} failed with exception {e}")
                    node_obj["status"] = NodeStatus.FAILED  
            else:
                self.logger.info(f"Skipping {node} submission due to {node_obj['status']}") 
        if len(tasks) > 0:
            dag_task = self.batch.compile(list(tasks.values()))
            dag_task.start()
            for node in node_execution_order:
                futures[node] = DragonCompiledTaskFuture(task=tasks[node],dag_task=dag_task)
        return dag, futures

    def _create_task(self, node_obj: Any, args: Sequence) -> Task:
        """
        Submits a task using the TAPS engine.

        Args:
            task
            args: arguments
        """
        wc: WorkflowComponent =  node_obj["component"]

        nnodes = wc.nnodes
        ppn = wc.ppn
        cpu_affinity = wc.cpu_affinity
        gpu_affinity = wc.gpu_affinity
        nodes = wc.nodes
        np = ppn*nnodes

        templates = []
        for pid in range(np):
            policy = Policy(
                placement = Policy.Placement.HOST_NAME if nodes else Policy.Placement.DEFAULT,
                host_name = nodes[pid//ppn] if nodes else "",
                cpu_affinity = cpu_affinity,
                gpu_env_str = os.environ.get("SIMAIBENCH_GPUSELECTOR","ZE_AFFINITY_MASK"),
                gpu_affinity = [self.sys_info.gpus.index(gid) for gid in gpu_affinity] if gpu_affinity else None,
            )
            templates.append(
                ProcessTemplate(target=node_obj["callable"],args=args,env = os.environ.copy(), policy=policy)
            )
        if np == 1:
            task = self.batch.process(templates[0])
        else:
            task = self.batch.job(templates)
        
        return task

        
    def cleanup(self):
        try:
            self.batch.close()
            self.batch.join(timeout=5.0)
        except Exception as e:
            self.batch.terminate()