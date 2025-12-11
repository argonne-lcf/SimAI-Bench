from .dag import NodeStatus
from .store import DagStore
import logging

logger = logging.getLogger(__name__)


class DagFuture:
    def __init__(self, dag_store: DagStore, node_name: str):
        self._node_name = node_name
        self._dag_store = dag_store
    
    def done(self):
        dag,last_updated = self._dag_store.get_dag()
        return dag.graph.nodes[self._node_name]["status"] == NodeStatus.COMPLETED or dag.graph.nodes[self._node_name]["status"] == NodeStatus.FAILED
    
    def exception(self):
        pass