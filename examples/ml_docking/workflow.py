import json

from sim import sim_main
from inference import infer_main
from sorter import sorter_main
from fine_tuning import finetune_main

from SimAIBench import Workflow
from SimAIBench import DataStore
from SimAIBench import ServerManager
from SimAIBench import server_registry, OchestratorConfig, SystemConfig
from SimAIBench.utils import get_nodes
import random
import string
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='ML Docking Workflow')
    parser.add_argument(
        '--server_type',
        type=str,
        default='filesystem',
        choices=['redis', 'filesytem', 'dragon', 'daos'],
        help='Type of server to use for data stores (default: filesystem)'
    )
    parser.add_argument(
        '--inference_nodes',
        type=int,
        default=1,
        help='Number of nodes to use for inference component'
    )
    parser.add_argument(
        '--training_nodes',
        type=int,
        default=1,
        help='Number of nodes to use for training component'
    )
    parser.add_argument(
        '--sorter_nodes',
        type=int,
        default=1,
        help='Number of nodes to use for sorter component'
    )
    
    return parser.parse_args()

def split_nodes(all_nodes, inference_nodes, training_nodes, sorter_nodes):
    """
    Split nodes into separate lists for each workflow component.
    
    Args:
        all_nodes: List of all available nodes
        inference_nodes: Number of nodes for inference
        training_nodes: Number of nodes for training
        sorter_nodes: Number of nodes for sorting
        
    Returns:
        tuple: (node_counts dict, nodelists dict)
    """
    num_tot_nodes = len(all_nodes)
    
    # Calculate node counts
    node_counts = {
        "sorting": sorter_nodes,
        "training": training_nodes,
        "inference": inference_nodes
    }
    
    assert sorter_nodes == inference_nodes, "Sorter and inference nodes must be equal"
    
    node_counts["simulation"] = num_tot_nodes - node_counts["sorting"]
    
    if node_counts["simulation"] <= 0:
        raise ValueError(
            f"Node partitioning not valid! "
            f"{num_tot_nodes=} simulation={node_counts['simulation']} "
            f"inference={node_counts['inference']} sorting={node_counts['sorting']} "
            f"training={node_counts['training']}"
        )
    
    # Split nodes into separate lists for each component
    nodelists = {}
    for key in node_counts.keys():
        if key == "sorting" or key == "inference":
            nodelists[key] = all_nodes[:node_counts[key]]
        elif key == "training":
            nodelists[key] = all_nodes[-1:]
        else:
            nodelists[key] = all_nodes[node_counts["sorting"]:node_counts["sorting"]+node_counts["simulation"]]
    
    return node_counts, nodelists

args = parse_args()

# Get total available nodes and split them
all_nodes = get_nodes()
node_counts, nodelists = split_nodes(
    all_nodes,
    args.inference_nodes,
    args.training_nodes,
    args.sorter_nodes
)

print(f"Node allocation: {node_counts}")
print(f"Node lists: {nodelists}")

##create configs
config = OchestratorConfig(name="process-pool")
aurora_cpus = list(range(104))
aurora_cpus.pop(52)
aurora_cpus.pop(0)
aurora_gpus = [f"{i}.{j}" for i in range(6) for j in range(2)]
system_config = SystemConfig(name="cluster",cpus=aurora_cpus,gpus=aurora_gpus,ncpus=len(aurora_cpus),ngpus=len(aurora_gpus))

if args.server_type == "filesystem":
    infer_server_config = server_registry.create_config(type="filesystem",server_address=".inference_store")
    top_candidates_server_config = server_registry.create_config(type="filesystem",server_address=".top_candidate_datastore",)
    training_data_server_config = server_registry.create_config(type="filesystem",server_address=".training_data_store")
elif args.server_type == "redis":
    with open("configs/server/redis_cluster.json","r") as f:
        redis_config = json.load(f)
    del redis_config["type"]
    print(f"Using redis config: {redis_config}")
    infer_input_config = redis_config.copy()
    infer_input_config["server_address"] = ",".join([f"{node}:6382" for node in nodelists["inference"]])
    infer_server_config = server_registry.create_config(type="redis",**infer_input_config)
    
    training_data_input_config = redis_config.copy()
    training_data_input_config["server_address"] = ",".join([f"{node}:6380" for node in nodelists["simulation"]])
    training_data_server_config = server_registry.create_config(type="redis",**training_data_input_config)
    
    top_candidates_input_config = redis_config.copy()
    top_candidates_input_config["server_address"] = ",".join([f"{node}:6381" for node in all_nodes])
    top_candidates_server_config = server_registry.create_config(type="redis",**top_candidates_input_config)
else:
    print(f"Server type {args.server_type} not supported in this example.")
    exit(1)

##create data store servers
infer_dataserver = ServerManager(name="inference",config=infer_server_config)
top_candidate_dataserver = ServerManager(name="top_candidate",config=top_candidates_server_config)
training_dataserver = ServerManager(name="training",config=training_data_server_config)
dataservers = [infer_dataserver,top_candidate_dataserver,training_dataserver]
for server in dataservers:
    server.start_server()
    if server.config.type == "redis" and server.config.is_clustered:
        ServerManager.create_redis_cluster(
            server_addresses=server.config.server_address.split(","),
            redis_cli_path="/home/ht1410/redis/src/redis-cli",
            replicas=0,  # No replicas for this setup
            timeout=30,
            logging=True
        )

infer_serverinfo = infer_dataserver.get_server_info()
top_candiate_serverinfo = top_candidate_dataserver.get_server_info()
training_serverinfo = training_dataserver.get_server_info()

##create workflow
workflow = Workflow(orchestrator_config=config, system_config=system_config)

    # # Required fields (no defaults) must come first
    # name: str
    # executable: Union[str, Callable]
    # type: str  #["remote","local"].
    # args: Dict[str, Any] = field(default_factory=dict)  # Arguments for the component
    # nodes: List[str] = field(default_factory=list)
    # nnodes: int = 1
    # ppn: int = 1
    # num_gpus_per_process: int = 0
    # cpu_affinity: List[int] = None
    # gpu_affinity: List[str] = None
    # env_vars: Dict[str, str] = field(default_factory=dict)
    # dependencies: List[Union[str, Dict[str, int]]] = field(default_factory=list)
    # return_dim: Sequence[int] = field(default_factory=list)

##load smiles into the inference data store
infer_ds = DataStore(name="inference_data",server_info=infer_serverinfo)
for smile_num in range(1000):
    random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
    infer_ds.stage_write(f"smile_{smile_num}", random_string)

workflow.register_component(
    name="inference",
    executable=infer_main,
    type="remote",
    args={"infer_serverinfo":infer_serverinfo,"top_candidate_serverinfo":top_candiate_serverinfo},
    nodes=nodelists["inference"],
    ppn=12,
    num_gpus_per_process=1,
    cpu_affinity=[1,5,9,13,17,21,53,57,61,65,69,73],
    gpu_affinity=["0.0","0.1","1.0","1.1","2.0","2.1","3.0","3.1","4.0","4.1","5.0","5.1"],
)

workflow.register_component(
    name="sorter",
    executable=sorter_main,
    type="remote",
    nodes=nodelists["sorting"],
    ppn=90,
    cpu_affinity=[cpu for cpu in system_config.cpus if cpu not in [1,5,9,13,17,21,53,57,61,65,69,73]],
    args={"infer_serverinfo":infer_serverinfo,"top_candidate_serverinfo":top_candiate_serverinfo}
)

workflow.register_component(
    name="simulation",
    executable=sim_main,
    type="remote",
    nodes=nodelists["simulation"],
    ppn=98,
    cpu_affinity=[cpu for cpu in system_config.cpus[4:]],
    args={"top_candidate_serverinfo":top_candiate_serverinfo,"training_serverinfo":training_serverinfo}
)

workflow.register_component(
    name="finetune",
    executable=finetune_main,
    type="remote",
    nodes=nodelists["training"],
    ppn=1,
    num_gpus_per_process=1,
    cpu_affinity=[1,2,3,4],
    gpu_affinity=["0.0"],
    args={"training_serverinfo":training_serverinfo,"top_candiate_serverinfo":top_candiate_serverinfo}
)

wf = workflow.launch()

infer_dataserver.stop_server()
top_candidate_dataserver.stop_server()
training_dataserver.stop_server()