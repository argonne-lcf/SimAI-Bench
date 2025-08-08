from SimAIBench import Workflow, ServerManager
from sim_exec import main as sim_main
from train_ai_exec import main as train_ai_main
import os
import argparse
import json
import logging

def get_nodes():
    """Extract node names from PBS_NODEFILE environment variable."""
    with open(os.getenv("PBS_NODEFILE", "/dev/null"), "r") as f:
        nodes = [line.split(".")[0] for line in f.readlines()]
    return nodes

def main():
    # Parse command line arguments for workflow configuration
    parser = argparse.ArgumentParser(description="Launch a workflow with a server component.")
    parser.add_argument("--server_config", type=str, required=True, help="Path to the server configuration file")
    parser.add_argument("--server_location", type=str, default="simulation", help="Location of the server (e.g., 'simulation', 'training', 'neither')")
    parser.add_argument("--sim_config", type=str, default="configs/sim_config.json", help="Path to the simulation configuration file")
    parser.add_argument("--staging_dir", type=str, default=".tmp", help="Staging directory of the file system")
    parser.add_argument("--data_size", type=int, default=1000, help="Size of the data to be processed in each simulation step")
    parser.add_argument("--infer_ai_config", type=str, default="configs/infer_ai_config.json", help="Path to the inference AI configuration file")
    parser.add_argument("--train_ai_config", type=str, default="configs/train_ai_config.json", help="Path to the training AI configuration file")
    args = parser.parse_args()

    # Set Intel GPU environment variable for Aurora system
    os.environ["ZE_FLAT_DEVICE_HIERARCHY"] = "COMPOSITE"
    
    # Initialize workflow with MPI launcher and Aurora system specifications
    my_workflow = Workflow(launcher={"mode":"mpi"}, sys_info={"name": "aurora", "ncores_per_node": 104, "ngpus_per_node": 12})

    # Get available compute nodes from PBS scheduler
    nodes = get_nodes()
    if not nodes:
        raise RuntimeError("No nodes found in PBS_NODEFILE.")
    
    # Allocate nodes: first node for AI training, remaining for simulations
    ai_nodes = nodes[:1]  # First node for AI training
    sim_nodes = nodes[1:]  # Remaining nodes for simulations
    nsims = len(sim_nodes)
    
    # Determine where to place the database server based on user preference
    if args.server_location == "simulation":
        db_nodes = sim_nodes  # Place server on simulation nodes (colocated with sims)
    elif args.server_location == "training":
        db_nodes = ai_nodes  # Place server on AI training nodes
    else:
        db_nodes = nodes  # Place server on all nodes

        db_nodes = nodes  # Place server on all nodes

    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load and configure the data server (Redis, Dragon, or filesystem)
    with open(os.path.join(root_dir, args.server_config), "r") as f:
        server_config = json.load(f)
        
    # For Redis/Dragon servers, set up network addresses using allocated nodes
    if server_config.get("type", "filesystem") == "redis" or server_config.get("type", "filesystem") == "dragon":
        server_config["server-address"] = ",".join([f"{n}:6875" for n in db_nodes])
        if server_config.get("type", "filesystem") == "dragon":
            if args.server_location == "simulation":
                server_config["server-options"] = {"total_mem": 107374182400*len(nodes)}
            else:
                server_config["server-options"] = {"total_mem": min(107374182400*len(nodes)//4, 107374182400*8)}
    elif server_config.get("type", "filesystem") == "filesystem":
        server_config["nshards"] = 8*len(nodes)
        if args.staging_dir.startswith("/"):
            server_config["server-address"] = args.staging_dir
        else:
            server_config["server-address"] = os.path.join(os.getcwd(), args.staging_dir)

    # Start the data server with logging enabled
    server = ServerManager("server", config=server_config, logging=True, log_level=logging.DEBUG)
    server.start_server()
    
    # If using clustered Redis, create the cluster after individual servers are started
    if server_config.get("type", "filesystem") == "redis" and server_config.get("is_clustered", False):
        ServerManager.create_redis_cluster(
            server_addresses=server_config["server-address"],
            redis_cli_path="/home/ht1410/redis/src/redis-cli",
            replicas=0,  # No replicas for this setup
            timeout=30,
            logging=True
        )

    # Get server information to pass to workflow components
    server_info = server.get_server_info()

    # Configure and register simulation components
    with open(os.path.join(root_dir, args.sim_config), "r") as f:
        sim_config = json.load(f)
    sim_config["data_size"] = args.data_size  # Set data size from command line argument
    print(f"Simulation config: {sim_config}")
    
    # Create one simulation component per simulation node
    for sim_id, node in enumerate(sim_nodes):
        my_workflow.register_component(
            name=f"sim_{sim_id}",
            executable=sim_main,
            type="remote" if server_info["type"] != "dragon" else "dragon",
            args= {
                "sim_config": sim_config,
                "server_info": server_info,
                "sim_id": sim_id,
                "is_colocated": args.server_location == "simulation"  # True if server is on same nodes as sims
            },
            nodes=[node],
            ppn=12,  # 12 processes per node (matches Aurora GPU count)
            num_gpus_per_process=1,  # 1 GPU per process
            # GPU affinity mapping for Aurora's 12 GPUs per node (tiles 0.0, 0.1, 1.0, 1.1, etc.)
            gpu_affinity=["0.0","0.1","1.0","1.1","2.0","2.1","3.0","3.1","4.0","4.1","5.0","5.1"],
        )

    # Configure and register AI training component
    with open(os.path.join(root_dir, args.train_ai_config), "r") as f:
        train_ai_config = json.load(f)
    train_ai_config["data_size"] = args.data_size  # Set data size for AI training
    
    my_workflow.register_component(
        name="train_ai",
        executable=train_ai_main,
        type="remote" if server_info["type"] != "dragon" else "dragon",
        args={
            "ai_config": train_ai_config,
            "server_info": server_info,
            "nsims": nsims,  # Number of simulation components to coordinate with
            "is_colocated": args.server_location == "training",  # True if server is on AI training nodes
        },
        nodes=ai_nodes,  # Run on dedicated AI training nodes
        ppn=12,  # 12 processes per node
        num_gpus_per_process=1,  # 1 GPU per process
        # GPU affinity mapping for Aurora's 12 GPUs per node
        gpu_affinity=["0.0","0.1","1.0","1.1","2.0","2.1","3.0","3.1","4.0","4.1","5.0","5.1"]
    )

    # Launch the complete workflow (simulations + AI training)
    my_workflow.launch()

    # Clean shutdown: stop the data server after workflow completion
    server.stop_server()



if __name__ == "__main__":
    main()