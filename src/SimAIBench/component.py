"""
Component module for workflow mini-apps providing data storage and server management.

Complete Workflow Examples:

1. Filesystem-based Workflow:
```python
# Server side
server_config = {"type": "filesystem", "server-address": "./data", "nshards": 32}
server = ServerManager("fs_server", server_config)
server.start_server()

# Client side - use server info directly
server_info = server.get_server_info()
datastore = DataStore("worker1", server_info)
# OR use serialized string
serialized = server.serialize()
datastore = DataStore("worker1", serialized)

datastore.stage_write("results", {"accuracy": 0.95})
data = datastore.stage_read("results")
```

2. Redis Server + Client Workflow (Colocated):
```python
# Server side
server_config = {
    "type": "redis",
    "server-address": "localhost:6379",
    "redis-server-exe": "/usr/bin/redis-server"
}
server = ServerManager("redis_server", server_config)
server.start_server()

# Client side - use server info directly with colocated option
server_info = server.get_server_info()
datastore = DataStore("worker1", server_info, is_colocated=True)
# OR use serialized string for remote clients
serialized = server.serialize()
datastore = DataStore("worker1", serialized, is_colocated=True)

datastore.stage_write("model_weights", weights_data)
```

3. Redis Cluster Workflow:
```python
# Start multiple Redis servers first, then create cluster
server_addresses = ["node1:6379", "node2:6379", "node3:6379"]
ServerManager.create_redis_cluster(server_addresses)

# Server side
server_config = {
    "type": "redis",
    "is_clustered": True,
    "server-address": "node1:6379,node2:6379,node3:6379",
    "redis-server-exe": "/usr/bin/redis-server"
}
server = ServerManager("redis_cluster", server_config)
server.start_server()

# Client side - use server info directly
datastore = DataStore("worker1", server.get_server_info())
```

4. Dragon Dictionary Workflow:
```python
# Server side
server_config = {
    "type": "dragon",
    "server-address": "node1:7777,node2:7777",
    "server-options": {"n_nodes": 2, "wait_for_keys": True}
}
server = ServerManager("dragon_server", server_config)
server.start_server()

# Client side - use server info directly
server_info = server.get_server_info()
datastore = DataStore("worker1", server_info)
# OR use serialized string for transmission
serialized = server.serialize()
datastore = DataStore("worker1", serialized)
```

5. DAOS (POSIX via Dfuse) Workflow:
```python
# Server side (no daemon to launch here; just point to a dfuse mount path)
server_config = {
    "type": "daos",                     # new option
    "mode": "posix",                    # posix uses a dfuse mount path
    "server-address": "/path/to/dfuse/mount",  # dfuse mount directory
    "nshards": 64
}
server = ServerManager("daos_posix_server", server_config)
server.start_server()

# Client side
server_info = server.get_server_info()
datastore = DataStore("worker1", server_info)

datastore.stage_write("results", {"accuracy": 0.95})
data = datastore.stage_read("results")
```

6. DAOS (Key-Value via PyDAOS) Workflow (optional):
```python
# Requires a DAOS pool+container and the PyDAOS bindings available in the env.
# This example shows the configuration shape; connect logic is implemented behind the scenes
# when `mode` is set to "kv" and PyDAOS is importable.
server_config = {
    "type": "daos",
    "mode": "kv",                       # use DAOS KV store via PyDAOS
    "pool_label": "pool0",              # or use pool_uuid
    "container_label": "cont0"          # or use container_uuid
}
server = ServerManager("daos_kv_server", server_config)
server.start_server()

datastore = DataStore("worker1", server.get_server_info())
datastore.stage_write("model_weights", weights_data)
```

7. Serialization for Remote Deployment:
```python
# Server side
server = ServerManager("my_server", config)
server.start_server()

# Serialize server info to base64 string
serialized_server = server.serialize()

# Send serialized_server to remote clients via network, file, etc.

# Client side (potentially on different machine)
datastore = DataStore("remote_client", serialized_server)
# DataStore automatically deserializes and configures client
```
"""

import time
import os
import sys
import pickle
import logging as logging_
import sqlite3
import shutil
import subprocess
import redis
import socket
from redis.cluster import RedisCluster
import zlib
from redis.cluster import ClusterNode
import base64
 
# Optional backends
try:
    import dragon
    from dragon.data.ddict import DDict
    from dragon.native.process_group import ProcessGroup
    from dragon.native.process import ProcessTemplate, Process
    from dragon.infrastructure.policy import Policy as DragonPolicy
    DRAGON_AVAILABLE=True
except:
    DRAGON_AVAILABLE=False

try:
    import pydaos
    PYDAOS_AVAILABLE = True
except Exception:
    PYDAOS_AVAILABLE = False

try:
    import cloudpickle
    CLOUDPICKLE_AVAILABLE = True
except ImportError:
    import pickle as cloudpickle
    CLOUDPICKLE_AVAILABLE = False
from typing import Union, List, Dict
import tempfile

class ServerManager:
    """
    Manages server setup and teardown for Filesystem, Node-local, Redis, Dragon, and DAOS servers.
    Responsible for launching, monitoring, and stopping server processes.
    """
    def __init__(self, name, config: dict, logging=False, log_level=logging_.INFO):
        self.name = name
        self.config = config
        self.redis_processes = []  # Changed from single process to list
        self.dragon_dict = None
        
        log_dir = os.getenv("WFMINI_LOG_DIR", "logs")
        # Setup logging
        if logging:
            self.logger = logging_.getLogger(f"{name}_server")
            self.logger.setLevel(log_level)
            log_dir = os.path.join(os.getcwd(), log_dir)
            os.makedirs(log_dir, exist_ok=True)
            
            log_file = os.path.join(log_dir, f"{name}_server.log")
            file_handler = logging_.FileHandler(log_file, mode="w")
            file_handler.setLevel(logging_.INFO)
            
            formatter = logging_.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            
            self.logger.debug(f"ServerManager {name} initialized with config {config}")
        else:
            self.logger = None
    
    def start_server(self):
        self._setup_server()

    def _setup_server(self):
        """Setup the appropriate server based on configuration."""
        if self.logger:
            self.logger.info(f"Setting up {self.config['type']} server on {self.config.get('server-address', 'unknown')}")
        
        if self.config["type"] == "filesystem":
            if "server-address" not in self.config:
                self.config["server-address"] = os.path.join(os.getcwd(), ".tmp")
            if "nshards" not in self.config:
                self.config["nshards"] = 64
            dirname = self.config["server-address"]
            os.makedirs(dirname, exist_ok=True)
            if self.logger:
                self.logger.info(f"Created filesystem directory at {dirname}")
        
        elif self.config["type"] == "node-local":
            self.config["server-address"] = "/tmp"
            if "nshards" not in self.config:
                self.config["nshards"] = 64
            if self.logger:
                self.logger.info(f"Using node-local directory {self.config['server-address']}")
        
        elif self.config["type"] == "redis":
            if "redis-server-exe" not in self.config:
                raise ValueError("redis-server-exe must be specified for Redis server")
            if "server-address" not in self.config:
                raise ValueError("Server address is required")
            self.redis_processes = self._start_redis_server()
        
        elif self.config["type"] == "dragon":
            if not DRAGON_AVAILABLE:
                raise ValueError("Dragon is not available")
            try:
                self.dragon_dict = self._start_dragon_dictionary()
                if isinstance(self.dragon_dict, DDict):
                    self.dragon_dict.setup_logging()
                elif isinstance(self.dragon_dict, list):
                    for ddict in self.dragon_dict:
                        ddict.setup_logging()
                if self.logger:
                    self.logger.info("Dragon dictionary created successfully!")
            except Exception as e:
                if self.logger:
                    self.logger.error("Dragon dictionary creation failed!")
                raise e
        
        elif self.config["type"] == "daos":
            # Two modes supported:
            # - POSIX (through a dfuse mount path). Behaves like the filesystem backend with sharded subdirs
            # - KV (through PyDAOS). Connection is deferred to client side; nothing to launch here
            mode = self.config.get("mode", "posix")
            if mode not in ("posix", "kv"):
                raise ValueError("DAOS mode must be one of {'posix','kv'}")
            if mode == "posix":
                if "server-address" not in self.config:
                    raise ValueError("For DAOS POSIX mode, 'server-address' must point to a dfuse mount path")
                if "nshards" not in self.config:
                    self.config["nshards"] = 64
                dirname = self.config["server-address"]
                os.makedirs(dirname, exist_ok=True)
                if self.logger:
                    self.logger.info(f"Using DAOS-POSIX mount at {dirname}")
            else:
                if not PYDAOS_AVAILABLE:
                    raise ValueError("PyDAOS not available. Install/activate PyDAOS or use DAOS POSIX mode.")
                # No server process to start; pool/container access happens in the client.
                if self.logger:
                    self.logger.info("DAOS KV mode selected; no server process to launch.")
    
    def _start_redis_server(self):
        """Start Redis server processes for all addresses."""
        addresses = self.config["server-address"].split(",")
        is_clustered = self.config.get("is_clustered", False)
        redis_processes = []
        
        for address in addresses:
            host = address.strip().split(":")[0]
            port = int(address.strip().split(":")[1])

            cmd_base = f"mpirun -np 1 -ppn 1 -hosts {host} {self.config.get('server-options',{}).get('mpi-options','')} {self.config['redis-server-exe']} --port {port} --bind 0.0.0.0 --protected-mode no --dir /tmp "
            cmd = f"{cmd_base} --cluster-enabled yes --cluster-config-file {self.name}_{host}_{port}.conf" if is_clustered else cmd_base
                
            redis_process = subprocess.Popen(cmd, shell=True, env=os.environ.copy(), stdout=subprocess.DEVNULL)
            if self.logger:
                self.logger.debug(f"Started Redis {'cluster ' if is_clustered else ''}server at {address}")
    
            redis_processes.append(redis_process)

        ready_servers = []
        while len(ready_servers) < len(addresses):
            if self.logger:
                self.logger.info(f"Waiting for {len(addresses) - len(ready_servers)}/{len(addresses)} Redis servers to be ready...")
            for address in addresses:
                if address in ready_servers:
                    continue
                host = address.strip().split(":")[0]
                port = int(address.strip().split(":")[1])
                if self._is_redis_server_running(host, port):
                    ready_servers.append(address)
            time.sleep(5)  # Wait before checking again
        if self.logger:
            self.logger.info(f"All Redis servers are ready: {len(ready_servers)}/{len(addresses)}")
        return redis_processes
    
    def _start_dragon_dictionary(self):
        """Start a Dragon dictionary server."""
        addresses = self.config["server-address"].split(",")
        is_clustered = self.config.get("is_clustered", False)
        nodes = [address.split(":")[0] for address in addresses]
        ports = [address.split(":")[1] for address in addresses]
        n_nodes = len(nodes)
        n_nodes_in = self.config.get("server-options", {}).get("n_nodes", None)
        
        if n_nodes_in is not None and n_nodes_in != n_nodes:
            if self.logger:
                self.logger.warning("Number of nodes in server-address differs from options. Using server-address count.")
            self.config["server-options"]["n_nodes"] = n_nodes
        
        policies = []
        for node in nodes:
            policy = DragonPolicy(placement=DragonPolicy.Placement.HOST_NAME,host_name=node)
            policies.append(policy)
        
        if is_clustered:
            opts = {
                "n_nodes": n_nodes,
                "policy": policies
            }
        
            if "policy" in self.config.get("server-options", {}):
                if self.logger:
                    self.logger.warning("Policy option provided as input. Replacing it!")
                self.config["server-options"]["policy"] = opts["policy"]
        
            opts.update(self.config.get("server-options", {}))
            opts["n_nodes"] = None
            opts["managers_per_node"] = None
        
            d = DDict(**opts)
            if self.logger:
                self.logger.info(f"Dragon dictionary created with options {opts}")
            while True:
                if self._is_dragon_server_running(d):
                    break
                if self.logger:
                    self.logger.info("Waiting for Dragon dictionary server to be ready...")
                time.sleep(5)  # Wait before checking again
            return d
        else:
            def create_local_kvstore(total_mem):
                import dragon.utils as du
                dd = DDict(managers_per_node=1,n_nodes=1,total_mem=total_mem)
                ##dd.serialize() will return string
                # du.set_local_kv("local_store", dd.serialize())
                with open(os.path.join("/tmp", "local_store.pickle"), "w") as f:
                    f.write(dd.serialize())

            if self.logger:
                self.logger.info("Creating local Dragon dictionary using Dragon utils")
                self.logger.info(f"Total memory for local Dragon dictionary: {self.config.get('server-options', {}).get('total_mem', 1024*1024*1024*5)/1024/1024/1024} GB")
            policy = DragonPolicy(distribution=DragonPolicy.Distribution.BLOCK)
            pg = ProcessGroup(policy=policy)
            for policy in policies:
                pg.add_process(
                        nproc=1,
                        template=ProcessTemplate(
                                    target=create_local_kvstore,
                                    args=(self.config.get("server-options", {}).get("total_mem", 1024*1024*1024*5),),
                                    cwd=os.getcwd(),
                                    policy=policy
                                )
                )
            pg.init()
            pg.start()
            pg.join()
            pg.close()
            return []
            # ddicts = []
            # for node_id in range(n_nodes):  # Create a Dragon dictionary for each node
            #     opts = {
            #         "n_nodes": 1,
            #         "policy": [policies[node_id]]
            #     }
            #     if "policy" in self.config.get("server-options", {}):
            #         if self.logger:
            #             self.logger.warning("Policy option provided as input. Replacing it!")
            #         self.config["server-options"]["policy"] = opts["policy"]

            #     opts.update(self.config.get("server-options", {}))
            #     opts["n_nodes"] = None
            #     opts["managers_per_node"] = None
            #     d = DDict(**opts)
            #     ddicts.append(d)
            # ready_ddicts = set()
            # while len(ready_ddicts) < len(ddicts):
            #     for id,ddict in enumerate(ddicts):
            #         if id in ready_ddicts:
            #             continue
            #         if self._is_dragon_server_running(ddict):
            #             ready_ddicts.add(id)
            #     if self.logger:
            #         self.logger.info("Waiting for Dragon dictionary servers to be ready...")
            #     time.sleep(5)  # Wait before checking again
            # return ddicts

    def _is_redis_server_running(self, host, port):
        """Check if a Redis server is running at the specified host and port."""
        try:
            r = redis.Redis(host=host, port=port, socket_connect_timeout=5)
            r.ping()
            r.close()
            return True
        except (redis.ConnectionError, redis.TimeoutError):
            return False

    def _is_dragon_server_running(self, ddict):
        """Check if the Dragon dictionary server is running."""
        try:
            from dragon.utils import host_id
            manager_nodes = ddict.manager_nodes
            for manager_id,manager_node in enumerate(manager_nodes):
                local_ddict = ddict.manager(manager_id)
                local_ddict.pput("test_key", "test_value")
                value = local_ddict["test_key"]
            return True
        except Exception as e:
            return False

    def _wait_for_redis_server(self, host, port, max_retries=30, retry_delay=1.0):
        """Wait for Redis server to be ready by attempting connections."""
        if self.logger:
            self.logger.info(f"Waiting for Redis server at {host}:{port} to be ready...")
        
        for attempt in range(max_retries):
            try:
                sock = socket.create_connection((host, port), timeout=5)
                sock.close()
                
                test_client = redis.Redis(host=host, port=port, socket_connect_timeout=5)
                test_client.ping()
                test_client.close()
                
                if self.logger:
                    self.logger.info(f"Redis server at {host}:{port} is ready (attempt {attempt + 1})")
                return
                
            except (socket.timeout, socket.error, redis.ConnectionError, redis.TimeoutError) as e:
                if self.logger and attempt == 0:
                    self.logger.debug(f"Redis server not ready yet: {e}")
                
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    error_msg = f"Redis server at {host}:{port} failed to start after {max_retries} attempts"
                    if self.logger:
                        self.logger.error(error_msg)
                    raise ConnectionError(error_msg)
    
    def poll_redis_server(self):
        """Check if all Redis server processes are still running."""
        if self.redis_processes:
            return all(process.poll() is None for process in self.redis_processes)
        return False
    
    def stop_server(self):
        """Stop the managed server."""
        if self.logger:
            self.logger.info("Stopping server!")
        
        if self.config["type"] == "redis" and self.redis_processes:
            for process in self.redis_processes:
                process.terminate()
                process.wait()
            if self.logger:
                self.logger.info(f"Stopped {len(self.redis_processes)} Redis server(s)")
        
        elif self.config["type"] == "dragon" and self.dragon_dict:
            if isinstance(self.dragon_dict, list):
                for d in self.dragon_dict:
                    d.destroy()
            else:
                self.dragon_dict.destroy()
            if self.logger:
                self.logger.info("Dragon dictionary destroyed")
        
        if self.logger:
            self.logger.info("Done stopping server!")
    
    def get_server_info(self):
        """Get information about the managed server."""
        info = {
            "name": self.name,
            "type": self.config["type"],
            "config": self.config.copy()  # Include the full config
        }
        
        if self.config["type"] == "redis":
            info["running"] = self.poll_redis_server()
        elif self.config["type"] == "dragon":
            info["dragon_dict"] = self.dragon_dict
            if self.dragon_dict:
                if isinstance(self.dragon_dict, list):
                    info["serial_dragon_dict"] = [d.serialize() if hasattr(d, 'serialize') else None for d in self.dragon_dict]
                else:
                    info["serial_dragon_dict"] = self.dragon_dict.serialize() if hasattr(self.dragon_dict, 'serialize') else None
            else:
                info["serial_dragon_dict"] = None

        elif self.config["type"] == "daos":
            info["daos_mode"] = self.config.get("mode", "posix")

        return info

    @classmethod
    def create_redis_cluster(cls, server_addresses: list, redis_cli_path: str = "redis-cli", 
                           replicas: int = 0, timeout: int = 30, logging=True):
        """
        Create a Redis cluster from existing Redis server instances.
        
        Args:
            server_addresses: List of Redis server addresses (e.g., ["host1:6379", "host2:6379"])
            redis_cli_path: Path to redis-cli executable (default: "redis-cli")
            replicas: Number of replicas per master (default: 0)
            timeout: Timeout in seconds for cluster creation (default: 30)
            logging: Whether to enable logging (default: True)
        
        Returns:
            True if cluster creation was successful, False otherwise
        
        Raises:
            ValueError: If server_addresses is empty or redis-cli command fails
        """
        if not server_addresses:
            raise ValueError("server_addresses cannot be empty")
        
        # Setup logging if enabled
        logger = None
        if logging:
            logger = logging_.getLogger("redis_cluster_creator")
            logger.setLevel(logging_.INFO)
            
            # Only add handler if not already present
            if not logger.handlers:
                console_handler = logging_.StreamHandler()
                formatter = logging_.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                console_handler.setFormatter(formatter)
                logger.addHandler(console_handler)
        
        try:
            start_time = time.time()
            
            # Build the redis-cli cluster create command
            create_cmd = (
                f"{redis_cli_path} --cluster create "
                f"{' '.join(server_addresses)} "
                f"--cluster-replicas {replicas} "
                f"--cluster-yes"
            )
            
            if logger:
                logger.info(f"Creating Redis cluster with addresses: {server_addresses}")
                logger.debug(f"Executing command: {create_cmd}")
            
            # Execute the cluster creation command
            env = os.environ.copy()
            result = subprocess.run(
                create_cmd, 
                shell=True, 
                check=True, 
                env=env, 
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                text=True
            )
            
            elapsed_time = time.time() - start_time
            
            if logger:
                logger.info(f"Redis cluster created successfully in {elapsed_time:.2f}s")
                if result.stdout:
                    logger.debug(f"Command stdout: {result.stdout}")
                if result.stderr:
                    logger.debug(f"Command stderr: {result.stderr}")
            
            return True
            
        except subprocess.TimeoutExpired:
            error_msg = f"Redis cluster creation timed out after {timeout} seconds"
            if logger:
                logger.error(error_msg)
            raise TimeoutError(error_msg)
            
        except subprocess.CalledProcessError as e:
            error_msg = f"Redis cluster creation failed with return code {e.returncode}"
            if logger:
                logger.error(error_msg)
                if e.stdout:
                    logger.error(f"Command stdout: {e.stdout}")
                if e.stderr:
                    logger.error(f"Command stderr: {e.stderr}")
            raise RuntimeError(error_msg)
            
        except Exception as e:
            error_msg = f"Unexpected error during Redis cluster creation: {e}"
            if logger:
                logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def serialize(self):
        """
        Serialize the server object for transmission or storage.
        
        Returns:
            str: Base64-encoded serialized server info
            
        Raises:
            RuntimeError: If serialization fails
        """
        try:
            server_info = self.get_server_info()
            serialized_bytes = cloudpickle.dumps(server_info)
            serialized_str = base64.b64encode(serialized_bytes).decode('utf-8')
            if self.logger:
                self.logger.info(f"Server info serialized successfully")
            return serialized_str
        except Exception as e:
            error_msg = f"Failed to serialize server info: {e}"
            if self.logger:
                self.logger.error(error_msg)
            raise RuntimeError(error_msg)

    @classmethod
    def deserialize(cls, serialized_data):
        """
        Deserialize server data and return server info.
        
        Args:
            serialized_data (str): Base64-encoded serialized server data
            
        Returns:
            dict: Server info dictionary
            
        Raises:
            RuntimeError: If deserialization fails
        """
        try:
            # Decode base64 string to bytes
            serialized_bytes = base64.b64decode(serialized_data.encode('utf-8'))
            
            # Use cloudpickle to deserialize
            server_info = cloudpickle.loads(serialized_bytes)
            
            if not isinstance(server_info, dict):
                raise ValueError("Invalid serialized server data format")
            
            return server_info
                
        except Exception as e:
            error_msg = f"Failed to deserialize server data: {e}"
            raise RuntimeError(error_msg)


class DataStore:
    """
    Handles client-side data operations including read, write, send, receive, and staging.
    Works with various backends: filesystem, node-local, Redis, Dragon, and DAOS (POSIX or KV).
    
    Initialized with serialized server info from ServerManager.serialize() or ServerManager.get_server_info()
    
    Args:
        name: Name of the DataStore instance
        server_info: Server information (serialized string or dict)
        logging: Enable logging (default: False)
        log_level: Logging level (default: logging.INFO)
        is_colocated: For Redis clients, only connect to servers on the same hostname (default: False)
    """
    def __init__(self, name, server_info:Union[str, dict], logging=False, log_level=logging_.INFO, is_colocated=False):
        self.name = name
        self.connections = []
        self.redis_client = None
        self.dragon_dict = None
        self.is_colocated = is_colocated

        log_dir = os.getenv("WFMINI_LOG_DIR", "logs")
        # Setup logging
        if logging:
            self.logger = logging_.getLogger(f"{name}_datastore")
            self.logger.setLevel(log_level)
            log_dir = os.path.join(os.getcwd(), log_dir)
            os.makedirs(log_dir, exist_ok=True)
        
            log_file = os.path.join(log_dir, f"{name}_datastore.log")
            file_handler = logging_.FileHandler(log_file, mode="w")
            file_handler.setLevel(logging_.INFO)
        
            formatter = logging_.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        else:
            self.logger = None

        # Use ServerManager.deserialize to get server info and extract client config
        if isinstance(server_info, str):
            # Handle base64-encoded serialized data
            deserialized_server_info = ServerManager.deserialize(server_info)
            self.config = deserialized_server_info["config"].copy()
            # Add Dragon-specific info if needed
            if deserialized_server_info.get("type") == "dragon" and "serial_dragon_dict" in deserialized_server_info:
                self.config["server-obj"] = deserialized_server_info["serial_dragon_dict"]
        elif isinstance(server_info, dict):
            if "config" in server_info:
                self.config = server_info["config"].copy()
                # Add Dragon-specific info if needed
                if server_info.get("type") == "dragon":
                    if server_info.get("dragon_dict", None) is not None and isinstance(server_info["dragon_dict"], DDict):
                        if self.logger:
                            self.logger.info("Using provided Dragon dictionary object")
                        self.config["server-obj"] = server_info["dragon_dict"]
                    elif "serial_dragon_dict" in server_info:
                        self.config["server-obj"] = server_info["serial_dragon_dict"]
            else:
                raise ValueError("Invalid server info dict format")
        else:
            raise ValueError("server_info must be str (base64) or dict")

        if self.logger:
            self.logger.debug(f"DataStore {name} initialized with config {self.config}")
        # Initialize client connections
        self._setup_client()
    
    def _setup_client(self):
        """Setup client connections based on configuration."""
        if self.config["type"] == "filesystem":
            if "server-address" not in self.config:
                self.config["server-address"] = os.path.join(os.getcwd(), ".tmp")
            if "nshards" not in self.config:
                self.config["nshards"] = 64
            dirname = self.config["server-address"]
            for shard in range(self.config["nshards"]):
                shard_dir = os.path.join(dirname, str(shard))
                os.makedirs(shard_dir, exist_ok=True)

        elif self.config["type"] == "node-local":
            if "server-address" not in self.config or not self.config["server-address"].startswith("/tmp"):
                self.config["server-address"] = "/tmp"
            if "nshards" not in self.config:
                self.config["nshards"] = 64
            dirname = self.config["server-address"]
            for shard in range(self.config["nshards"]):
                shard_dir = os.path.join(dirname, str(shard))
                os.makedirs(shard_dir, exist_ok=True)

        elif self.config["type"] == "redis":
            if "server-address" not in self.config:
                raise ValueError("Server address is required for Redis client")
            self.redis_client = self._create_redis_client()
        
        elif self.config["type"] == "dragon":
            if not DRAGON_AVAILABLE:
                raise ValueError("Dragon is not available")
            """
                if is_clustered:
                    when clustered, if you want tp write to local dragon_dict you need to use is_local=True in
                    read and write methods.
                    dragon_dict will the global DDict object, will be a list one
                    if is_colocated:
                        self.dragon_dict will be a list of dragon_dicts that are colocated with this client
                    else:
                        local_ddict will be a dict of ddicts for each manager node
                else:
                    ### if not clustered, you need a client_id to write to the correct ddict.
                    ###By default client_id is 0, which is the first manager node.
                    ##you can use is_local=True to write to local dict. This will through and error if is_colocated is True. 
                    if is_colocated:
                        self.dragon_dict = list of dragon_dicts that are colocated with this client
                        self.local_ddict will be none
                    else:
                        self.dragon_dict = [DDict for each node]
                        sellf.local_ddicts will be ddicts of all managers on this client node
            """
            self.dragon_dicts, self.local_ddicts = self._create_dragon_client()
        
        elif self.config["type"] == "daos":
            mode = self.config.get("mode", "posix")
            if mode not in ("posix", "kv"):
                raise ValueError("DAOS mode must be one of {'posix','kv'}")
            if mode == "posix":
                # Treat like filesystem sharded directory on a dfuse mount
                if "server-address" not in self.config:
                    raise ValueError("For DAOS POSIX mode, 'server-address' must point to a dfuse mount path")
                if "nshards" not in self.config:
                    self.config["nshards"] = 64
                dirname = self.config["server-address"]
                for shard in range(self.config["nshards"]):
                    shard_dir = os.path.join(dirname, str(shard))
                    os.makedirs(shard_dir, exist_ok=True)
            else:
                if not PYDAOS_AVAILABLE:
                    raise ValueError("PyDAOS not available. Install/activate PyDAOS or use DAOS POSIX mode.")
                # Establish the KV connection
                self._create_daos_kv_client()

    def _create_daos_kv_client(self):
        """Create/connect a DAOS KV client via PyDAOS.

        Expected config keys (either labels or UUIDs):
          - pool_label or pool_uuid
          - container_label or container_uuid

        Note: The concrete PyDAOS API may vary by version. Implementations should open the pool
        and container and bind a KV handle to self.daos_kv. Values are pickled bytes.
        """
        if not PYDAOS_AVAILABLE:
            raise ValueError("PyDAOS not available in environment")
        pool_label = self.config.get("pool_label") or self.config.get("pool_uuid")
        cont_label = self.config.get("container_label") or self.config.get("container_uuid")
        if not pool_label or not cont_label:
            raise ValueError("DAOS KV mode requires 'pool_label/pool_uuid' and 'container_label/container_uuid'")
        self.daos_cont = pydaos.DCont(pool_label, cont_label)
        self.daos_kv = self.daos_cont.dict("default")

    def _create_dragon_client(self):
        """Create a Dragon dictionary client connection."""
        is_clustered = self.config.get("is_clustered", False)
        if is_clustered:
            if isinstance(self.config["server-obj"], bytes):
                dragon_dicts = [DDict.attach(self.config["server-obj"], trace=True)]
            elif isinstance(self.config["server-obj"], str):
                dragon_dicts = [DDict.attach(self.config["server-obj"], trace=True)]
            elif isinstance(self.config["server-obj"], DDict):
                dragon_dicts = [self.config["server-obj"]]
            else:
                raise ValueError("Unknown server-obj type for Dragon client")
            
            local_ddicts = []
            ddict = dragon_dicts[0]
            if self.is_colocated:
                if self.logger:
                    self.logger.info("Using colocated Dragon dictionary client")
                from dragon.utils import host_id
                current_host = host_id()
                manager_nodes = ddict.manager_nodes
                for manager_id,manager_node in enumerate(manager_nodes):
                    if manager_node.h_uid == current_host:
                        local_ddicts.append(ddict.manager(manager_id))
            else:
                manager_nodes = ddict.manager_nodes
                for manager_id in range(len(manager_nodes)):
                    local_ddicts.append(ddict.manager(manager_id))
        else:
            if self.is_colocated:
                dragon_dicts = []
                local_ddicts = []
                # for obj in self.config["server-obj"]:
                #     if isinstance(obj, bytes):
                #         ddict = DDict.attach(obj, trace=True)
                #     elif isinstance(obj, str):
                #         ddict = DDict.attach(obj, trace=True)
                #     elif isinstance(obj, DDict):
                #         ddict = obj
                #     else:
                #         raise ValueError("Unknown server-obj type for Dragon client")
                    
                #     from dragon.utils import host_id
                #     current_host = host_id()
                #     manager_nodes = ddict.manager_nodes
                #     ###even if one of the manager nodes is colocated, we will use it
                #     for manager_id,manager_node in enumerate(manager_nodes):
                #         if manager_node.h_uid == current_host:
                #             dragon_dicts.append(ddict)
                #             break
                if not dragon_dicts:
                    if self.logger:
                        self.logger.info("Using colocated kv store for Dragon client")
                    import dragon.utils as du
                    try:
                        dd_ser = du.get_local_kv("local_store")
                        dd = DDict.attach(dd_ser)
                        if self.logger:
                            self.logger.info(f"Attached local Dragon dictionary using du")
                    except Exception as e:
                        if self.logger:
                            self.logger.error(f"Failed to attach local Dragon dictionary using du. falling back on file")
                        try:
                            with open(os.path.join("/tmp", "local_store.pickle"), "r") as f:
                                dd_ser = f.read()
                            dd = DDict.attach(dd_ser)
                        except Exception as e:
                            if self.logger:
                                self.logger.error(f"Failed to attach local Dragon dictionary from file: {e}")
                            raise e
                    dragon_dicts.append(dd)
            else:
                dragon_dicts = []
                local_ddicts = []
                for obj in self.config["server-obj"]:
                    if isinstance(obj, bytes):
                        dragon_dicts.append(DDict.attach(obj, trace=True))
                    elif isinstance(obj, str):
                        dragon_dicts.append(DDict.attach(obj, trace=True))
                    elif isinstance(obj, DDict):
                        dragon_dicts.append(obj)
                    else:
                        raise ValueError("Unknown server-obj type for Dragon client")
                    
                    ddict = dragon_dicts[-1]
                    from dragon.utils import host_id
                    current_host = host_id()
                    manager_nodes = ddict.manager_nodes
                    ###even if one of the manager nodes is colocated, we will use it
                    for manager_id,manager_node in enumerate(manager_nodes):
                        if manager_node.h_uid == current_host:
                            local_ddicts.append(ddict)
                            break
        return dragon_dicts, local_ddicts

    def _create_redis_client(self):
        """Create a Redis client connection."""
        is_clustered = self.config.get("is_clustered", False)
        clients = []
        
        try:
            if is_clustered:
                hosts = []
                ports = []
                for address in self.config["server-address"].split(","):
                    host, port_str = address.split(":")
                    port = int(port_str)
                    hosts.append(host)
                    ports.append(port)
                
                # Check which hosts are reachable
                reachable_hosts = []
                reachable_ports = []
                for host, port in zip(hosts, ports):
                    try:
                        sock = socket.create_connection((host, port), timeout=5)
                        sock.close()
                        reachable_hosts.append(host)
                        reachable_ports.append(port)
                        if self.logger:
                            self.logger.debug(f"Host {host}:{port} is reachable")
                    except (socket.timeout, socket.error) as e:
                        if self.logger:
                            self.logger.warning(f"Host {host}:{port} is not reachable: {e}")

                if not reachable_hosts:
                    error_msg = "No reachable Redis hosts found"
                    if self.logger:
                        self.logger.error(error_msg)
                    raise ConnectionError(error_msg)

                hosts = reachable_hosts
                ports = reachable_ports
                startup_nodes = [ClusterNode(host=host, port=port) for host, port in zip(hosts, ports)]
                client = RedisCluster(startup_nodes=startup_nodes)
                client.ping()
                clients.append(client)
                
            elif self.is_colocated:
                my_hostname = socket.gethostname()
                for address in self.config["server-address"].split(","):
                    if my_hostname not in address:
                        if self.logger:
                            self.logger.warning(f"Skipping address {address} as it does not match hostname {my_hostname}")
                        continue
                    else:
                        if self.logger:
                            self.logger.info(f"Creating colocated Redis client for address {address}")
                    host, port_str = address.split(":")
                    port = int(port_str)
                    client = redis.Redis(host=host, port=port)
                    client.ping()
                    clients.append(client)
                assert len(clients) > 0, "No colocated Redis clients created"
                
            else:  # Non-clustered Redis server
                for address in self.config["server-address"].split(","):
                    host, port_str = address.split(":")
                    port = int(port_str)
                    client = redis.Redis(host=host, port=port)
                    client.ping()
                    clients.append(client)
            
            if self.logger:
                self.logger.debug(f"Connected to Redis {'cluster' if is_clustered else 'server'}")
            return clients
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to connect to Redis: {e}")
            raise

    def connect(self, other_node):
        """Connect this node to another node."""
        if other_node not in self.connections:
            self.connections.append(other_node)
            if self.logger:
                self.logger.debug(f"Connected to {other_node.name}")

    def disconnect(self, other_node):
        """Disconnect this node from another node."""
        if other_node in self.connections:
            self.connections.remove(other_node)
            if self.logger:
                self.logger.debug(f"Disconnected from {other_node.name}")

    def send(self, data, targets: list = None):
        """Send data to all or selected connections."""
        targets = targets or self.connections
        if self.logger:
            self.logger.debug(f"Sending data: {data}")
        
        if self.config["type"] == "filesystem" or self.config["type"] == "node-local":
            dirname = self.config.get("server-address", os.path.join(os.getcwd(), ".tmp"))
            os.makedirs(dirname, exist_ok=True)
            for target in targets:
                filename = os.path.join(dirname, f"{self.name}_{target.name}_data.pickle")
                with open(filename, "wb") as f:
                    pickle.dump(data, f)
                if self.logger:
                    self.logger.debug(f"Data sent to {target.name} at {filename}")
        else:
            if self.logger:
                self.logger.error("Unsupported data transport type")
            raise ValueError("Unsupported data transport type")

    def receive(self, senders: list = None):
        """Receive data from connected senders."""
        data = {}
        senders = senders or self.connections
        dirname = self.config.get("server-address", os.path.join(os.getcwd(), ".tmp"))
        
        if not os.path.exists(dirname):
            if self.logger:
                self.logger.error(f"Directory {dirname} does not exist")
            raise AssertionError(f"Directory {dirname} does not exist")
            
        if self.config["type"] == "filesystem" or self.config["type"] == "node-local":
            for sender in senders:
                filename = os.path.join(dirname, f"{sender.name}_{self.name}_data.pickle")
                if not os.path.exists(filename):
                    if self.logger:
                        self.logger.error(f"File {filename} does not exist")
                    raise AssertionError(f"File {filename} does not exist")
                    
                with open(filename, "rb") as f:
                    data[sender.name] = pickle.load(f)
                    if self.logger:
                        self.logger.debug(f"Received data from {sender.name}")
        else:
            if self.logger:
                self.logger.error("Unsupported data transport type")
            raise ValueError("Unsupported data transport type")
        return data
    
    def stage_write(self, key, data, persistant: bool = True, client_id: int = 0, is_local: bool = False):
        """Stage data as a key-value pair."""
        if self.config["type"] == "dragon":
            assert DRAGON_AVAILABLE, "dragon is not available"
            if is_local:
                assert client_id < len(self.local_ddicts), "client_id must be < number of local Dragon dictionaries"
                ddict = self.local_ddicts[client_id]
            else:
                assert client_id < len(self.dragon_dicts), "client_id must be < number of local Dragon dictionaries"
                ddict = self.dragon_dicts[client_id]

            try:
                wait_for_keys = self.config.get("server-options", {}).get("wait_for_keys", None)
                if wait_for_keys is not None and wait_for_keys == True:
                    if persistant:
                        ddict.pput(key, data)
                    else:
                        ddict[key] = data
                else:
                    ddict[key] = data
                    if self.logger and not persistant:
                        self.logger.warning("Doing a persistant put!")
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Writing {key} failed with exception {e}")
                raise
        elif self.config["type"] == "redis":
            if not self.redis_client:
                raise ValueError("Redis client not initialized")
            try:
                serialized_data = pickle.dumps(data)
                self.redis_client[client_id].set(key, serialized_data)
                if self.logger:
                    self.logger.debug(f"Staged data for {key} in Redis")
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Failed to stage data in Redis: {e}")
                raise
        elif self.config["type"] == "filesystem" or self.config["type"] == "node-local":
            dirname = self.config["server-address"]
            h = zlib.crc32(key.encode('utf-8'))
            shard_number = h % self.config["nshards"]
            shard_dir = os.path.join(dirname, str(shard_number))
            os.makedirs(shard_dir, exist_ok=True)
            filename = os.path.join(shard_dir, f"{key}.pickle")
            with tempfile.NamedTemporaryFile(delete=False, dir=shard_dir) as temp_file:
                pickle.dump(data, temp_file)
                temp_filename = temp_file.name
            os.replace(temp_filename, filename)
            if self.logger:
                self.logger.debug(f"Staged data for {key} at {filename}")
        elif self.config["type"] == "daos":
            mode = self.config.get("mode", "posix")
            if mode == "posix":
                dirname = self.config["server-address"]
                h = zlib.crc32(key.encode('utf-8'))
                shard_number = h % self.config["nshards"]
                shard_dir = os.path.join(dirname, str(shard_number))
                os.makedirs(shard_dir, exist_ok=True)
                filename = os.path.join(shard_dir, f"{key}.pickle")
                with tempfile.NamedTemporaryFile(delete=False, dir=shard_dir) as temp_file:
                    pickle.dump(data, temp_file)
                    temp_filename = temp_file.name
                os.replace(temp_filename, filename)
                if self.logger:
                    self.logger.debug(f"Staged data for {key} at {filename} (DAOS POSIX)")
            else:
                if not hasattr(self, "daos_kv"):
                    raise RuntimeError("DAOS KV client not initialized; ensure PyDAOS connection is wired in _create_daos_kv_client")
                serialized_data = pickle.dumps(data)
                self.daos_kv[key] = serialized_data

        else:
            if self.logger:
                self.logger.error("Unsupported data transport type")
            raise ValueError("Unsupported data transport type")
    
    def stage_read(self, key, client_id: int = 0, timeout: int = 30, is_local: bool = False):
        """
            Read staged data using the key.
            is_local: If True, read from local Dragon dictionary manager nodes.
        """
        if self.config["type"] == "dragon":
            assert DRAGON_AVAILABLE, "dragon is not available"
            if is_local:
                assert client_id < len(self.local_ddicts), "client_id must be < number of local Dragon dictionaries"
                ddict = self.local_ddicts[client_id]
            else:
                assert client_id < len(self.dragon_dicts), "client_id must be < number of local Dragon dictionaries"
                ddict = self.dragon_dicts[client_id]

            try:
                return ddict[key]
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Reading {key} from local Dragon manager failed with exception {e}")
                raise

        elif self.config["type"] == "redis":
            if not self.redis_client:
                raise ValueError("Redis client not initialized")
            try:
                serialized_data = self.redis_client[client_id].get(key)
                if serialized_data is None:
                    if self.logger:
                        self.logger.error(f"Key {key} not found in Redis")
                    raise ValueError(f"Key {key} not found in Redis")
                data = pickle.loads(serialized_data)
                if self.logger:
                    self.logger.debug(f"Read staged data for {key} from Redis")
                return data
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Failed to read data from Redis: {e}")
                raise
        elif self.config["type"] == "filesystem" or self.config["type"] == "node-local":
            dirname = self.config["server-address"]
            h = zlib.crc32(key.encode('utf-8'))
            shard_number = h % self.config["nshards"]
            shard_dir = os.path.join(dirname, str(shard_number))
            filename = os.path.join(shard_dir, f"{key}.pickle")
            time_start = time.time()
            while not os.path.exists(filename):
                if time.time() - time_start > timeout:
                    if self.logger:
                        self.logger.error(f"Timed out waiting for file {filename} to be staged")
                    raise TimeoutError(f"Timed out waiting for file {filename} to be staged")
                time.sleep(0.1)
            with open(filename, "rb") as f:
                data = pickle.load(f)
                if self.logger:
                    self.logger.debug(f"Read staged data for {key} from {filename}")
                return data
        elif self.config["type"] == "daos":
            mode = self.config.get("mode", "posix")
            if mode == "posix":
                dirname = self.config["server-address"]
                h = zlib.crc32(key.encode('utf-8'))
                shard_number = h % self.config["nshards"]
                shard_dir = os.path.join(dirname, str(shard_number))
                filename = os.path.join(shard_dir, f"{key}.pickle")
                time_start = time.time()
                while not os.path.exists(filename):
                    if time.time() - time_start > timeout:
                        if self.logger:
                            self.logger.error(f"Timed out waiting for file {filename} to be staged")
                        raise TimeoutError(f"Timed out waiting for file {filename} to be staged")
                    time.sleep(0.1)
                with open(filename, "rb") as f:
                    data = pickle.load(f)
                    if self.logger:
                        self.logger.debug(f"Read staged data for {key} from {filename} (DAOS POSIX)")
                    return data
            else:
                if not hasattr(self, "daos_kv"):
                    raise RuntimeError("DAOS KV client not initialized; ensure PyDAOS connection is wired in _create_daos_kv_client")
                if key not in self.daos_kv:
                    raise ValueError(f"{key} doesn't exist")
                return pickle.loads(self.daos_kv[key])
        else:
            if self.logger:
                self.logger.error("Unsupported data transport type")
            raise ValueError("Unsupported data transport type")
    
    def poll_staged_data(self, key, client_id: int = 0, is_local: bool = False):
        """Check if data for the key is staged."""
        if self.config["type"] == "dragon":
            assert DRAGON_AVAILABLE, "dragon is not available"
            if is_local:
                assert client_id < len(self.local_ddicts), "client_id must be < number of local Dragon dictionaries"
                ddict = self.local_ddicts[client_id]
            else:
                assert client_id < len(self.dragon_dicts), "client_id must be < number of local Dragon dictionaries"
                ddict = self.dragon_dicts[client_id]

            try:
                return key in ddict.keys()
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Polling {key} failed with exception {e}")
                return False
        elif self.config["type"] == "redis":
            if not self.redis_client:
                raise ValueError("Redis client not initialized")
            try:
                return self.redis_client[client_id].exists(key) > 0
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Failed to poll data in Redis: {e}")
                raise
        elif self.config["type"] == "filesystem" or self.config["type"] == "node-local":
            dirname = self.config["server-address"]
            h = zlib.crc32(key.encode('utf-8'))
            shard_number = h % self.config["nshards"]
            shard_dir = os.path.join(dirname, str(shard_number))
            filename = os.path.join(shard_dir, f"{key}.pickle")
            return os.path.exists(filename)
        elif self.config["type"] == "daos":
            mode = self.config.get("mode", "posix")
            if mode == "posix":
                dirname = self.config["server-address"]
                h = zlib.crc32(key.encode('utf-8'))
                shard_number = h % self.config["nshards"]
                shard_dir = os.path.join(dirname, str(shard_number))
                filename = os.path.join(shard_dir, f"{key}.pickle")
                return os.path.exists(filename)
            else:
                if not hasattr(self, "daos_kv"):
                    raise RuntimeError("DAOS KV client not initialized; ensure PyDAOS connection is wired in _create_daos_kv_client")
                return key in self.daos_kv
        else:
            if self.logger:
                self.logger.error("Unsupported data transport type")
            raise ValueError("Unsupported data transport type")
        
    def clean_staged_data(self, key, client_id: int = 0, is_local: bool = False):
        """Clear the staging area for the given key."""
        if self.config["type"] == "dragon":
            assert DRAGON_AVAILABLE, "dragon is not available"
            if is_local:
                assert client_id < len(self.local_ddicts), "client_id must be < number of local Dragon dictionaries"
                ddict = self.local_ddicts[client_id]
            else:
                assert client_id < len(self.dragon_dicts), "client_id must be < number of local Dragon dictionaries"
                ddict = self.dragon_dicts[client_id]

            try:
                ddict.pop(key)
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Deleting {key} failed with exception {e}")
                    
        elif self.config["type"] == "redis":
            if not self.redis_client:
                raise ValueError("Redis client not initialized")
            
            try:
                if not self.redis_client[client_id].exists(key):
                    if self.logger:
                        self.logger.error(f"Key {key} not found in Redis")
                    raise ValueError(f"Key {key} not found in Redis")
                
                self.redis_client[client_id].delete(key)
                if self.logger:
                    self.logger.debug(f"Cleared staged data for {key} from Redis")
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Failed to clean data in Redis: {e}")
                raise
            
        elif self.config["type"] == "filesystem" or self.config["type"] == "node-local":
            # Remove file directly
            dirname = self.config["server-address"]
            h = zlib.crc32(key.encode('utf-8'))
            shard_number = h % self.config["nshards"]
            shard_dir = os.path.join(dirname, str(shard_number))
            filename = os.path.join(shard_dir, f"{key}.pickle")
            if os.path.exists(filename):
                try:
                    os.remove(filename)
                    if self.logger:
                        self.logger.debug(f"Cleared staged data for {key} and deleted file {filename}")
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"Failed to delete file {filename}: {e}. The file may be in use or you may not have permission.")
            else:
                if self.logger:
                    self.logger.error(f"File {filename} does not exist")
                raise ValueError(f"File {filename} does not exist")
        elif self.config["type"] == "daos":
            mode = self.config.get("mode", "posix")
            if mode == "posix":
                dirname = self.config["server-address"]
                h = zlib.crc32(key.encode('utf-8'))
                shard_number = h % self.config["nshards"]
                shard_dir = os.path.join(dirname, str(shard_number))
                filename = os.path.join(shard_dir, f"{key}.pickle")
                if os.path.exists(filename):
                    try:
                        os.remove(filename)
                        if self.logger:
                            self.logger.debug(f"Cleared staged data for {key} and deleted file {filename} (DAOS POSIX)")
                    except Exception as e:
                        if self.logger:
                            self.logger.warning(f"Failed to delete file {filename}: {e}. The file may be in use or you may not have permission.")
                else:
                    if self.logger:
                        self.logger.error(f"File {filename} does not exist")
                    raise ValueError(f"File {filename} does not exist")
            else:
                if not hasattr(self, "daos_kv"):
                    raise RuntimeError("DAOS KV client not initialized; ensure PyDAOS connection is wired in _create_daos_kv_client")
                if key not in self.daos_kv:
                    raise ValueError(f"{key} doesn't exist")
                del self.daos_kv[key]
        else:
            if self.logger:
                self.logger.error("Unsupported data transport type")
            raise ValueError("Unsupported data transport type")

    def get_connections(self):
        """Return a list of connected nodes."""
        return self.connections

    def clean(self):
        """Clean up the datastore."""
        if self.config["type"] == "filesystem" or self.config["type"] == "node-local":
            dirname = self.config.get("server-address", os.path.join(os.getcwd(), ".tmp"))
            if os.path.exists(dirname):
                shutil.rmtree(dirname)
                if self.logger:
                    self.logger.debug(f"Cleaned up directory {dirname}")
        elif self.config["type"] == "dragon":
            for ddict in self.dragon_dicts:
                ddict.detach()
            if self.logger:
                self.logger.info("Dragon client detached")
        else:
            if self.logger:
                self.logger.debug("No cleanup needed for this backend type")

    def flush_logger(self):
        if self.logger:
            for handler in self.logger.handlers:
                handler.flush()

    def __repr__(self):
        return f"<DataStore name={self.name}, type={self.config['type']}>"

