"""
Component module for workflow mini-apps providing data storage and server management.

Complete Workflow Examples:

1. Filesystem-based Workflow:
```python
# Server side
server_config = {"type": "filesystem", "server_address": "./data", "nshards": 32}
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
    "server_address": "localhost:6379",
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
    "server_address": "node1:6379,node2:6379,node3:6379",
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
    "server_address": "node1:7777,node2:7777",
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
    "server_address": "/path/to/dfuse/mount",  # dfuse mount directory
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
        
        if self.config["type"] == "filesystem":
            if "server_address" not in self.config:
                self.config["server_address"] = os.path.join(os.getcwd(), ".tmp")
            if "nshards" not in self.config:
                self.config["nshards"] = 64
            dirname = self.config["server_address"]
            os.makedirs(dirname, exist_ok=True)
            if self.logger:
                self.logger.info(f"Created filesystem directory at {dirname}")
        
        elif self.config["type"] == "node-local":
            self.config["server_address"] = "/tmp"
            if "nshards" not in self.config:
                self.config["nshards"] = 64
            if self.logger:
                self.logger.info(f"Using node-local directory {self.config['server-address']}")
        
        elif self.config["type"] == "redis":
            if "redis-server-exe" not in self.config:
                raise ValueError("redis-server-exe must be specified for Redis server")
            if "server_address" not in self.config:
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
                if "server_address" not in self.config:
                    raise ValueError("For DAOS POSIX mode, 'server-address' must point to a dfuse mount path")
                if "nshards" not in self.config:
                    self.config["nshards"] = 64
                dirname = self.config["server_address"]
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
        addresses = self.config["server_address"].split(",")
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
        addresses = self.config["server_address"].split(",")
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