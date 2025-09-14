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
from .servermanager import ServerManager
 
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

