"""
Redis backend implementation for DataStore and ServerManager.

This module provides Redis-based storage for data staging operations.
Supports both standalone Redis servers and Redis clusters.
"""

import os
import pickle
import socket
import subprocess
import time
import logging as logging_
import contextlib
from typing import Any, Union, List
from .base import BaseDataStore, BaseServerManager

# Import for type checking
from SimAIBench.config import RedisServerConfig

try:
    import redis
    from redis.cluster import RedisCluster, ClusterNode
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class DataStoreRedis(BaseDataStore):
    """
    Redis-based DataStore implementation.
    
    Stores data using Redis key-value store with pickle serialization.
    Supports both standalone Redis and Redis cluster configurations.
    """
    
    def __init__(self, name: str, server_info: Union[str, dict], logging: bool = False, 
                 log_level: int = logging_.INFO, is_colocated: bool = False):
        super().__init__(name, server_info, logging, log_level, is_colocated)
        
        if not REDIS_AVAILABLE:
            raise ImportError("Redis client library not available")
        
        self.redis_client = None
        
        # Parse server info
        if isinstance(server_info, str):
            from .base import BaseServerManager
            deserialized_server_info = BaseServerManager.deserialize(server_info)
            self.config = deserialized_server_info["config"].copy()
        elif isinstance(server_info, dict):
            if "config" in server_info:
                self.config = server_info["config"].copy()
            else:
                self.config = server_info.copy()
        else:
            raise ValueError("server_info must be str (base64) or dict")
            
        if self.logger:
            self.logger.debug(f"DataStoreRedis {name} initialized with config {self.config}")
            
        # Initialize client connections
        self._setup_client()
    
    def _setup_client(self):
        """Setup Redis client connection."""
        if "server_address" not in self.config:
            raise ValueError("Server address is required for Redis client")
        self.redis_client = self._create_redis_client()
    
    def _create_redis_client(self):
        """Create a Redis client connection."""
        is_clustered = self.config.get("is_clustered", False)
        clients = []
        
        try:
            if is_clustered:
                hosts = []
                ports = []
                for address in self.config["server_address"].split(","):
                    host = address.strip().split(":")[0]
                    port = int(address.strip().split(":")[1])
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
                    except (socket.timeout, socket.error):
                        if self.logger:
                            self.logger.warning(f"Redis server at {host}:{port} is not reachable")

                if not reachable_hosts:
                    raise ConnectionError("No reachable Redis servers found")

                hosts = reachable_hosts
                ports = reachable_ports
                startup_nodes = [ClusterNode(host=host, port=port) for host, port in zip(hosts, ports)]
                client = RedisCluster(startup_nodes=startup_nodes)
                client.ping()
                clients.append(client)
                
            elif self.is_colocated:
                my_hostname = socket.gethostname()
                for address in self.config["server_address"].split(","):
                    host = address.strip().split(":")[0]
                    port = int(address.strip().split(":")[1])
                    if self.logger:
                        self.logger.info(f"Server hostname:{host}, my hostname:{my_hostname}. Server port: {port}")
                    if host == my_hostname or host == "localhost" or host == "127.0.0.1":
                        client = redis.Redis(host=host, port=port)
                        client.ping()
                        clients.append(client)
                        if self.logger:
                            self.logger.debug(f"Connected to colocated Redis server at {host}:{port}")
                assert len(clients) > 0, "No colocated Redis clients created"
                
            else:  # Non-clustered Redis server
                for address in self.config["server_address"].split(","):
                    host = address.strip().split(":")[0]
                    port = int(address.strip().split(":")[1])
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
    
    def stage_write(self, key: str, data: Any, persistant: bool = True, client_id: int = 0, is_local: bool = False):
        """Stage data as a key-value pair in Redis."""
        if not self.redis_client:
            raise ValueError("Redis client not initialized")
        try:
            serialized_data = pickle.dumps(data)
            self.redis_client[client_id].set(key, serialized_data)
            if self.logger:
                self.logger.debug(f"Staged data for {key} in Redis")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to stage data for {key}: {e}")
            raise
    
    def stage_read(self, key: str, client_id: int = 0, timeout: int = 30, is_local: bool = False):
        """Read staged data using the key from Redis."""
        if not self.redis_client:
            raise ValueError("Redis client not initialized")
        try:
            serialized_data = self.redis_client[client_id].get(key)
            if serialized_data is None:
                raise KeyError(f"Key {key} not found in Redis")
            data = pickle.loads(serialized_data)
            if self.logger:
                self.logger.debug(f"Read data for {key} from Redis")
            return data
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to read data for {key}: {e}")
            raise
    
    def poll_staged_data(self, key: str, client_id: int = 0, is_local: bool = False) -> bool:
        """Check if data for the key is staged in Redis."""
        if not self.redis_client:
            raise ValueError("Redis client not initialized")
        try:
            return self.redis_client[client_id].exists(key) > 0
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to poll data for {key}: {e}")
            raise
    
    def clean_staged_data(self, key: str, client_id: int = 0, is_local: bool = False):
        """Clear the staging area for the given key from Redis."""
        if not self.redis_client:
            raise ValueError("Redis client not initialized")
        
        try:
            if not self.redis_client[client_id].exists(key):
                if self.logger:
                    self.logger.warning(f"Key {key} does not exist in Redis")
                raise KeyError(f"Key {key} does not exist in Redis")
            
            self.redis_client[client_id].delete(key)
            if self.logger:
                self.logger.debug(f"Deleted key {key} from Redis")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to delete key {key}: {e}")
            raise
    
    def acquire_lock(self, lock_name: str, acquire_timeout: int = 100, lock_timeout: int = 300):
        """Acquire a distributed lock using Redis."""
        if not self.redis_client:
            raise ValueError("Redis client not initialized")
            
        import contextlib
        from redis.exceptions import LockError
        
        @contextlib.contextmanager
        def redis_lock():
            lock = self.redis_client[0].lock(
                name=lock_name,
                timeout=lock_timeout,
                blocking_timeout=acquire_timeout
            )
            try:
                acquired = lock.acquire()
                if not acquired:
                    raise TimeoutError(f"Could not acquire lock '{lock_name}' within {acquire_timeout} seconds")
                if self.logger:
                    self.logger.debug(f"Acquired lock: {lock_name}")
                yield
            finally:
                try:
                    lock.release()
                    if self.logger:
                        self.logger.debug(f"Released lock: {lock_name}")
                except LockError:
                    if self.logger:
                        self.logger.warning(f"Lock {lock_name} was already released or expired")
        
        return redis_lock()


class ServerManagerRedis(BaseServerManager):
    """
    Redis-based ServerManager implementation.
    
    Manages Redis server processes and cluster setup.
    Supports both standalone Redis servers and Redis clusters.
    """
    
    def __init__(self, name: str, config: RedisServerConfig, logging: bool = False, log_level: int = logging_.INFO):
        super().__init__(name, config, logging, log_level)
        self.redis_processes = []
        
        if not REDIS_AVAILABLE:
            raise ImportError("Redis client library not available")
    
    def start_server(self):
        """Start Redis server(s)."""
        self._setup_server()

    def _setup_server(self):
        """Setup Redis server(s)."""
        if self.logger:
            self.logger.info(f"Setting up Redis server on {getattr(self.config, 'server_address', 'unknown')}")
        
        if not hasattr(self.config, 'redis_server_exe') \
            or not self.config.redis_server_exe or \
                not os.path.exists(self.config.redis_server_exe):
            raise ValueError("redis_server_exe must be specified for Redis server")
        if not hasattr(self.config, 'server_address') or not self.config.server_address:
            raise ValueError("Server address is required")
        self.redis_processes = self._start_redis_server()
    
    def _start_redis_server(self):
        """Start Redis server processes for all addresses."""
        addresses = self.config.server_address.split(",")
        is_clustered = getattr(self.config, 'is_clustered', False)
        redis_processes = []
        
        for address in addresses:
            host = address.strip().split(":")[0]
            port = int(address.strip().split(":")[1])

            server_options = getattr(self.config, 'server_options', {})
            mpi_options = server_options.get('mpi-options', '') if isinstance(server_options, dict) else ''
            cmd_base = f"mpirun -np 1 -ppn 1 -hosts {host} {mpi_options} {self.config.redis_server_exe} --port {port} --bind 0.0.0.0 --protected-mode no --dir /tmp "
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
    
    def _is_redis_server_running(self, host: str, port: int) -> bool:
        """Check if a Redis server is running at the specified host and port."""
        try:
            r = redis.Redis(host=host, port=port, socket_connect_timeout=5)
            r.ping()
            r.close()
            return True
        except (redis.ConnectionError, redis.TimeoutError):
            return False
    
    def poll_redis_server(self) -> bool:
        """Check if all Redis server processes are still running."""
        if self.redis_processes:
            return all(process.poll() is None for process in self.redis_processes)
        return False
    
    def stop_server(self):
        """Stop the Redis server(s)."""
        if self.logger:
            self.logger.info("Stopping Redis server(s)!")
        
        if self.redis_processes:
            for process in self.redis_processes:
                process.terminate()
                process.wait()
            if self.logger:
                self.logger.info(f"Stopped {len(self.redis_processes)} Redis server(s)")
        
        if self.logger:
            self.logger.info("Done stopping Redis server(s)!")
    
    def get_server_info(self) -> dict:
        """Get information about the Redis server."""
        info = {
            "name": self.name,
            "type": self.config.type,
            "config": self.config.model_dump(),
            "running": self.poll_redis_server()
        }
        return info
    
    @classmethod
    def create_redis_cluster(cls, server_addresses: List[str], redis_cli_path: str = "redis-cli", 
                           replicas: int = 0, timeout: int = 30, logging: bool = True) -> bool:
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