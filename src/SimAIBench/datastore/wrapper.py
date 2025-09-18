"""
Wrapper classes for DataStore and ServerManager that delegate to appropriate backends.

This module provides the main DataStore and ServerManager classes that users interact with.
These classes automatically detect the backend type and delegate to the appropriate implementation.
"""

import logging as logging_
from typing import Union, Any, Optional
import contextlib

from .base import BaseDataStore, BaseServerManager
from .filesystem import DataStoreFilesystem, ServerManagerFilesystem
from .redis import DataStoreRedis, ServerManagerRedis
from .dragon import DataStoreDragon, ServerManagerDragon
from .daos import DataStoreDaos, ServerManagerDaos
from SimAIBench.config import ServerConfig



# Registry of available backends
DATASTORE_BACKENDS = {
    "filesystem": DataStoreFilesystem,
    "node-local": DataStoreFilesystem,  # node-local uses the same implementation as filesystem
    "redis": DataStoreRedis,
    "dragon": DataStoreDragon,
    "daos": DataStoreDaos,
}

SERVERMANAGER_BACKENDS = {
    "filesystem": ServerManagerFilesystem,
    "node-local": ServerManagerFilesystem,  # node-local uses the same implementation as filesystem
    "redis": ServerManagerRedis,
    "dragon": ServerManagerDragon,
    "daos": ServerManagerDaos,
}


class DataStore:
    """
    DataStore wrapper class that automatically delegates to the appropriate backend.
    
    This class maintains the original API while internally routing operations to
    backend-specific implementations based on the server configuration.
    
    Args:
        name: Name of the DataStore instance
        server_info: Server information (serialized string or dict)
        logging: Enable logging (default: False)
        log_level: Logging level (default: logging.INFO)
        is_colocated: For some backends, only connect to servers on the same hostname (default: False)
    """
    
    def __init__(self, name: str, server_info: Union[str, dict], logging: bool = False, 
                 log_level: int = logging_.INFO, is_colocated: bool = False):
        self.name = name
        
        # Determine backend type from server info
        backend_type = self._extract_backend_type(server_info)
        
        if backend_type not in DATASTORE_BACKENDS:
            raise ValueError(f"Unsupported DataStore backend type: {backend_type}. "
                           f"Available types: {list(DATASTORE_BACKENDS.keys())}")
        
        # Create the appropriate backend instance
        backend_class = DATASTORE_BACKENDS[backend_type]
        try:
            self._backend = backend_class(name, server_info, logging, log_level, is_colocated)
        except ImportError as e:
            raise ImportError(f"Required dependencies for {backend_type} backend are not available: {e}")
        
        # Expose backend properties for compatibility
        self.logger = self._backend.logger
        self.connections = self._backend.connections
        self.is_colocated = self._backend.is_colocated
        self.config = getattr(self._backend, 'config', {})
    
    def _extract_backend_type(self, server_info: Union[str, dict]) -> str:
        """Extract backend type from server info."""
        if isinstance(server_info, str):
            # Deserialize string to get config
            from .base import BaseServerManager
            deserialized_info = BaseServerManager.deserialize(server_info)
            config = deserialized_info.get("config", {})
        elif isinstance(server_info, dict):
            if "config" in server_info:
                config = server_info["config"]
            else:
                config = server_info
        else:
            raise ValueError("server_info must be str (base64) or dict")
        
        backend_type = config.get("type")
        if not backend_type:
            raise ValueError("Server info must contain 'type' field")
        
        return backend_type
    
    # Delegate all DataStore methods to the backend
    def stage_write(self, key: str, data: Any, persistant: bool = True, client_id: int = 0, is_local: bool = False):
        """Stage data as a key-value pair."""
        return self._backend.stage_write(key, data, persistant, client_id, is_local)
    
    def stage_read(self, key: str, client_id: int = 0, timeout: int = 30, is_local: bool = False):
        """Read staged data using the key."""
        return self._backend.stage_read(key, client_id, timeout, is_local)
    
    def poll_staged_data(self, key: str, client_id: int = 0, is_local: bool = False) -> bool:
        """Check if data for the key is staged."""
        return self._backend.poll_staged_data(key, client_id, is_local)
    
    def clean_staged_data(self, key: str, client_id: int = 0, is_local: bool = False):
        """Clear the staging area for the given key."""
        return self._backend.clean_staged_data(key, client_id, is_local)
    
    def connect(self, other_node):
        """Connect this node to another node."""
        return self._backend.connect(other_node)

    def disconnect(self, other_node):
        """Disconnect this node from another node."""
        return self._backend.disconnect(other_node)

    def get_connections(self):
        """Return a list of connected nodes."""
        return self._backend.get_connections()
    
    def send(self, data, targets: list = None):
        """Send data to all or selected connections."""
        return self._backend.send(data, targets)
        
    def receive(self, senders: list = None):
        """Receive data from connected senders."""
        return self._backend.receive(senders)
    
    @contextlib.contextmanager
    def acquire_lock(self, lock_name: str, acquire_timeout: int = 100, lock_timeout: int = 300):
        """Acquire a distributed lock (if supported by backend)."""
        with self._backend.acquire_lock(lock_name, acquire_timeout, lock_timeout):
            yield
    
    def clean(self):
        """Clean up the datastore."""
        return self._backend.clean()
    
    def flush_logger(self):
        """Flush logger handlers."""
        return self._backend.flush_logger()

    def __repr__(self):
        return f"<DataStore name={self.name}, backend={type(self._backend).__name__}>"


class ServerManager:
    """
    ServerManager wrapper class that automatically delegates to the appropriate backend.
    
    This class maintains the original API while internally routing operations to
    backend-specific implementations based on the server configuration.
    
    Args:
        name: Name of the server instance
        config: Configuration dictionary for the server
        logging: Enable logging (default: False)
        log_level: Logging level (default: logging.INFO)
    """
    
    def __init__(self, name: str, config: ServerConfig, logging: bool = False, log_level: int = logging_.INFO):
        self.name = name
        self.config = config
        
        # Determine backend type from config
        backend_type = config.type
        if not backend_type:
            raise ValueError("Configuration must contain 'type' field")
        
        if backend_type not in SERVERMANAGER_BACKENDS:
            raise ValueError(f"Unsupported ServerManager backend type: {backend_type}. "
                           f"Available types: {list(SERVERMANAGER_BACKENDS.keys())}")
        
        # Create the appropriate backend instance
        backend_class = SERVERMANAGER_BACKENDS[backend_type]
        try:
            self._backend = backend_class(name, config, logging, log_level)
        except ImportError as e:
            raise ImportError(f"Required dependencies for {backend_type} backend are not available: {e}")
        
        # Expose backend properties for compatibility
        self.logger = getattr(self._backend, 'logger', None)
    
    # Delegate all ServerManager methods to the backend
    def start_server(self):
        """Start the server."""
        return self._backend.start_server()
    
    def stop_server(self):
        """Stop the server."""
        return self._backend.stop_server()
    
    def get_server_info(self) -> dict:
        """Get information about the managed server."""
        return self._backend.get_server_info()
    
    def serialize(self) -> str:
        """Serialize the server object for transmission or storage."""
        return self._backend.serialize()

    @classmethod
    def deserialize(cls, serialized_data: str) -> dict:
        """Deserialize server data and return server info."""
        from .base import BaseServerManager
        return BaseServerManager.deserialize(serialized_data)
    
    # Expose any Redis-specific class methods
    @classmethod
    def create_redis_cluster(cls, server_addresses, redis_cli_path="redis-cli", 
                           replicas=0, timeout=30, logging=True):
        """Create a Redis cluster from existing Redis server instances."""
        from .redis import ServerManagerRedis
        return ServerManagerRedis.create_redis_cluster(
            server_addresses, redis_cli_path, replicas, timeout, logging
        )

    def __repr__(self):
        return f"<ServerManager name={self.name}, backend={type(self._backend).__name__}>"