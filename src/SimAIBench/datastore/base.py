"""
Abstract base classes for DataStore and ServerManager.

This module defines the common interfaces that all backend implementations must follow.
"""

from abc import ABC, abstractmethod
from typing import Union, List, Dict, Optional, Any
import logging as logging_
import contextlib

from SimAIBench.config import ServerConfig


class BaseDataStore(ABC):
    """
    Abstract base class for DataStore implementations.
    
    Handles client-side data operations including read, write, send, receive, and staging.
    All backend implementations must inherit from this class and implement the abstract methods.
    """
    
    def __init__(self, name: str, server_info: Union[str, dict], logging: bool = False, 
                 log_level: int = logging_.INFO, is_colocated: bool = False):
        """
        Initialize the DataStore.
        
        Args:
            name: Name of the DataStore instance
            server_info: Server information (serialized string or dict)
            logging: Enable logging (default: False)
            log_level: Logging level (default: logging.INFO)
            is_colocated: For some backends, only connect to servers on the same hostname (default: False)
        """
        self.name = name
        self.connections = []
        self.is_colocated = is_colocated
        
        # Setup logging
        if logging:
            self.logger = logging_.getLogger(f"{self.__class__.__name__}_{name}")
            self.logger.setLevel(log_level)
        else:
            self.logger = None
    
    @abstractmethod
    def _setup_client(self):
        """Setup client connections based on configuration."""
        pass
    
    @abstractmethod
    def stage_write(self, key: str, data: Any, persistant: bool = True, client_id: int = 0, is_local: bool = False):
        """Stage data as a key-value pair."""
        pass
    
    @abstractmethod
    def stage_read(self, key: str, client_id: int = 0, timeout: int = 30, is_local: bool = False):
        """Read staged data using the key."""
        pass
    
    @abstractmethod
    def poll_staged_data(self, key: str, client_id: int = 0, is_local: bool = False) -> bool:
        """Check if data for the key is staged."""
        pass
    
    @abstractmethod
    def clean_staged_data(self, key: str, client_id: int = 0, is_local: bool = False):
        """Clear the staging area for the given key."""
        pass
    
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

    def get_connections(self):
        """Return a list of connected nodes."""
        return self.connections
    
    def send(self, data, targets: list = None):
        """Send data to all or selected connections."""
        targets = targets or self.connections
        if self.logger:
            self.logger.debug(f"Sending data: {data}")
        # Default implementation - can be overridden by backends that support it
        raise NotImplementedError("Send operation not supported by this backend")
        
    def receive(self, senders: list = None):
        """Receive data from connected senders."""
        senders = senders or self.connections
        # Default implementation - can be overridden by backends that support it
        raise NotImplementedError("Receive operation not supported by this backend")
    
    @contextlib.contextmanager
    def acquire_lock(self, lock_name: str, acquire_timeout: int = 100, lock_timeout: int = 300):
        """Acquire a distributed lock (if supported by backend)."""
        # Default implementation - backends can override if they support locking
        yield
    
    def clean(self):
        """Clean up the datastore."""
        if self.logger:
            self.logger.info(f"Cleaning up {self.__class__.__name__}")
    
    def flush_logger(self):
        """Flush logger handlers."""
        if self.logger:
            for handler in self.logger.handlers:
                handler.flush()
    
    def dump(self):
        if self.logger:
            self.logger.warning("Server dump not implemented")
        pass

    def __repr__(self):
        return f"<{self.__class__.__name__} name={self.name}>"


class BaseServerManager(ABC):
    """
    Abstract base class for ServerManager implementations.
    
    Manages server setup and teardown for different backend types.
    Responsible for launching, monitoring, and stopping server processes.
    """
    
    def __init__(self, name: str, config: ServerConfig, logging: bool = False, log_level: int = logging_.INFO):
        """
        Initialize the ServerManager.
        
        Args:
            name: Name of the server instance
            config: Configuration object for the server
            logging: Enable logging (default: False)
            log_level: Logging level (default: logging.INFO)
        """
        self.name = name
        self.config = config
        
        # Setup logging
        if logging:
            self.logger = logging_.getLogger(f"{self.__class__.__name__}_{name}")
            self.logger.setLevel(log_level)
        else:
            self.logger = None
    
    @abstractmethod
    def start_server(self):
        """Start the server."""
        pass
    
    @abstractmethod
    def stop_server(self):
        """Stop the server."""
        pass
    
    @abstractmethod
    def get_server_info(self) -> dict:
        """Get information about the managed server."""
        pass
    
    def serialize(self) -> str:
        """
        Serialize the server object for transmission or storage.
        
        Returns:
            str: Base64-encoded serialized server info
        """
        try:
            import base64
            try:
                import cloudpickle
            except ImportError:
                import pickle as cloudpickle
                
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
    def deserialize(cls, serialized_data: str) -> dict:
        """
        Deserialize server data and return server info.
        
        Args:
            serialized_data: Base64-encoded serialized server data
            
        Returns:
            dict: Server info dictionary
        """
        try:
            import base64
            try:
                import cloudpickle
            except ImportError:
                import pickle as cloudpickle
                
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