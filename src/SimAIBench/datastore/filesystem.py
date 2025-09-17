"""
Filesystem backend implementation for DataStore and ServerManager.

This module provides filesystem-based storage for data staging operations.
Supports both regular filesystem and node-local temporary storage.
"""

import os
import pickle
import tempfile
import zlib
import time
import fcntl
import errno
import contextlib
import logging as logging_
from typing import Any, Union
from .base import BaseDataStore, BaseServerManager


class DataStoreFilesystem(BaseDataStore):
    """
    Filesystem-based DataStore implementation.
    
    Stores data as pickled files in a sharded directory structure.
    Supports both regular filesystem and node-local (/tmp) storage.
    """
    
    def __init__(self, name: str, server_info: Union[str, dict], logging: bool = False, 
                 log_level: int = logging_.INFO, is_colocated: bool = False):
        super().__init__(name, logging, log_level, is_colocated)
        
        # Parse server info
        if isinstance(server_info, str):
            deserialized_server_info = self.__class__.deserialize(server_info)
            self.config = deserialized_server_info["config"].copy()
        elif isinstance(server_info, dict):
            if "config" in server_info:
                self.config = server_info["config"].copy()
            else:
                self.config = server_info.copy()
        else:
            raise ValueError("server_info must be str (base64) or dict")
            
        if self.logger:
            self.logger.debug(f"DataStoreFilesystem {name} initialized with config {self.config}")
            
        # Initialize client connections
        self._setup_client()
    
    def _setup_client(self):
        """Setup filesystem client - create directory structure."""
        if self.config["type"] == "node-local":
            if "server-address" not in self.config or not self.config["server-address"].startswith("/tmp"):
                self.config["server-address"] = "/tmp"
        elif self.config["type"] == "filesystem":
            if "server-address" not in self.config:
                self.config["server-address"] = os.path.join(os.getcwd(), ".tmp")
                
        if "nshards" not in self.config:
            self.config["nshards"] = 64
            
        dirname = self.config["server-address"]
        for shard in range(self.config["nshards"]):
            shard_dir = os.path.join(dirname, str(shard))
            os.makedirs(shard_dir, exist_ok=True)
    
    def stage_write(self, key: str, data: Any, persistant: bool = True, client_id: int = 0, is_local: bool = False):
        """Stage data as a key-value pair in filesystem."""
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
    
    def stage_read(self, key: str, client_id: int = 0, timeout: int = 30, is_local: bool = False):
        """Read staged data using the key from filesystem."""
        import time
        
        dirname = self.config["server-address"]
        h = zlib.crc32(key.encode('utf-8'))
        shard_number = h % self.config["nshards"]
        shard_dir = os.path.join(dirname, str(shard_number))
        filename = os.path.join(shard_dir, f"{key}.pickle")
        
        time_start = time.time()
        while not os.path.exists(filename):
            if time.time() - time_start > timeout:
                raise TimeoutError(f"Timeout waiting for key {key} after {timeout} seconds")
            time.sleep(0.1)
            
        with open(filename, "rb") as f:
            data = pickle.load(f)
            if self.logger:
                self.logger.debug(f"Read data for {key} from {filename}")
            return data
    
    def poll_staged_data(self, key: str, client_id: int = 0, is_local: bool = False) -> bool:
        """Check if data for the key is staged in filesystem."""
        dirname = self.config["server-address"]
        h = zlib.crc32(key.encode('utf-8'))
        shard_number = h % self.config["nshards"]
        shard_dir = os.path.join(dirname, str(shard_number))
        filename = os.path.join(shard_dir, f"{key}.pickle")
        return os.path.exists(filename)
    
    def clean_staged_data(self, key: str, client_id: int = 0, is_local: bool = False):
        """Clear the staging area for the given key from filesystem."""
        dirname = self.config["server-address"]
        h = zlib.crc32(key.encode('utf-8'))
        shard_number = h % self.config["nshards"]
        shard_dir = os.path.join(dirname, str(shard_number))
        filename = os.path.join(shard_dir, f"{key}.pickle")
        
        if os.path.exists(filename):
            try:
                os.remove(filename)
                if self.logger:
                    self.logger.debug(f"Removed file {filename}")
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Failed to remove file {filename}: {e}")
                raise
        else:
            if self.logger:
                self.logger.error(f"File {filename} does not exist")
            raise ValueError(f"File {filename} does not exist")
    
    def send(self, data, targets: list = None):
        """Send data to all or selected connections via filesystem."""
        targets = targets or self.connections
        if self.logger:
            self.logger.debug(f"Sending data: {data}")
        
        dirname = self.config.get("server-address", os.path.join(os.getcwd(), ".tmp"))
        os.makedirs(dirname, exist_ok=True)
        for target in targets:
            filename = os.path.join(dirname, f"{self.name}_{target.name}_data.pickle")
            with open(filename, "wb") as f:
                pickle.dump(data, f)
            if self.logger:
                self.logger.debug(f"Sent data to {target.name} via {filename}")

    def receive(self, senders: list = None):
        """Receive data from connected senders via filesystem."""
        data = {}
        senders = senders or self.connections
        dirname = self.config.get("server-address", os.path.join(os.getcwd(), ".tmp"))
        
        if not os.path.exists(dirname):
            if self.logger:
                self.logger.error(f"Directory {dirname} does not exist")
            raise AssertionError(f"Directory {dirname} does not exist")
            
        for sender in senders:
            filename = os.path.join(dirname, f"{sender.name}_{self.name}_data.pickle")
            if not os.path.exists(filename):
                if self.logger:
                    self.logger.warning(f"Data file {filename} from {sender.name} not found")
                continue
                    
            with open(filename, "rb") as f:
                data[sender.name] = pickle.load(f)
                
        return data
    
    @contextlib.contextmanager
    def acquire_lock(self, lock_name: str, acquire_timeout: int = 100, lock_timeout: int = 300):
        """Acquire a filesystem-based lock using fcntl."""
        dirname = self.config["server-address"]
        lock_key = f"lock:{lock_name}"
        lock_file_path = os.path.join(dirname, lock_key)
        
        # Ensure directory exists
        os.makedirs(dirname, exist_ok=True)
        
        file_handle = None
        acquired = False
        
        try:
            end = time.time() + acquire_timeout
            
            while time.time() < end:
                try:
                    # Create file if it doesn't exist, open for read/write
                    file_handle = open(lock_file_path, "a+") 
                    fcntl.flock(file_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                    if self.logger:
                        self.logger.debug(f"Lock acquired: {lock_key}")
                    break
                    
                except (OSError, IOError) as e:
                    if file_handle:
                        file_handle.close()
                        file_handle = None
                    
                    if e.errno in (errno.EAGAIN, errno.EACCES):
                        # Lock is held by another process
                        time.sleep(0.1)
                        continue
                    else:
                        # Other error - re-raise
                        raise
            
            if not acquired:
                if self.logger:
                    self.logger.error(f"Lock acquisition timeout: {lock_key}")
                raise TimeoutError(f"Could not acquire lock {lock_name} within {acquire_timeout} seconds")
            
            yield
            
        finally:
            if acquired and file_handle:
                try:
                    fcntl.flock(file_handle.fileno(), fcntl.LOCK_UN)
                    file_handle.close()
                    if self.logger:
                        self.logger.debug(f"Lock released: {lock_key}")
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"Error releasing lock {lock_key}: {e}")
            elif file_handle:
                # Clean up file handle even if lock wasn't acquired
                try:
                    file_handle.close()
                except:
                    pass
    
    def clean(self):
        """Clean up the filesystem datastore."""
        import shutil
        
        dirname = self.config.get("server-address", os.path.join(os.getcwd(), ".tmp"))
        if os.path.exists(dirname):
            shutil.rmtree(dirname)
            if self.logger:
                self.logger.info(f"Removed directory {dirname}")


class ServerManagerFilesystem(BaseServerManager):
    """
    Filesystem-based ServerManager implementation.
    
    Manages filesystem-based storage directories.
    No actual server process is launched for filesystem backends.
    """
    
    def start_server(self):
        """Setup filesystem directories."""
        self._setup_server()

    def _setup_server(self):
        """Setup the filesystem server - create directory structure."""
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
    
    def stop_server(self):
        """Stop the filesystem server (no-op since no process is running)."""
        if self.logger:
            self.logger.info("Filesystem server stopped (no process to terminate)")
    
    def get_server_info(self) -> dict:
        """Get information about the filesystem server."""
        info = {
            "name": self.name,
            "type": self.config["type"],
            "config": self.config.copy()
        }
        return info