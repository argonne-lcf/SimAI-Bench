"""
DAOS backend implementation for DataStore and ServerManager.

This module provides DAOS-based storage for data staging operations.
Supports both POSIX (via dfuse mount) and KV (via PyDAOS) modes.
"""

import os
import pickle
import tempfile
import time
import zlib
import logging as logging_
from typing import Any, Union
from .base import BaseDataStore, BaseServerManager

# Import for type checking
from SimAIBench.config import DaosServerConfig

try:
    import pydaos
    PYDAOS_AVAILABLE = True
except ImportError:
    PYDAOS_AVAILABLE = False


class DataStoreDaos(BaseDataStore):
    """
    DAOS-based DataStore implementation.
    
    Supports two modes:
    - POSIX: Uses DAOS via dfuse mount (similar to filesystem backend)
    - KV: Uses DAOS key-value store via PyDAOS
    """
    
    def __init__(self, name: str, server_info: Union[str, dict], logging: bool = False, 
                 log_level: int = logging_.INFO, is_colocated: bool = False):
        super().__init__(name, server_info, logging, log_level, is_colocated)
        
        self.daos_cont = None
        self.daos_kv = None
        
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
            self.logger.debug(f"DataStoreDaos {name} initialized with config {self.config}")
            
        # Initialize client connections
        self._setup_client()
    
    def _setup_client(self):
        """Setup DAOS client connection."""
        mode = self.config.get("mode", "posix")
        if mode not in ("posix", "kv"):
            raise ValueError("DAOS mode must be one of {'posix','kv'}")
            
        if mode == "posix":
            # Treat like filesystem sharded directory on a dfuse mount
            if "server_address" not in self.config:
                raise ValueError("For DAOS POSIX mode, 'server-address' must point to a dfuse mount path")
            if "nshards" not in self.config:
                self.config["nshards"] = 64
            dirname = self.config["server_address"]
            for shard in range(self.config["nshards"]):
                shard_dir = os.path.join(dirname, str(shard))
                os.makedirs(shard_dir, exist_ok=True)
        else:
            if not PYDAOS_AVAILABLE:
                raise ImportError("PyDAOS not available in environment")
            # Establish the KV connection
            self._create_daos_kv_client()

    def _create_daos_kv_client(self):
        """Create/connect a DAOS KV client via PyDAOS."""
        if not PYDAOS_AVAILABLE:
            raise ValueError("PyDAOS not available in environment")
            
        pool_label = self.config.get("pool_label") or self.config.get("pool_uuid")
        cont_label = self.config.get("container_label") or self.config.get("container_uuid")
        if not pool_label or not cont_label:
            raise ValueError("DAOS KV mode requires 'pool_label/pool_uuid' and 'container_label/container_uuid'")
            
        self.daos_cont = pydaos.DCont(pool_label, cont_label)
        self.daos_kv = self.daos_cont.dict("default")
    
    def stage_write(self, key: str, data: Any, persistant: bool = True, client_id: int = 0, is_local: bool = False):
        """Stage data as a key-value pair in DAOS."""
        mode = self.config.get("mode", "posix")
        
        if mode == "posix":
            dirname = self.config["server_address"]
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
                raise ValueError("DAOS KV client not initialized")
            serialized_data = pickle.dumps(data)
            self.daos_kv[key] = serialized_data
            if self.logger:
                self.logger.debug(f"Staged data for {key} in DAOS KV")
    
    def stage_read(self, key: str, client_id: int = 0, timeout: int = 30, is_local: bool = False):
        """Read staged data using the key from DAOS."""
        mode = self.config.get("mode", "posix")
        
        if mode == "posix":
            dirname = self.config["server_address"]
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
                    self.logger.debug(f"Read data for {key} from {filename} (DAOS POSIX)")
                return data
        else:
            if not hasattr(self, "daos_kv"):
                raise ValueError("DAOS KV client not initialized")
            if key not in self.daos_kv:
                raise KeyError(f"Key {key} not found in DAOS KV")
            data = pickle.loads(self.daos_kv[key])
            if self.logger:
                self.logger.debug(f"Read data for {key} from DAOS KV")
            return data
    
    def poll_staged_data(self, key: str, client_id: int = 0, is_local: bool = False) -> bool:
        """Check if data for the key is staged in DAOS."""
        mode = self.config.get("mode", "posix")
        
        if mode == "posix":
            dirname = self.config["server_address"]
            h = zlib.crc32(key.encode('utf-8'))
            shard_number = h % self.config["nshards"]
            shard_dir = os.path.join(dirname, str(shard_number))
            filename = os.path.join(shard_dir, f"{key}.pickle")
            return os.path.exists(filename)
        else:
            if not hasattr(self, "daos_kv"):
                raise ValueError("DAOS KV client not initialized")
            return key in self.daos_kv
    
    def clean_staged_data(self, key: str, client_id: int = 0, is_local: bool = False):
        """Clear the staging area for the given key from DAOS."""
        mode = self.config.get("mode", "posix")
        
        if mode == "posix":
            dirname = self.config["server_address"]
            h = zlib.crc32(key.encode('utf-8'))
            shard_number = h % self.config["nshards"]
            shard_dir = os.path.join(dirname, str(shard_number))
            filename = os.path.join(shard_dir, f"{key}.pickle")
            
            if os.path.exists(filename):
                try:
                    os.remove(filename)
                    if self.logger:
                        self.logger.debug(f"Removed file {filename} (DAOS POSIX)")
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"Failed to remove file {filename}: {e}")
                    raise
            else:
                if self.logger:
                    self.logger.error(f"File {filename} does not exist")
                raise ValueError(f"File {filename} does not exist")
        else:
            if not hasattr(self, "daos_kv"):
                raise ValueError("DAOS KV client not initialized")
            if key not in self.daos_kv:
                raise KeyError(f"Key {key} not found in DAOS KV")
            del self.daos_kv[key]
            if self.logger:
                self.logger.debug(f"Deleted key {key} from DAOS KV")
    
    def clean(self):
        """Clean up the DAOS datastore."""
        mode = self.config.get("mode", "posix")
        
        if mode == "posix":
            import shutil
            dirname = self.config.get("server_address")
            if dirname and os.path.exists(dirname):
                shutil.rmtree(dirname)
                if self.logger:
                    self.logger.info(f"Removed DAOS POSIX directory {dirname}")
        else:
            if hasattr(self, "daos_kv") and self.daos_kv:
                try:
                    self.daos_kv.clear()
                    if self.logger:
                        self.logger.info("Cleared DAOS KV store")
                except:
                    pass


class ServerManagerDaos(BaseServerManager):
    """
    DAOS-based ServerManager implementation.
    
    Manages DAOS storage setup. For POSIX mode, creates directory structure.
    For KV mode, validates configuration (actual DAOS pool/container setup is external).
    """
    
    def start_server(self):
        """Setup DAOS storage."""
        self._setup_server()

    def _setup_server(self):
        """Setup DAOS server configuration."""
        if self.logger:
            self.logger.info(f"Setting up DAOS server with mode {getattr(self.config, 'mode', 'posix')}")
        
        mode = getattr(self.config, 'mode', 'posix')
        if mode not in ("posix", "kv"):
            raise ValueError("DAOS mode must be one of {'posix','kv'}")
            
        if mode == "posix":
            if not hasattr(self.config, 'server_address') or not self.config.server_address:
                raise ValueError("For DAOS POSIX mode, 'server_address' must point to a dfuse mount path")
            if not hasattr(self.config, 'nshards') or not self.config.nshards:
                self.config.nshards = 64
            dirname = self.config.server_address
            os.makedirs(dirname, exist_ok=True)
            if self.logger:
                self.logger.info(f"Using DAOS-POSIX mount at {dirname}")
        else:
            if not PYDAOS_AVAILABLE:
                raise ValueError("PyDAOS not available. Install/activate PyDAOS or use DAOS POSIX mode.")
            # Validate configuration
            pool_label = getattr(self.config, 'pool_label', None) or getattr(self.config, 'pool_uuid', None)
            cont_label = getattr(self.config, 'container_label', None) or getattr(self.config, 'container_uuid', None)
            if not pool_label or not cont_label:
                raise ValueError("DAOS KV mode requires 'pool_label/pool_uuid' and 'container_label/container_uuid'")
            if self.logger:
                self.logger.info(f"DAOS KV mode configured with pool={pool_label}, container={cont_label}")
    
    def stop_server(self):
        """Stop the DAOS server (no-op since no process is running)."""
        if self.logger:
            self.logger.info("DAOS server stopped (no process to terminate)")
    
    def get_server_info(self) -> dict:
        """Get information about the DAOS server."""
        info = {
            "name": self.name,
            "type": self.config.type,
            "config": self.config.model_dump(),
            "daos_mode": getattr(self.config, 'mode', 'posix')
        }
        return info