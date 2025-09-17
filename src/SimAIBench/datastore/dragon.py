"""
Dragon backend implementation for DataStore and ServerManager.

This module provides Dragon distributed dictionary-based storage for data staging operations.
Supports both clustered and non-clustered Dragon dictionary configurations.
"""

import os
import time
import logging as logging_
from typing import Any, Union, List
from .base import BaseDataStore, BaseServerManager

try:
    import dragon
    from dragon.data.ddict import DDict
    from dragon.native.process_group import ProcessGroup
    from dragon.native.process import ProcessTemplate, Process
    from dragon.infrastructure.policy import Policy as DragonPolicy
    DRAGON_AVAILABLE = True
except ImportError:
    DRAGON_AVAILABLE = False


class DataStoreDragon(BaseDataStore):
    """
    Dragon-based DataStore implementation.
    
    Uses Dragon distributed dictionaries for high-performance data staging
    across distributed systems.
    """
    
    def __init__(self, name: str, server_info: Union[str, dict], logging: bool = False, 
                 log_level: int = logging_.INFO, is_colocated: bool = False):
        super().__init__(name, server_info, logging, log_level, is_colocated)
        
        if not DRAGON_AVAILABLE:
            raise ImportError("Dragon library not available")
        
        self.dragon_dicts = None
        self.local_ddicts = None
        
        # Parse server info
        if isinstance(server_info, str):
            deserialized_server_info = self.__class__.deserialize(server_info)
            self.config = deserialized_server_info["config"].copy()
            # Add Dragon-specific info if needed
            if deserialized_server_info.get("type") == "dragon" and "serial_dragon_dict" in deserialized_server_info:
                self.config["server-obj"] = deserialized_server_info["serial_dragon_dict"]
        elif isinstance(server_info, dict):
            if "config" in server_info:
                self.config = server_info["config"].copy()
                # Add Dragon-specific info if needed
                if server_info.get("type") == "dragon":
                    self.config["server-obj"] = server_info.get("serial_dragon_dict")
            else:
                self.config = server_info.copy()
        else:
            raise ValueError("server_info must be str (base64) or dict")
            
        if self.logger:
            self.logger.debug(f"DataStoreDragon {name} initialized with config {self.config}")
            
        # Initialize client connections
        self._setup_client()
    
    def _setup_client(self):
        """Setup Dragon client connection."""
        self.dragon_dicts, self.local_ddicts = self._create_dragon_client()
    
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
                    self.logger.debug("Setting up colocated Dragon client")
                from dragon.utils import host_id
                current_host = host_id()
                manager_nodes = ddict.manager_nodes
                for manager_id, manager_node in enumerate(manager_nodes):
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
                if not dragon_dicts:
                    raise ValueError("No colocated Dragon dictionaries found")
            else:
                dragon_dicts = []
                local_ddicts = []
                for obj in self.config["server-obj"]:
                    if isinstance(obj, bytes):
                        ddict = DDict.attach(obj, trace=True)
                    elif isinstance(obj, str):
                        ddict = DDict.attach(obj, trace=True)
                    elif isinstance(obj, DDict):
                        ddict = obj
                    else:
                        raise ValueError("Unknown server-obj type for Dragon client")
                    dragon_dicts.append(ddict)
        
        return dragon_dicts, local_ddicts
    
    def stage_write(self, key: str, data: Any, persistant: bool = True, client_id: int = 0, is_local: bool = False):
        """Stage data as a key-value pair in Dragon dictionary."""
        if not DRAGON_AVAILABLE:
            raise RuntimeError("Dragon is not available")
            
        if is_local:
            assert client_id < len(self.local_ddicts), "client_id must be < number of local Dragon dictionaries"
            ddict = self.local_ddicts[client_id]
        else:
            assert client_id < len(self.dragon_dicts), "client_id must be < number of Dragon dictionaries"
            ddict = self.dragon_dicts[client_id]

        try:
            wait_for_keys = self.config.get("server-options", {}).get("wait_for_keys", None)
            if wait_for_keys is not None and wait_for_keys == True:
                ddict.pput(key, data)
                if self.logger:
                    self.logger.debug(f"Staged data for {key} in Dragon dictionary (persistent)")
            else:
                ddict[key] = data
                if self.logger:
                    self.logger.debug(f"Staged data for {key} in Dragon dictionary")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to stage data for {key}: {e}")
            raise
    
    def stage_read(self, key: str, client_id: int = 0, timeout: int = 30, is_local: bool = False):
        """Read staged data using the key from Dragon dictionary."""
        if not DRAGON_AVAILABLE:
            raise RuntimeError("Dragon is not available")
            
        if is_local:
            assert client_id < len(self.local_ddicts), "client_id must be < number of local Dragon dictionaries"
            ddict = self.local_ddicts[client_id]
        else:
            assert client_id < len(self.dragon_dicts), "client_id must be < number of Dragon dictionaries"
            ddict = self.dragon_dicts[client_id]

        try:
            data = ddict[key]
            if self.logger:
                self.logger.debug(f"Read data for {key} from Dragon dictionary")
            return data
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to read data for {key}: {e}")
            raise
    
    def poll_staged_data(self, key: str, client_id: int = 0, is_local: bool = False) -> bool:
        """Check if data for the key is staged in Dragon dictionary."""
        if not DRAGON_AVAILABLE:
            raise RuntimeError("Dragon is not available")
            
        if is_local:
            assert client_id < len(self.local_ddicts), "client_id must be < number of local Dragon dictionaries"
            ddict = self.local_ddicts[client_id]
        else:
            assert client_id < len(self.dragon_dicts), "client_id must be < number of Dragon dictionaries"
            ddict = self.dragon_dicts[client_id]

        try:
            return key in ddict.keys()
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to poll data for {key}: {e}")
            return False
    
    def clean_staged_data(self, key: str, client_id: int = 0, is_local: bool = False):
        """Clear the staging area for the given key from Dragon dictionary."""
        if not DRAGON_AVAILABLE:
            raise RuntimeError("Dragon is not available")
            
        if is_local:
            assert client_id < len(self.local_ddicts), "client_id must be < number of local Dragon dictionaries"
            ddict = self.local_ddicts[client_id]
        else:
            assert client_id < len(self.dragon_dicts), "client_id must be < number of Dragon dictionaries"
            ddict = self.dragon_dicts[client_id]

        try:
            ddict.pop(key)
            if self.logger:
                self.logger.debug(f"Deleted key {key} from Dragon dictionary")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to delete key {key}: {e}")
            raise
    
    def clean(self):
        """Clean up the Dragon datastore."""
        if self.dragon_dicts:
            for ddict in self.dragon_dicts:
                try:
                    ddict.clear()
                except:
                    pass
            if self.logger:
                self.logger.info("Cleared Dragon dictionaries")


class ServerManagerDragon(BaseServerManager):
    """
    Dragon-based ServerManager implementation.
    
    Manages Dragon distributed dictionary setup and lifecycle.
    Supports both clustered and non-clustered configurations.
    """
    
    def __init__(self, name: str, config: dict, logging: bool = False, log_level: int = logging_.INFO):
        super().__init__(name, config, logging, log_level)
        self.dragon_dict = None
        
        if not DRAGON_AVAILABLE:
            raise ImportError("Dragon library not available")
    
    def start_server(self):
        """Start Dragon dictionary server."""
        self._setup_server()

    def _setup_server(self):
        """Setup Dragon dictionary server."""
        if self.logger:
            self.logger.info(f"Setting up Dragon server on {self.config.get('server-address', 'unknown')}")
        
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
            policy = DragonPolicy(placement=DragonPolicy.Placement.HOST_NAME, host_name=node)
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
                dd = DDict(managers_per_node=1, n_nodes=1, total_mem=total_mem)
                with open(os.path.join("/tmp", "local_store.pickle"), "w") as f:
                    f.write(dd.serialize())

            if self.logger:
                self.logger.info("Creating local Dragon dictionary using Dragon utils")
                total_mem_gb = self.config.get('server-options', {}).get('total_mem', 1024*1024*1024*5) / (1024*1024*1024)
                self.logger.info(f"Total memory for local Dragon dictionary: {total_mem_gb} GB")
                
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
    
    def _is_dragon_server_running(self, ddict):
        """Check if the Dragon dictionary server is running."""
        try:
            manager_nodes = ddict.manager_nodes
            for manager_id, manager_node in enumerate(manager_nodes):
                local_ddict = ddict.manager(manager_id)
                local_ddict.pput("test_key", "test_value")
                value = local_ddict["test_key"]
            return True
        except Exception as e:
            return False
    
    def stop_server(self):
        """Stop the Dragon dictionary server."""
        if self.logger:
            self.logger.info("Stopping Dragon server!")
        
        if self.dragon_dict:
            if isinstance(self.dragon_dict, list):
                for d in self.dragon_dict:
                    try:
                        d.destroy()
                    except:
                        pass
            else:
                try:
                    self.dragon_dict.destroy()
                except:
                    pass
            if self.logger:
                self.logger.info("Dragon dictionary destroyed")
        
        if self.logger:
            self.logger.info("Done stopping Dragon server!")
    
    def get_server_info(self) -> dict:
        """Get information about the Dragon server."""
        info = {
            "name": self.name,
            "type": self.config["type"],
            "config": self.config.copy(),
            "dragon_dict": self.dragon_dict
        }
        
        if self.dragon_dict:
            if isinstance(self.dragon_dict, list):
                info["serial_dragon_dict"] = [d.serialize() if hasattr(d, 'serialize') else None for d in self.dragon_dict]
            else:
                info["serial_dragon_dict"] = self.dragon_dict.serialize() if hasattr(self.dragon_dict, 'serialize') else None
        else:
            info["serial_dragon_dict"] = None

        return info