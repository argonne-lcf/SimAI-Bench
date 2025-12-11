"""
SimAI-Bench DataStore Module

This module provides a unified interface for data storage and server management
across multiple backend implementations including filesystem, Redis, Dragon, and DAOS.

The module uses wrapper classes that automatically delegate to the appropriate backend
based on configuration.

Usage Examples:

1. Standard usage (recommended):
```python
from SimAIBench.datastore import DataStore, ServerManager

# Create server
server = ServerManager("my_server", {"type": "redis", "server-address": "localhost:6379"})
server.start_server()

# Create client
datastore = DataStore("my_client", server.get_server_info())
datastore.stage_write("key1", {"data": "value"})
```

2. Inheritance usage (for Simulation and AI classes):
```python
from SimAIBench.simulation import Simulation
from SimAIBench.training import AI

# These classes inherit from DataStore and get all its functionality
sim = Simulation(name="my_sim", server_info={"type": "filesystem"})
ai = AI(name="my_ai", server_info={"type": "redis", "server-address": "localhost:6379"})
```

3. Direct backend access (for advanced users):
```python
from SimAIBench.datastore.redis import DataStoreRedis, ServerManagerRedis
from SimAIBench.datastore.filesystem import DataStoreFilesystem

# Use specific implementations directly
redis_store = DataStoreRedis("client", server_info)
```
"""

# Import wrapper classes (main API)
from .wrapper import DataStore, ServerManager

# Import base classes for typing and extension
from .base import BaseDataStore, BaseServerManager

# Import specific backend implementations for direct access
from .filesystem import DataStoreFilesystem, ServerManagerFilesystem
from .redis import DataStoreRedis, ServerManagerRedis
from .dragon import DataStoreDragon, ServerManagerDragon
from .daos import DataStoreDaos, ServerManagerDaos

# Main exports - wrapper classes provide the primary API
__all__ = [
    # Primary API (wrapper classes)
    "DataStore",
    "ServerManager",
    
    # Base classes for typing
    "BaseDataStore",
    "BaseServerManager",
    
    # Specific backend implementations (direct access)
    "DataStoreFilesystem",
    "ServerManagerFilesystem",
    "DataStoreRedis", 
    "ServerManagerRedis",
    "DataStoreDragon",
    "ServerManagerDragon",
    "DataStoreDaos",
    "ServerManagerDaos",
]