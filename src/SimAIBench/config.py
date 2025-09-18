from pydantic import BaseModel, Field
import multiprocessing as mp
from typing import Literal, Dict, Any, Type
from difflib import get_close_matches


class SystemConfig(BaseModel):
    """Input configuration of the system"""
    name: str
    ncpus: int = mp.cpu_count()
    ngpus: int = 0


class OchestratorConfig(BaseModel):
    name: str = "process-pool"

class ServerConfig(BaseModel):
    type: str
    server_address: str


class ServerConfigRegistry:
    """Registry for server configuration classes"""
    
    def __init__(self):
        self._configs: Dict[str, Type[ServerConfig]] = {}
    
    def register(self, server_type: str):
        """Decorator to register server config classes"""
        def decorator(cls: Type[ServerConfig]):
            self._configs[server_type] = cls
            return cls
        return decorator
    
    def get_config_class(self, server_type: str) -> Type[ServerConfig]:
        """Get the appropriate server config class for a given type"""
        if server_type not in self._configs:
            # Find the closest match using edit distance
            available_types = list(self._configs.keys())
            suggestions = get_close_matches(server_type, available_types, n=1, cutoff=0.6)
            
            error_msg = f"Unknown server type: {server_type}. "
            if suggestions:
                error_msg += f"Did you mean: {', '.join(suggestions)}?"
            else:
                error_msg += f"Available types: {', '.join(available_types)}"
            
            raise ValueError(error_msg)
        return self._configs[server_type]
    
    def create_config(self, server_type: str, **kwargs) -> ServerConfig:
        """Create a server config instance for a given type"""
        config_class = self.get_config_class(server_type)
        return config_class(**kwargs)
    
    def list_types(self) -> list[str]:
        """List all available server types"""
        return list(self._configs.keys())
    
    def is_registered(self, server_type: str) -> bool:
        """Check if a server type is registered"""
        return server_type in self._configs


# Create global registry instance
server_registry = ServerConfigRegistry()

# Register existing server configs
@server_registry.register("filesystem")
@server_registry.register("node-local")
class FilesystemServerConfig(ServerConfig):
    type: Literal["filesystem", "node-local"] = "filesystem"
    server_address: str = "./.tmp"
    nshards: int = 64

@server_registry.register("redis")
class RedisServerConfig(ServerConfig):
    type: Literal["redis"] = "redis"
    server_address: str = "localhost:6379"
    redis_server_exe: str = "redis-server"
    is_clustered: bool = False

@server_registry.register("dragon")
class DragonServerConfig(ServerConfig):
    type: Literal["dragon"] = "dragon"
    server_address: str = "localhost:8888"
    server_options: Dict[str, Any] = Field(default_factory=dict)

@server_registry.register("daos")
class DaosServerConfig(ServerConfig):
    type: Literal["daos"] = "daos"
    server_address: str = "/path/to/dfuse/mount"
    mode: Literal["posix", "kv"] = "posix"
    nshards: int = 64
