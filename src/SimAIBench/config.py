from pydantic import BaseModel, Field
import multiprocessing as mp

class SystemConfig(BaseModel):
    """Input configuration of the system"""
    name: str
    ncpus: int = mp.cpu_count()
    ngpus: int = 0


class OchestratorConfig(BaseModel):
    name: str = "process-pool"
