from SimAIBench.datastore import DataStore, ServerManager
from SimAIBench.orchestrator import Orchestrator
from SimAIBench import Simulation
from SimAIBench.config import SystemConfig, OchestratorConfig, server_registry, ServerConfig
from SimAIBench.component import WorkflowComponent


#A simple simulation
def sim(server_info:dict,profile_server_info:dict):
    import numpy as np
    sim = Simulation(server_info=server_info,profile_store=True,profile_server_info=profile_server_info)
    sim.add_kernel("kernel1",run_time=10.0)
    sim.run()
    sim.stage_write("key",np.empty((100,)))
    value = sim.stage_read("key")


def test_profiler():
    ##create configs
    server_config = server_registry.create_config("filesystem")
    system_config = SystemConfig(name="local")
    orchestrator_config = OchestratorConfig(profile=True)
    
    #start the data store server
    datastore_server = ServerManager("global_store",server_config)
    datastore_server.start_server()
    server_info = datastore_server.get_server_info()

    #start the orchestrator
    orchestrator = Orchestrator(system_config,orchestrator_config)
    client = orchestrator.start()
    profile_server_info = orchestrator.get_profile_store()

    ##create workflow component to run
    wc = WorkflowComponent(
        "sim",
        executable=sim,
        type="local",
        args={
            "server_info":server_info,
            "profile_server_info":profile_server_info
        }
    )

    client.submit(wc)

    orchestrator.wait()
    orchestrator.stop()
    datastore_server.stop_server()


if __name__ == "__main__":
    test_profiler()


    