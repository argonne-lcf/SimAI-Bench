from SimAIBench import Workflow,Simulation, OchestratorConfig, SystemConfig, server_registry
from SimAIBench import ServerManager, DataStore
import os, json
import logging

def test_workflow():
    ##create a workflow
    my_workflow = Workflow(orchestrator_config=OchestratorConfig(name="process-pool"),
                           system_config=SystemConfig(name="local",ncpus=12,ngpus=0))
    
    ##create server
    server_config = server_registry.create_config("filesystem")
    server = ServerManager("server",server_config)
    server.start_server()
    server_info = server.get_server_info()

    ##register components
    @my_workflow.component(name="sim",type="remote",args={"runcount": 100000, "server_info": server_info})
    def run_simulation(server_info: dict, runcount:int):
        sim = Simulation(name="sim", logging=True, log_level=logging.DEBUG, server_info=server_info)
        ##add two kernels to the simulation
        sim.add_kernel("MatMulSimple2D", run_count=runcount)
        sim.run()
        sim.stage_write("key","value-hello")

    @my_workflow.component(name="sim2", type="local",args={"runcount": 22, "server_info": server_info}, dependencies=["sim"])
    def sim2(server_info: dict, runcount: int):
        sim = Simulation(name="sim2", logging=True, log_level=logging.DEBUG, server_info= server_info)
        sim.add_kernel("MatMulGeneral", run_count=runcount)
        sim.run()
        value = sim.stage_read("key")
        if sim.logger: sim.logger.info(f"Received {value} from sim")

    my_workflow.launch()


if __name__ == "__main__":
    test_workflow()
