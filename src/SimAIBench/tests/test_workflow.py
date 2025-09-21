from SimAIBench import Workflow,Simulation, OchestratorConfig, SystemConfig, server_registry
from SimAIBench import ServerManager, DataStore
from SimAIBench.orchestrator import OrchetratorClient, Orchestrator
from SimAIBench.component import WorkflowComponent
import os, json
import logging
import time

def test_static_workflow():
    ##create server
    server_config = server_registry.create_config("filesystem")
    server = ServerManager("server",server_config)
    server.start_server()
    server_info = server.get_server_info()

    ##create a workflow
    my_workflow = Workflow(orchestrator_config=OchestratorConfig(name="process-pool"),
                           system_config=SystemConfig(name="local",ncpus=12,ngpus=0))
    ##register components
    @my_workflow.component(name="sim",type="remote",args={"runcount": 100, "server_info": server_info}, return_dim=[32,32,32])
    def run_simulation(server_info: dict, runcount:int):
        sim = Simulation(name="sim", logging=True, log_level=logging.DEBUG, server_info=server_info)
        ##add two kernels to the simulation
        sim.add_kernel("MatMulSimple2D", run_count=runcount)
        sim.run()
        sim.stage_write("key","value-hello")
    
    @my_workflow.component(name="sim2",type="remote",args={"runcount": 100, "server_info": server_info}, return_dim=[32,32,32],dependencies=["sim"])
    def run_simulation(server_info: dict, runcount:int):
        sim = Simulation(name="sim", logging=True, log_level=logging.DEBUG, server_info=server_info)
        ##add two kernels to the simulation
        sim.add_kernel("MatMulSimple2D", run_count=runcount)
        sim.run()
        value = sim.stage_read("key")
        sim.logger.info(f"Received {value} from other sim")
    
    my_workflow.launch()
    
    server.stop_server()
    

def test_dynamic_workflow():

    ##A task that generates a chain of tasks.
    ##Each task simply appends the data to the key
    def sim(client:OrchetratorClient, server_info: dict, ntasks: int = 10):
        from SimAIBench.component import WorkflowComponent
        from SimAIBench.datastore import DataStore
        import logging
        import time

        logger = logging.getLogger(__name__)

        ds = DataStore("ds",server_info=server_info)
        ds.stage_write("key","sim")

        ###sub function to run
        def sub_sim(simid:int, ds: DataStore):
            import time
            value = ds.stage_read("key")
            logger.info(f"Got {value}")
            ds.stage_write("key",value+f"-sim{simid}")
            time.sleep(1)
        
        ##create a chain of tasks
        futures = []
        for i in range(ntasks):
            if i > 0:
                dependencies = [f"sim{i-1}"]
            else:
                dependencies = []
            future = client.submit(WorkflowComponent(f"sim{i}",sub_sim,type="local",args={"simid":i,"ds":ds},dependencies=dependencies))
            futures.append(future)
        while not all([future.done() for future in futures]):
            time.sleep(5)
    
    ##start data store
    server_config = server_registry.create_config("filesystem")
    server = ServerManager("server",server_config)
    server.start_server()
    server_info = server.get_server_info()

    #create necessary configs
    orchestrator_config=OchestratorConfig(name="process-pool")
    system_config=SystemConfig(name="local",ncpus=12,ngpus=0)
    
    orchestrator = Orchestrator(system_config,orchestrator_config)

    #start the orchestrator
    client = orchestrator.start()


    ##create a workflow component using sim
    component = WorkflowComponent(
        "sim",
        executable=sim,
        type="local",
        args={"client":client, "server_info":server_info}
    )
    future = client.submit(component)

    while not future.done():
        print("Waiting")
        time.sleep(10)
    orchestrator.wait(timeout=10)
    orchestrator.stop()
    server.stop_server()



if __name__ == "__main__":
    # test_static_workflow()
    test_dynamic_workflow()
