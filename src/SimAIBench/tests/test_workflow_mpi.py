from SimAIBench import Workflow,Simulation
import os, json
import logging

class RC:
    def __init__(self, data:int):
        self.data = data

def test_workflow():
    my_workflow = Workflow(launcher={"mode":"high throughput"})
    @my_workflow.component(name="sim",args={"runcount": RC(100000)},ppn=6)
    def run_simulation(runcount:RC=RC(10000)):
        from mpi4py import MPI
        MPI.Init()
        comm = MPI.COMM_WORLD
        comm.barrier()
        rank = comm.Get_rank()
        size = comm.Get_size()
        sim = Simulation(name="sim", logging=True, log_level=logging.DEBUG,comm=comm)
        ##add two kernels to the simulation
        sim.add_kernel("MatMulSimple2D", run_count=runcount.data)
        sim.run()
        MPI.Finalize()

    @my_workflow.component(name="sim2",args={"runcount": 22},ppn=7)
    def sim2(runcount=10):
        from mpi4py import MPI
        MPI.Init()
        comm = MPI.COMM_WORLD
        comm.barrier()
        rank = comm.Get_rank()
        size = comm.Get_size()
        sim = Simulation(name="sim2", logging=True, log_level=logging.DEBUG)
        sim.add_kernel("MatMulGeneral", run_count=runcount)
        sim.run()
        MPI.Finalize()

    # my_workflow.register_component(name="sim3",executable="echo 'Hello from sim3'", type="local",dependencies=["sim","sim2"])
    my_workflow.launch()


if __name__ == "__main__":
    test_workflow()
