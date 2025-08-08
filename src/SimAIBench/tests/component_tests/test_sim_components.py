from SimAIBench.simulation import Simulation
from SimAIBench.kernel import *
import os
import subprocess


def test_simulation_run():
    sim = Simulation()
    sim.add_kernel("MatMulSimple2D",run_time=0.1)
    sim.add_kernel("MatMulGeneral",run_count=10)
    if DPNP_AVAILABLE:
        sim.add_kernel("MatMulSimple2D",device="xpu",run_time=0.1)
        sim.add_kernel("MatMulGeneral",device="xpu",run_count=10)
    sim.run()

def test_run_mpi():
    cmd  = ["mpiexec", "-n", "2", "python3", "-c", 
            "'from mpi4py import MPI; from SimAIBench.simulation import Simulation; comm = MPI.COMM_WORLD; sim = Simulation(comm); sim.add_kernel(\"MatMulSimple2D\",run_count=10); sim.run()'"]
    result = subprocess.run(" ".join(cmd), shell=True, capture_output=True)
    assert result.returncode == 0, f"Error running MPI simulation: {result.stderr.decode()}"
    if DPNP_AVAILABLE:
        cmd  = ["mpiexec", "-n", "2", "python3", "-c", 
            "'from mpi4py import MPI; from SimAIBench.simulation import Simulation; comm = MPI.COMM_WORLD; sim = Simulation(comm); sim.add_kernel(\"MatMulSimple2D\",device=\"xpu\",run_count=10); sim.run()'"]
        result = subprocess.run(" ".join(cmd), shell=True, capture_output=True)
        assert result.returncode == 0, f"Error running MPI simulation: {result.stderr.decode()}"

def test_simulation_init_from_json():
    sim = Simulation()
    sim.init_from_json(os.path.join(os.path.dirname(__file__), "init_sim.json"))
    sim.run()

def test_set_run_time():
    sim = Simulation()
    sim.add_kernel("MatMulSimple2D",run_time=1.0)
    dt = sim.run()
    assert abs(dt-1.0) < 0.1 ##10%
    if DPNP_AVAILABLE:
        sim = Simulation()
        sim.add_kernel("MatMulSimple2D",run_time=1.0,device="xpu")
        dt = sim.run()
        assert abs(dt-1.0) < 0.1,f"target runtime 1.0, actual runtime {dt}"



if __name__ == "__main__":
    test_simulation_run()
    test_simulation_init_from_json()
    test_set_run_time()
    test_run_mpi()
    print("All tests passed!")