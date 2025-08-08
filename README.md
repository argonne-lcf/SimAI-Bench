# SimAI-Bench
Performance benchmarks for coupled AI and simulation workflows on HPC systems.


## Description

SimAI-Bench is a flexible and extensible interface for emulating, benchmarking, and prototyping AI-coupled HPC workflows.
With its easy-to-use Python packages and intuitive API, SimAI-Bench allows users to compose faithful mini-apps of their full end-to-end workflows with diverse simulation and AI components and multiple in situ or in transit data transfer patterns.


<center>

![](utils/Fig1_architecture.png)

</center>

SimAI-Bench is composed of the following modules:

* *Kernel*: The `Kernel` module, adapted from the [workflow-mini-API](https://ieeexplore.ieee.org/document/10701365), provides primitives for key compute, IO, communication, and copy operations. It leverages CuPy and dpnp for compute and copy operations on NVIDIA, AMD and Intel GPUs, mpi4py for communication across ranks, and HDF5 for I/O operations to the disk.
* *torch.nn*: SimAI-Bench leverages the `torch.nn` module in PyTorch to execute ML kernels or build ML models of various architecture.
* *ServerManager* and *DataStore*: The `ServerManager` and `DataStore` classes provide a unified interface for data staging and streaming through various approaches and software libraries, also referred to as data transport backends. SimAI-Bench currently supports data staging through Redis, DragonHPC, the parallel file system, and high-performance node-local flash storage or RAM memory. The `ServerManager` is used to initialize and manage the resources of a particular backend, whereas the `DataStore` provides unified client API which enable the data to be transferred between any workflow component.
* *Simulation*: The `Simulation` class is the main component that emulates traditional HPC simulation components within AI-coupled workflows. It uses the underlying `Kernels` to emulate the computational and communication aspects, and the `DataStore` module to interact with the data staging backends. A simulation is configured using a Python dictionary or a JSON file that specifies all the necessary parameters.
* *AI*: The `AI` class is used to model the machine learning portions of the workflow by leveraging the `torch.nn` and `torch.distributed` modules in PyTorch. Similarly to the `Simulation` class, it can use the `DataStore` API to stage and stream data to other workflow components.
* *Workflow*:  At the highest level, the `Workflow` class provides the functionality to compose a workflow from various `Simulation` and `AI` components and deploy it on an HPC system with the desired data transport strategy.



## Installation

### ALCF Aurora

To install SimAI-Bench on Aurora with all of its required dependencies, clone and install the project locally with `pip` after creating a Python virtual environment

```bash
module load frameworks
python -m venv _simai --system-site-packages
. _simai/bin/activate

git clone https://github.com/argonne-lcf/SimAI-Bench.git
cd SimAI-Bench
pip install .
```

To configure the DragonHPC's high-speed transport agent (HSTA) to use Libfabric on Aurora, execute
```bash
dragon-config -a "ofi-runtime-lib=/opt/cray/libfabric/1.22.0/lib64"
```

## Examples

### Basic, two-component workflow with staging through the file system and a task dependency

This example demonstrates how to use the SimAI-Bench modules to create a simple workflow with two simulations components, sharing data through the file system and with a task dependency.

```python
from SimAIBench import Workflow
from SimAIBench import Simulation
from SimAIBench import ServerManager

# Initialize the server for data staging and the workflow instance
server = ServerManager("server", config=server_config) # the default staging backend is the file system
server.start_server()
info=server.get_server_info()
w = Workflow(sys_info=sys_config)

# Add the first simulation component to the workflow
@w.component(name="sim",
             type="remote", # set remote execution, meaning launching through mpirun/mpiexec, needed for multi-node or MPI components
             args={"info":info})
def run_sim(info=None):
  sim = Simulation(name="sim",
                   server_info=info)
  sim.add_kernel("MatMulSimple2D")
  sim.run()
  sim.stage_write("key","value") # write data to the staging backend

# Add the second simulation component to the workflow
@w.component(name="sim2", 
             type="local", # set local execution, meaning launching through Python mp.Process on the local node
             args={"info":info},
             dependencies=["sim"]) # set a dependency on the first simulation component
def run_sim2(info=None):
  sim = Simulation(name="sim2",
                   server_info=info)
  sim.add_kernel("MatMulGeneral")
  value = sim.stage_read("key") # read data from the staging backend
  sim.run()

# Execute the workflow
w.launch()

# Stop the server
server.stop_server()
```


## Contributing

1. Fork it (<https://github.com/argonne-lcf/SimAI-Bench/fork>)
2. Cline it (`git clone https://github.com/username/SimAI-Bench.git`)
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push -u origin feature/fooBar`)
5. Create a new Pull Request


## Contributors

Harikrishna Tummalapalli, Argonne National Lab, htummalapalli@anl.gov

Riccardo Balin, Argonne National Lab, rbalin@anl.gov

Christine Simpson, Argonne National Lab, csimpson@anl.gov






