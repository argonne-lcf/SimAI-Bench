# SimAI-Bench
Performance benchmarks for coupled simulation and AI workflows on HPC systems

## Description

The goal of SimAI-Bench is to host a series of micro, mini, and full benchmarks for various coupled simulation and AI/ML workflow motifs designed for leadership HPC systems.
These benchmarks may be used to evaluate the performance and scalability of different workflow libraries or hardware for a given workflow motif.

### Online Training of ML Surrogates



![](utils/surrogate_workflow.png)

## Software Dependencies

### AI/ML

* PyTorch
* PyTorch Geometric and PyTorch Cluster

### Workflows

* [SmartSim](https://github.com/CrayLabs/SmartSim) and [SmartRedis](https://github.com/CrayLabs/SmartRedis)
* [Dragon](https://github.com/DragonHPC/dragon)

### Other

* [MPIPartition](https://github.com/ArgonneCPAC/MPIPartition)


## Installation

### ALCF Polaris


## Usage Example

### Online Training of ML Surrogates


## Release History

* 0.0.1
    * Added an online training workflow for ML surrogate models implemented with SmartSim and Dragon
    * Tested on ALCF Polaris
    * Work in progress


## Contributing

1. Fork it (<https://github.com/argonne-lcf/SimAI-Bench/fork>)
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request


## Contributors

Riccardo Balin, Argonne National Lab, rbalin@anl.gov
Shivam Barwey, Argonne National Lab, sbarwey@anl.gov






