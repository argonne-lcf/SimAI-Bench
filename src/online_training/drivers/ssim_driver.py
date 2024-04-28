# general imports
import os
import sys 
from omegaconf import DictConfig, OmegaConf
import hydra

# smartsim and smartredis imports
from smartsim import Experiment
from smartsim.settings import RunSettings, PalsMpiexecSettings


## Define function to parse node list
def parseNodeList(fname):
    with open(fname) as file:
        nodelist = file.readlines()
        nodelist = [line.rstrip() for line in nodelist]
        nodelist = [line.split('.')[0] for line in nodelist]
    nNodes = len(nodelist)
    return nodelist, nNodes


## Co-located DB launch
def launch_coDB(cfg, nodelist, nNodes):
    # Print nodelist
    if (nodelist is not None):
        print(f"\nRunning on {nNodes} total nodes")
        print(nodelist, "\n")
        hosts = ','.join(nodelist)

    # Initialize the SmartSim Experiment
    PORT = cfg.database.port
    exp = Experiment(cfg.database.exp_name, launcher=cfg.database.launcher)

    # Set the run settings, including the client executable and how to run it
    client_exe = cfg.sim.executable
    extension = client_exe.split(".")[-1]
    if extension=="py":
        exe_args = client_exe
        client_exe = "python"
    else:
        exe_args = None
    if (cfg.database.launcher=='local'):
        sim_settings = RunSettings(client_exe,
                           exe_args=exe_args,
                           run_command='mpirun',
                           run_args={"-n" : cfg.run_args.simprocs},
                           env_vars=None)
        sim_settings.add_exe_args(cfg.sim.arguments)
    elif (cfg.database.launcher=='pals'):
        sim_settings = PalsMpiexecSettings(
                           client_exe,
                           exe_args=exe_args,
                           env_vars={'MPICH_OFI_CXI_PID_BASE':str(0)})
        sim_settings.set_tasks(cfg.run_args.simprocs)
        sim_settings.set_tasks_per_node(cfg.run_args.simprocs_pn)
        sim_settings.set_hostlist(hosts)
        sim_settings.set_cpu_binding_type(cfg.run_args.sim_cpu_bind)
        sim_settings.add_exe_args(cfg.sim.arguments)
        if (cfg.sim.affinity):
            sim_settings.set_gpu_affinity_script(cfg.sim.affinity,
                                                 cfg.run_args.simprocs_pn)

    # Create the co-located database model
    colo_model = exp.create_model("sim", sim_settings)
    kwargs = {
        'maxclients': 100000,
        'threads_per_queue': 4, # set to 4 for improved performance
        'inter_op_parallelism': 1,
        'intra_op_parallelism': 1,
        'cluster-node-timeout': 30000,
        }
    db_bind = None if cfg.run_args.db_cpu_bind=='None' else cfg.run_args.db_cpu_bind
    if (cfg.database.network_interface=='uds'):
        colo_model.colocate_db_uds(
                db_cpus=cfg.run_args.dbprocs_pn,
                custom_pinning=db_bind,
                debug=False,
                **kwargs
                )
    else:
        colo_model.colocate_db_tcp(
                port=PORT,
                ifname=cfg.database.network_interface,
                db_cpus=cfg.run_args.dbprocs_pn,
                custom_pinning=db_bind,
                debug=False,
                **kwargs
                )

    # Start the co-located model
    block = False if cfg.train.executable else True
    print("Launching simulation and SmartSim co-located DB ... ")
    exp.start(colo_model, block=block, summary=False)
    print("Done\n")

    # Setup and launch the training script
    if (cfg.train.executable):
        ml_exe = cfg.train.executable
        if (cfg.train.config_path): ml_exe += f' --config-path {cfg.train.config_path}'
        if (cfg.train.config_name): ml_exe += f' --config-name {cfg.train.config_name}'
        ml_exe = ml_exe + f' ppn={cfg.run_args.mlprocs_pn}' \
                        + f' online.backend=smartredis' \
                        + f' online.simprocs={cfg.run_args.simprocs}' \
                        + f' online.smartredis.db_launch={cfg.database.deployment}' \
                        + f' online.smartredis.db_nodes={cfg.run_args.db_nodes}'
        SSDB = colo_model.run_settings.env_vars['SSDB']
        if (cfg.database.launcher=='local'):
            ml_settings = RunSettings('python',
                                      exe_args=ml_exe,
                                      run_command='mpirun',
                                      run_args={"-n" : cfg.run_args.mlprocs},
                                      env_vars={'SSDB' : SSDB})
        elif (cfg.database.launcher=='pals'):
            ml_settings = PalsMpiexecSettings('python', 
                                              exe_args=ml_exe,
                                              run_args=None,
                                              env_vars={'SSDB':SSDB, 'MPICH_OFI_CXI_PID_BASE':str(1)})
            ml_settings.set_tasks(cfg.run_args.mlprocs)
            ml_settings.set_tasks_per_node(cfg.run_args.mlprocs_pn)
            ml_settings.set_hostlist(hosts)
            ml_settings.set_cpu_binding_type(cfg.run_args.ml_cpu_bind)
            if (cfg.train.affinity):
                ml_settings.set_gpu_affinity_script(cfg.train.affinity,
                                                    cfg.run_args.mlprocs_pn,
                                                    cfg.run_args.simprocs_pn)

        print("Launching training script ... ")
        ml_model = exp.create_model("train_model", ml_settings)
        exp.start(ml_model, block=True, summary=False)
        print("Done\n")


## Clustered DB launch
def launch_clDB(cfg, nodelist, nNodes):
    # Split nodes between the components
    dbNodes_list = None
    if (nodelist is not None):
        dbNodes = ','.join(nodelist[0: cfg.run_args.db_nodes])
        dbNodes_list = nodelist[0: cfg.run_args.db_nodes]
        simNodes = ','.join(nodelist[cfg.run_args.db_nodes: \
                                 cfg.run_args.db_nodes + cfg.run_args.sim_nodes])
        mlNodes = ','.join(nodelist[cfg.run_args.db_nodes + cfg.run_args.sim_nodes: \
                                cfg.run_args.db_nodes + cfg.run_args.sim_nodes + \
                                cfg.run_args.ml_nodes])
        print(f"Database running on {cfg.run_args.db_nodes} nodes:")
        print(dbNodes)
        print(f"Simulatiom running on {cfg.run_args.sim_nodes} nodes:")
        print(simNodes)
        print(f"ML running on {cfg.run_args.ml_nodes} nodes:")
        print(mlNodes, "\n")

    # Set up database and start it
    PORT = cfg.database.port
    exp = Experiment(cfg.database.exp_name, launcher=cfg.database.launcher)
    runArgs = {"np": 1, "ppn": 1, "cpu-bind": "numa"}
    kwargs = {
        'maxclients': 100000,
        'threads_per_queue': 4, # set to 4 for improved performance
        'inter_op_parallelism': 1,
        'intra_op_parallelism': cfg.run_args.cores_pn,
        'cluster-node-timeout': 30000,
        }
    if (cfg.database.launcher=='local'): run_command = 'mpirun'
    elif (cfg.database.launcher=='pals'): run_command = 'mpiexec'
    db = exp.create_database(port=PORT, 
                             batch=False,
                             db_nodes=cfg.run_args.db_nodes,
                             run_command=run_command,
                             interface=OmegaConf.to_object(cfg.database.network_interface), 
                             hosts=dbNodes_list,
                             run_args=runArgs,
                             single_cmd=True,
                             **kwargs
                            )
    exp.generate(db)
    print("Starting database ...")
    exp.start(db)
    print("Done\n")

    # Set the run settings, including the client executable and how to run it
    client_exe = cfg.sim.executable
    extension = client_exe.split(".")[-1]
    if extension=="py":
        exe_args = client_exe
        client_exe = "python"
    else:
        exe_args = None
    if (cfg.database.launcher=='local'):
        sim_settings = RunSettings(client_exe,
                                   exe_args=exe_args,
                                   run_command='mpirun',
                                   run_args={"-n" : cfg.run_args.simprocs},
                                   env_vars=None)
    elif (cfg.database.launcher=='pals'):
        sim_settings = PalsMpiexecSettings(client_exe,
                                           exe_args=exe_args,
                                           env_vars=None)
        sim_settings.set_tasks(cfg.run_args.simprocs)
        sim_settings.set_tasks_per_node(cfg.run_args.simprocs_pn)
        sim_settings.set_hostlist(simNodes)
        sim_settings.set_cpu_binding_type(cfg.run_args.sim_cpu_bind)
        sim_settings.add_exe_args(cfg.sim.arguments)
        if (cfg.sim.affinity):
            sim_settings.set_gpu_affinity_script(cfg.sim.affinity,
                                                 cfg.run_args.simprocs_pn)

    # Start the simulation
    print("Launching the simulation  ...")
    sim_exp = exp.create_model("sim", sim_settings)
    block = False if cfg.train.executable else True
    exp.start(sim_exp, summary=False, block=block)
    print("Done\n")
    
    # Set up and launch the training script
    if (cfg.train.executable):
        ml_exe = cfg.train.executable
        if (cfg.train.config_path): ml_exe += f' --config-path {cfg.train.config_path}'
        if (cfg.train.config_name): ml_exe += f' --config-name {cfg.train.config_name}'
        ml_exe = ml_exe + f' ppn={cfg.run_args.mlprocs_pn}' \
                        + f' online.backend=smartredis' \
                        + f' online.simprocs={cfg.run_args.simprocs}' \
                        + f' online.smartredis.db_launch={cfg.database.deployment}' \
                        + f' online.smartredis.db_nodes={cfg.run_args.db_nodes}'
        if (cfg.database.launcher=='local'):
            ml_settings = RunSettings('python',
                                      exe_args=ml_exe,
                                      run_command='mpirun',
                                      run_args={"-n" : cfg.run_args.mlprocs},
                                      env_vars=None)
        elif (cfg.database.launcher=='pals'):
            ml_settings = PalsMpiexecSettings('python', 
                                              exe_args=ml_exe, 
                                              run_args=None,
                                              env_vars=None)
            ml_settings.set_tasks(cfg.run_args.mlprocs)
            ml_settings.set_tasks_per_node(cfg.run_args.mlprocs_pn)
            ml_settings.set_hostlist(mlNodes)
            ml_settings.set_cpu_binding_type(cfg.run_args.ml_cpu_bind)
            if (cfg.train.affinity):
                ml_settings.set_gpu_affinity_script(cfg.train.affinity,
                                                    cfg.run_args.mlprocs_pn,
                                                    0)

        print("Launching the training client ...")
        ml_model = exp.create_model("train_model", ml_settings)
        exp.start(ml_model, block=True, summary=False)
        print("Done\n")

    # Stop database
    print("Stopping the Orchestrator ...")
    exp.stop(db)
    print("Done\n")


## Main function
@hydra.main(version_base=None, config_path="./conf", config_name="ssim_config")
def main(cfg: DictConfig):
    # Get nodes of this allocation (job)
    nodelist = nNodes = None
    if (cfg.database.launcher=='pals'):
        hostfile = os.getenv('PBS_NODEFILE')
        nodelist, nNodes = parseNodeList(hostfile)

    # Call appropriate launcher
    if (cfg.database.deployment == "colocated"):
        print(f"\nRunning {cfg.database.deployment} DB with {cfg.database.backend} backend\n")
        launch_coDB(cfg, nodelist, nNodes)
    elif (cfg.database.deployment == "clustered"):
        print(f"\nRunning {cfg.database.deployment} DB with {cfg.database.backend} backend\n")
        launch_clDB(cfg, nodelist, nNodes)
    else:
        print("\nERROR: Launcher is either colocated or clustered\n")

    # Quit
    print("Quitting")


## Run main
if __name__ == "__main__":
    main()
