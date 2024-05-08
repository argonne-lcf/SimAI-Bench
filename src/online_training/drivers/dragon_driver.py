import os
import sys
from typing import Tuple, List
from omegaconf import DictConfig, OmegaConf
import hydra

import multiprocessing as mp
import dragon
from dragon.data.ddict.ddict import DDict
#from dragon.data.distdictionary.dragon_dict import DragonDict
from dragon.native.process_group import ProcessGroup
from dragon.native.process import Process, ProcessTemplate, MSG_PIPE, MSG_DEVNULL
from dragon.infrastructure.connection import Connection

## Define function to parse node list
def parseNodeList(scheduler: str) -> List[str]:
    """
    Parse the node list provided by the scheduler

    :param scheduler: scheduler descriptor
    :type scheduler: str
    :return: tuple with node list and number of nodes
    :rtype: tuple
    """
    nodelist = []
    if scheduler=='pbs':
        hostfile = os.getenv('PBS_NODEFILE')
        with open(hostfile) as file:
            nodelist = file.readlines()
            nodelist = [line.rstrip() for line in nodelist]
            nodelist = [line.split('.')[0] for line in nodelist]
    return nodelist

## Read output from ProcessGroup
def read_output(stdout_conn: Connection) -> str:
    """Read stdout from the Dragon connection.

    :param stdout_conn: Dragon connection to rank 0's stdout
    :type stdout_conn: Connection
    :return: string with the output from stdout
    :rtype: str
    """
    output = ""
    try:
        # this is brute force
        while True:
            output += stdout_conn.recv()
    except EOFError:
        pass
    finally:
        stdout_conn.close()
    return output

## Read error from ProcessGroup
def read_error(stderr_conn: Connection) -> str:
    """Read stdout from the Dragon connection.

    :param stderr_conn: Dragon connection to rank 0's stderr
    :type stderr_conn: Connection
    :return: string with the output from stderr
    :rtype: str
    """
    output = ""
    try:
        # this is brute force
        while True:
            output += stderr_conn.recv()
    except EOFError:
        pass
    finally:
        stderr_conn.close()
    return output

## Colocated launch
def launch_colocated(cfg: DictConfig, dd: DDict, nodelist: List[str]) -> None:
    """
    Launch the workflow with the colocated deployment

    :param cfg: hydra config
    :type cfg: DictConfig
    :param dd: Dragon Distributed Dictionary
    :type dd: DDict
    :param nodelist: node list provided by scheduler
    :type nodelist: List[str]
    """
    # Print nodelist
    if (nodelist is not None):
        print(f"\nRunning on {len(nodelist)} total nodes")
        print(nodelist, "\n")
        hosts = ','.join(nodelist)

    dd_serialized = dd.serialize()

    # Set up and launch the simulation component
    print('Launching the simulation ...', flush=True)
    sim_args_list = []
    if (cfg.sim.executable.split("/")[-1].split('.')[-1]=='py'):
        sim_exe = sys.executable
        sim_args_list.append(cfg.sim.executable)
    sim_args_list.extend(cfg.sim.arguments.split(' '))
    sim_args_list.append(f'--dictionary={dd_serialized}')
    sim_run_dir = os.getcwd()

    sim_grp = ProcessGroup(restart=False, pmi_enabled=True)
    sim_grp.add_process(nproc=1, 
                    template=ProcessTemplate(target=sim_exe, 
                                             args=sim_args_list, 
                                             cwd=sim_run_dir, 
                                             stdout=MSG_PIPE))
    sim_grp.add_process(nproc=cfg.sim.procs - 1,
                    template=ProcessTemplate(target=sim_exe, 
                                             args=sim_args_list, 
                                             cwd=sim_run_dir, 
                                             stdout=MSG_DEVNULL))
    sim_grp.init()
    sim_grp.start()
    print('Done\n', flush=True)

    # Setup and launch the distributed training component
    print('Launching the training ...', flush=True)
    ml_args_list = []
    ml_exe = sys.executable
    ml_args_list.append(cfg.train.executable)
    if (cfg.train.config_path): ml_args_list.append(f'--config-path={cfg.train.config_path}')
    if (cfg.train.config_name): ml_args_list.append(f'--config-name={cfg.train.config_name}')
    ml_args_list.extend([f'ppn={cfg.train.procs_pn}',
                         f'online.simprocs={cfg.sim.procs}',
                         f'online.backend=dragon',
                         f'online.dragon.launch={cfg.deployment}'])
    dd_serialized_nice = dd_serialized.replace('=','\=')
    ml_args_list.append(f'online.dragon.dictionary={dd_serialized_nice}')
    ml_run_dir = os.getcwd()

    ml_grp = ProcessGroup(restart=False, pmi_enabled=True)
    ml_grp.add_process(nproc=1, 
                    template=ProcessTemplate(target=ml_exe, 
                                             args=ml_args_list, 
                                             cwd=ml_run_dir, 
                                             stdout=MSG_PIPE,
                                             stderr=MSG_PIPE))
    ml_grp.add_process(nproc=cfg.train.procs - 1,
                    template=ProcessTemplate(target=ml_exe, 
                                             args=ml_args_list, 
                                             cwd=ml_run_dir, 
                                             stdout=MSG_DEVNULL))
    ml_grp.init()
    ml_grp.start()
    print('Done\n', flush=True)

    # Read output
    #group_procs = [Process(None, ident=puid) for puid in sim_grp.puids]
    #for proc in group_procs:
    #    if proc.stdout_conn:
    #        std_out = read_output(proc.stdout_conn)
    #        print(std_out, flush=True)
    group_procs = [Process(None, ident=puid) for puid in ml_grp.puids]
    for proc in group_procs:
        #if proc.stdout_conn:
        #    std_out = read_output(proc.stdout_conn)
        #    print(std_out, flush=True)
        if proc.stderr_conn:
            std_err = read_error(proc.stderr_conn)
            print(std_err, flush=True)

    # Join both simulation and training
    ml_grp.join()
    ml_grp.stop()
    sim_grp.join()
    sim_grp.stop()
    print('Exiting driver ...', flush=True)


## Clustered DB launch
def launch_clustered(cfg, dd, nodelist) -> None:
    print("Not implemented yet")


## Main function
@hydra.main(version_base=None, config_path="./conf", config_name="dragon_config")
def main(cfg: DictConfig):
    # Assertions
    assert cfg.scheduler=='pbs' or cfg.scheduler=='local', print("Only allowed schedulers at this time are pbs and local")
    assert cfg.deployment == "colocated" or cfg.deployment == "clustered", \
                    print("Deployment is either colocated or clustered")

    # Get nodes of this allocation
    nodelist = parseNodeList(cfg.scheduler)

    # Start the Dragon Distributed Dictionary (DDict)
    mp.set_start_method("dragon")
    total_mem_size = cfg.dict.total_mem_size * (1024*1024*1024)
    dd = DDict(cfg.dict.managers_per_node, cfg.dict.num_nodes, total_mem_size)
    print("Launched the Dragon Dictionary \n", flush=True)
    
    if (cfg.deployment == "colocated"):
        print(f"Running with the {cfg.deployment} deployment \n")
        launch_colocated(cfg, dd, nodelist)
    elif (cfg.deployment == "clustered"):
        print(f"\nRunning with the {cfg.deployment} deployment \n")
        launch_clustered(cfg, dd, nodelist)
    else:
        print("\nERROR: Deployment is either colocated or clustered\n")

    # Close the DDict and quit
    dd.destroy()
    print("\nClosed the Dragon Dictionary and quitting ...", flush=True)


## Run main
if __name__ == "__main__":
    main()
