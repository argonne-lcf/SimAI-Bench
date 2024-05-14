import os
import sys
from typing import Tuple, List, Optional
from omegaconf import DictConfig, OmegaConf
import hydra

import multiprocessing as mp
import dragon
from dragon.data.ddict.ddict import DDict
#from dragon.data.distdictionary.dragon_dict import DragonDict
from dragon.native.process_group import ProcessGroup
from dragon.native.process import Process, ProcessTemplate, MSG_PIPE, MSG_DEVNULL
from dragon.infrastructure.connection import Connection
from dragon.infrastructure.policy import Policy
from dragon.native.machine import cpu_count, current, System, Node


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

## Launch a process group
def launch_ProcessGroup(num_procs: int, num_procs_pn: int, nodelist,
                        exe: str, args_list: List[str], run_dir: str, 
                        global_policy: Optional[Policy] = None,
                        cpu_bind: Optional[List[int]] = None) -> None:
    """
    Launch a ProcessGroup
    """ 
    grp = ProcessGroup(restart=False, pmi_enabled=True, 
                       ignore_error_on_exit=True, policy=global_policy)
    for node_num in range(len(nodelist)):   
        node_name = Node(nodelist[node_num]).hostname
        if cpu_bind is not None and len(cpu_bind)>0:
            for proc in range(num_procs_pn):
                local_policy = Policy(placement=Policy.Placement.HOST_NAME,host_name=node_name,
                                      cpu_affinity=[cpu_bind[proc]])
                grp.add_process(nproc=1, 
                                template=ProcessTemplate(target=exe, 
                                                         args=args_list, 
                                                         cwd=run_dir,
                                                         policy=local_policy, 
                                                         stdout=MSG_DEVNULL))
        else:
            local_policy = Policy(placement=Policy.Placement.HOST_NAME,host_name=node_name)
            grp.add_process(nproc=num_procs_pn, 
                            template=ProcessTemplate(target=exe, 
                                                     args=args_list, 
                                                     cwd=run_dir,
                                                     policy=local_policy, 
                                                     stdout=MSG_DEVNULL))
    grp.init()
    grp.start()
    grp.join()
    grp.stop()

## Colocated launch
def launch_mixed(cfg: DictConfig, dd_serialized: str, nodelist: List[str]) -> None:
    """
    Launch the workflow with the mixed deployment (components are colocated on same nodes,
    but data can still transfer across nodes to fill in DDict

    :param cfg: hydra config
    :type cfg: DictConfig
    :param dd_serialized: serialized Dragon Distributed Dictionary
    :type dd_serialized: str
    :param nodelist: node list provided by scheduler
    :type nodelist: List[str]
    """
    # Print nodelist
    print(f"\nRunning on {len(nodelist)} total nodes")
    print(nodelist, "\n")

    global_policy = Policy(distribution=Policy.Distribution.BLOCK)
    sim_nodelist = nodelist
    ml_nodelist = nodelist

    # Set up and launch the simulation component
    print('Launching the simulation ...', flush=True)
    sim_args_list = []
    if (cfg.sim.executable.split("/")[-1].split('.')[-1]=='py'):
        sim_exe = sys.executable
        sim_args_list.append(cfg.sim.executable)
    sim_args_list.extend(cfg.sim.arguments.split(' '))
    sim_args_list.append(f'--dictionary={dd_serialized}')
    sim_run_dir = os.getcwd()
    sim_launch_proc = mp.Process(target=launch_ProcessGroup, args=(cfg.sim.procs, cfg.sim.procs_pn, sim_nodelist, 
                                                                   sim_exe, sim_args_list, sim_run_dir,
                                                                   global_policy, list(cfg.sim.cpu_bind)))
    sim_launch_proc.start()
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
                         f'online.launch=clustered'],
                         )
    dd_serialized_nice = dd_serialized.replace('=','\=')
    ml_args_list.append(f'online.dragon.dictionary={dd_serialized_nice}')
    ml_run_dir = os.getcwd()
    ml_launch_proc = mp.Process(target=launch_ProcessGroup, args=(cfg.train.procs, cfg.train.procs_pn, ml_nodelist,
                                                                  ml_exe, ml_args_list, ml_run_dir,
                                                                  global_policy, list(cfg.train.cpu_bind)))
    ml_launch_proc.start()
    print('Done\n', flush=True)

    # Join both simulation and training
    ml_launch_proc.join()
    sim_launch_proc.join()
    print('Exiting driver ...', flush=True)


## Clustered DB launch
def launch_clustered(cfg, dd, nodelist) -> None:
    print("Not implemented yet")


## Main function
@hydra.main(version_base=None, config_path="./conf", config_name="dragon_config")
def main(cfg: DictConfig):
    # Assertions
    assert cfg.scheduler=='pbs' or cfg.scheduler=='local', print("Only allowed schedulers at this time are pbs and local")
    assert cfg.deployment == "colocated" or cfg.deployment == "clustered" or cfg.deployment == "mixed", \
                    print("Deployment is either colocated, clustered or mixed")

    # Get information on this allocation
    alloc = System()
    num_tot_nodes = alloc.nnodes()
    nodelist = alloc.nodes

    # Start the Dragon Distributed Dictionary (DDict)
    mp.set_start_method("dragon")
    total_mem_size = cfg.dict.total_mem_size * (1024*1024*1024)
    dd_policy = Policy(cpu_affinity=list(cfg.dict.cpu_bind)) if cfg.dict.cpu_bind else None
    dd = DDict(cfg.dict.managers_per_node, cfg.dict.num_nodes, total_mem_size, policy=dd_policy)
    print("Launched the Dragon Dictionary \n", flush=True)
    
    dd_serialized = dd.serialize()
    if (cfg.deployment == "colocated"):
        print(f"Running with the {cfg.deployment} deployment \n")
        launch_colocated(cfg, dd_serialized, nodelist)
    elif (cfg.deployment == "clustered"):
        print(f"\nRunning with the {cfg.deployment} deployment \n")
        launch_clustered(cfg, dd_serialized, nodelist)
    elif (cfg.deployment == "mixed"):
        print(f"\nRunning with the {cfg.deployment} deployment \n")
        launch_mixed(cfg, dd_serialized, nodelist)

    # Close the DDict and quit
    dd.destroy()
    print("\nClosed the Dragon Dictionary and quitting ...", flush=True)


## Run main
if __name__ == "__main__":
    main()
