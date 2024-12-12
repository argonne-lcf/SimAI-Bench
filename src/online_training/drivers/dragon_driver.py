import os
import sys
from typing import Tuple, List, Optional
from omegaconf import DictConfig, OmegaConf
import hydra

import dragon
import multiprocessing as mp
from dragon.data.ddict.ddict import DDict
#from dragon.data.distdictionary.dragon_dict import DragonDict
from dragon.native.process_group import ProcessGroup
from dragon.native.process import Process, ProcessTemplate, MSG_PIPE, MSG_DEVNULL
from dragon.infrastructure.connection import Connection
from dragon.infrastructure.policy import Policy
from dragon.native.machine import cpu_count, current, System, Node


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

## Launch a process group
def launch_ProcessGroup(num_procs: int, num_procs_pn: int, nodelist,
                        exe: str, args_list: List[str], run_dir: str, 
                        global_policy: Optional[Policy] = None,
                        cpu_bind: Optional[List[int]] = None,
                        ddicts: Optional[List[str]] = None) -> None:
    """
    Launch a ProcessGroup
    """ 
    grp = ProcessGroup(restart=False, pmi_enabled=True, 
                       ignore_error_on_exit=True, policy=global_policy)
    for node_num in range(len(nodelist)):   
        node_name = Node(nodelist[node_num]).hostname
        if ddicts is not None:
            args_list.pop(-1)
            if 'online.backend=dragon' in args_list:
                args_list.append(f'online.dragon.dictionary={ddicts[node_num]}')
            else:
                args_list.append(f'--dictionary={ddicts[node_num]}')
        if cpu_bind is not None and len(cpu_bind)>0:
            for proc in range(num_procs_pn):
                local_policy = Policy(placement=Policy.Placement.HOST_NAME,host_name=node_name,
                                      cpu_affinity=[cpu_bind[proc]])
                grp.add_process(nproc=1, 
                                template=ProcessTemplate(target=exe, 
                                                         args=list(args_list), 
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
def launch_colocated(cfg: DictConfig, dragon_nodelist: List[str]) -> None:
    """
    Launch the workflow with the colocated deployment (components are launched on same set of nodes,
    and data is kept local to each node, no inter-node transfers)

    :param cfg: hydra config
    :type cfg: DictConfig
    :param dragon_nodelist: node list provided by Dragon
    :type dragon_nodelist: List[str]
    """
    # Print nodelist
    print(f"\nRunning on {len(dragon_nodelist)} total nodes")
    print([Node(dragon_nodelist[i]).hostname for i in range(len(dragon_nodelist))], "\n")

    global_policy = Policy(distribution=Policy.Distribution.BLOCK)
    sim_nodelist = dragon_nodelist
    ml_nodelist = dragon_nodelist

    # Launch a DDict on each node
    num_dd_nodes = 1
    node_mem_size = cfg.dict.mem_size_per_node * (1024*1024*1024)
    ddicts = {}
    ddicts_serialized = []
    for node_num in range(len(dragon_nodelist)):
        try:
            node_name = Node(dragon_nodelist[node_num]).hostname
            dd_policy = Policy(placement=Policy.Placement.HOST_NAME, host_name=node_name)
            dd = DDict(cfg.dict.managers_per_node, num_dd_nodes, node_mem_size, policy=dd_policy)
            dd['node'] = node_name
            ddicts[node_name] = dd
            ddicts_serialized.append(dd.serialize())
        except Exception as e:
            print(e, flush=True)
    print('Launched the dictionaries on all the nodes \n', flush=True)

    # Set up and launch the simulation component
    print('Launching the simulation ...', flush=True)
    sim_args_list = []
    if (cfg.sim.executable.split("/")[-1].split('.')[-1]=='py'):
        sim_exe = sys.executable
        sim_args_list.append(cfg.sim.executable)
    sim_args_list.extend(cfg.sim.arguments.split(' '))
    sim_args_list.append(f'--dictionary={ddicts_serialized[0]}')
    sim_run_dir = os.getcwd()
    sim_launch_proc = mp.Process(target=launch_ProcessGroup, args=(cfg.sim.procs, cfg.sim.procs_pn, sim_nodelist,
                                                                   sim_exe, sim_args_list, sim_run_dir,
                                                                   global_policy, list(cfg.sim.cpu_bind),
                                                                   ddicts_serialized))
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
                         f'online.launch=colocated'],
                         )
    ddicts_serialized_nice = [dd_tmp.replace('=','\=') for dd_tmp in ddicts_serialized]
    ml_args_list.append(f'online.dragon.dictionary={ddicts_serialized_nice[0]}')
    ml_run_dir = os.getcwd()
    ml_launch_proc = mp.Process(target=launch_ProcessGroup, args=(cfg.train.procs, cfg.train.procs_pn, ml_nodelist,
                                                                  ml_exe, ml_args_list, ml_run_dir,
                                                                  global_policy, list(cfg.train.cpu_bind),
                                                                  ddicts_serialized_nice))
    ml_launch_proc.start()
    print('Done\n', flush=True)

    # Join both simulation and training
    ml_launch_proc.join()
    sim_launch_proc.join()
    print('Joined simulation and training \n', flush=True)

    # Destroy all the DDicts
    for node_num in range(len(dragon_nodelist)):
        node_name = Node(dragon_nodelist[node_num]).hostname
        dd = ddicts[node_name]
        dd.destroy()
    print('Destroyed all dictionaries \n', flush=True)
    print('Exiting launcher ...', flush=True) 

## Clustered launch
def launch_clustered(cfg: DictConfig, dd_serialized: str, dragon_nodelist: List[str]) -> None:
    """
    Launch the workflow with the clustered deployment (components are launched on separate set of nodes,
    so data is always transferred across nodes to fill in DDict)

    :param cfg: hydra config
    :type cfg: DictConfig
    :param dd_serialized: serialized Dragon Distributed Dictionary
    :type dd_serialized: str
    :param dragon_nodelist: node list provided by Dragon
    :type dragon_nodelist: List[str]
    """
    # Print nodelist
    print(f"\nRunning on {len(dragon_nodelist)} total nodes")
    dd_nodelist = [dragon_nodelist[i] for i in range(cfg.dict.num_nodes)]
    sim_nodelist = [dragon_nodelist[i] for i in range(cfg.dict.num_nodes, cfg.dict.num_nodes+cfg.sim.num_nodes)]
    ml_nodelist = [dragon_nodelist[i] for i in range(cfg.dict.num_nodes+cfg.sim.num_nodes, 
                                                     cfg.dict.num_nodes+cfg.sim.num_nodes+cfg.train.num_nodes)]
    print(f"Database running on {cfg.dict.num_nodes} nodes:")
    print([Node(dd_nodelist[i]).hostname for i in range(cfg.dict.num_nodes)])
    print(f"Simulatiom running on {cfg.sim.num_nodes} nodes:")
    print([Node(sim_nodelist[i]).hostname for i in range(cfg.sim.num_nodes)])
    print(f"ML running on {cfg.train.num_nodes} nodes:")
    print([Node(ml_nodelist[i]).hostname for i in range(cfg.train.num_nodes)])
    sys.stdout.flush()

    global_policy = Policy(distribution=Policy.Distribution.BLOCK)

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
    print('Exiting launcher ...', flush=True)

## Mixed launch
def launch_mixed(cfg: DictConfig, dd_serialized: str, dragon_nodelist: List[str]) -> None:
    """
    Launch the workflow with the mixed deployment (components are colocated on same nodes,
    but data can still transfer across nodes to fill in DDict)

    :param cfg: hydra config
    :type cfg: DictConfig
    :param dd_serialized: serialized Dragon Distributed Dictionary
    :type dd_serialized: str
    :param dragon_nodelist: node list provided by Dragon
    :type dragon_nodelist: List[str]
    """
    # Print nodelist
    print(f"\nRunning on {len(dragon_nodelist)} total nodes")
    print([Node(dragon_nodelist[i]).hostname for i in range(len(dragon_nodelist))], "\n")

    global_policy = Policy(distribution=Policy.Distribution.BLOCK)
    sim_nodelist = dragon_nodelist
    ml_nodelist = dragon_nodelist

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
    print('Exiting launcher ...', flush=True)

## Main function
@hydra.main(version_base=None, config_path="./conf", config_name="dragon_config")
def main(cfg: DictConfig):
    # Assertions
    assert cfg.deployment == "colocated" or cfg.deployment == "clustered" or cfg.deployment == "mixed", \
                    print("Deployment is either colocated, clustered or mixed")

    # Get information on this allocation
    #sched_nodelist = parseNodeList(cfg.scheduler)
    alloc = System()
    num_tot_nodes = alloc.nnodes
    dragon_nodelist = alloc.nodes

    # Start the Dragon Distributed Dictionary (DDict)
    mp.set_start_method("dragon")
    if cfg.deployment!='colocated':
        total_mem_size = cfg.dict.mem_size_per_node * cfg.dict.num_nodes * (1024*1024*1024)
        dd_policy = Policy(cpu_affinity=list(cfg.dict.cpu_bind)) if cfg.dict.cpu_bind else None
        dd = DDict(cfg.dict.managers_per_node, cfg.dict.num_nodes, 
                   total_mem_size, policy=dd_policy, timeout=3600,
                   num_streams_per_manager=72)
        dd_serialized = dd.serialize()
        print("Launched the Dragon Dictionary \n", flush=True)
    
    if (cfg.deployment == "colocated"):
        print(f"Running with the {cfg.deployment} deployment \n")
        launch_colocated(cfg, dragon_nodelist)
    elif (cfg.deployment == "clustered"):
        print(f"\nRunning with the {cfg.deployment} deployment \n")
        launch_clustered(cfg, dd_serialized, dragon_nodelist)
    elif (cfg.deployment == "mixed"):
        print(f"\nRunning with the {cfg.deployment} deployment \n")
        launch_mixed(cfg, dd_serialized, dragon_nodelist)

    # Close the DDict and quit
    if cfg.deployment!='colocated':
        dd.destroy()
        print("\nClosed the Dragon Dictionary", flush=True)
    print("\nQuitting ...", flush=True)


## Run main
if __name__ == "__main__":
    main()
