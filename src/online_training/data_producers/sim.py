import sys
import os
from argparse import ArgumentParser
from time import perf_counter, sleep, time
from datetime import datetime
import logging
import psutil

import mpi4py
mpi4py.rc.initialize = False
from mpi4py import MPI
mpi_ops = {
        "sum": MPI.SUM,
        "min": MPI.MINLOC,
        "max": MPI.MAXLOC
    }

try:
    import dragon
    from dragon.globalservices.api_setup import connect_to_infrastructure
    connect_to_infrastructure()
except:
    pass
from online_training.data_producers import utils

# Main data producer function
def main():
    """Emulate a data producing simulation with online training and inference
    """
    # MPI Init
    MPI.Init()
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    name = MPI.Get_processor_name()
    comm.Barrier()

    t_start = time()

    # Parse arguments
    parser = ArgumentParser(description='SmartRedis Data Producer')
    parser.add_argument('--backend', default="smartredis", type=str, help='Backend for client (smartredis)')
    parser.add_argument('--dictionary', default="", type=str, help='Serialized Dragon Dictionary')
    parser.add_argument('--model', default="mlp", type=str, help='ML model identifier (mlp, quadconv, gnn)')
    parser.add_argument('--problem_size', default="debug", type=str, help='Size of problem to emulate (debug)')
    parser.add_argument('--tolerance', default=0.01, type=float, help='ML model convergence tolerance')
    parser.add_argument('--ppn', default=4, type=int, help='Number of processes per node')
    parser.add_argument('--logging', default='debug', help='Level of performance logging (debug, info)')
    parser.add_argument('--train_interval', type=int, default=5, help='Time step interval used to sync with ML training')
    parser.add_argument('--launch', default="colocated", type=str, help='Workflow deployment (colocated, clustered)')
    parser.add_argument('--db_nodes', default=1, type=int, help='Number of database nodes if using SmartSim')
    parser.add_argument('--db_max_mem_size', default=1, type=float, help='Maximum size of database in GB')
    args = parser.parse_args()
    
    # Set up logging
    log_level = getattr(logging, args.logging.upper())
    logger = logging.getLogger(f'[{rank}]')                                
    logger.setLevel(log_level)
    date = datetime.now().strftime('%d.%m.%y_%H.%M') if rank==0 else None
    date = comm.bcast(date, root=0)
    comm.Barrier()
    #formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s')
    formatter = logging.Formatter('%(message)s')
    mh = utils.MPIFileHandler(f"sim_{date}.log", comm=comm)                                        
    mh.setFormatter(formatter)
    logger.addHandler(mh)
    #fh = logging.FileHandler(f'{os.getcwd()}/sim_{date}.log')
    #fh.setFormatter(formatter)
    #if rank==0: logger.addHandler(fh)

    rankl = rank % args.ppn
    if args.logging=='debug':
        try:
            p = psutil.Process()
            core_list = p.cpu_affinity()
        except:
            core_list = []
        logger.debug(f"Hello from MPI rank {rank}/{size}, local rank {rankl}, " \
                     +f"core {core_list}, and node {name}")
    if rank==0:
        logger.info(f'Running with {args.launch} deployment ')
        logger.info(f'and with {args.ppn} processes per node \n')

    # Initialize client
    if args.backend=='smartredis':
        try:
            from online_training.backends.smartredis import SmartRedis_Sim_Client
            client = SmartRedis_Sim_Client(args, rank, size)
        except Exception as e:
            logger.info('Could not import client, exception')
            logger.info(f'{e}')
    elif args.backend=='dragon':
        try:
            from online_training.backends.dragon import Dragon_Sim_Client
            client = Dragon_Sim_Client(args, rank, size)
        except Exception as e:
            logger.info('Could not import client, exception')
            logger.info(f'{e}')
    client.init()
    comm.Barrier()
    if rank==0:
        logger.info(f'All {args.backend} clients initialized \n')
    if rank==client.head_rank: logger.debug(f'Rank {rank} is a head rank')

    # Generate synthetic data for the specific model
    train_array, coords, data_stats = utils.generate_training_data(args, comm)

    # Send training metadata
    client.setup_training_problem(coords, data_stats)
    comm.Barrier()
    if rank==0:
        logger.info('Setup metadata for ML problem \n')

    # Emulate integration of PDEs with a do loop
    numts = 1000
    success = 0
    tic_loop = perf_counter()
    for step in range(numts):
        # Sleep for a while to emulate the time required by PDE integration
        if rank==0:
            logger.info(f"{step} \t {time()-t_start:>.2E}")
        sleep(0.5)
        train_array, _, _ = utils.generate_training_data(args, comm, step)

        #if step>0 and step%60==0:
        #    args.train_interval = int(args.train_interval*1.2)
        if (step%args.train_interval==0):
            # Check if model exists to perform inference
            exists = client.model_exists(comm)
            if exists:
                inputs = train_array[:,:data_stats['n_dim_in']]
                outputs = train_array[:,data_stats['n_dim_in']:]
                error = client.infer_model(comm, inputs, outputs)
                if (rank==0):
                    logger.info(f"\tPerformed inference with error={error:>8e}")
                if error <= args.tolerance:
                    success += 1
                else:
                    success = 0

            # Send training data
            local_free_mem = 1 if client.check_mem(train_array) else 0
            global_free_mem = comm.allreduce(local_free_mem)
            if global_free_mem==size:
                if (rank==0):
                    logger.info(f'\tSending training data with shape {train_array.shape}')
                client.send_snapshot(train_array, step)
                comm.Barrier()
                if (rank==0):
                    logger.info(f'\tAll ranks finished sending training data')
                client.send_step(step)
            else:
                if (rank==0):
                    logger.warning(f'\tOut of memory in staging area, did not send training data')

            # Exit if model has converged to tolerence for 5 consecutive checks
            if success>=5: 
                client.stop_train()
                if rank==0:
                    logger.info("\nModel has converged to tolerence for 5 consecutive checks")
                    logger.info("Told training to quit")
                break

    toc_loop = perf_counter()
    time_to_solution = toc_loop - tic_loop

    # Sync with training
    client.check_train_status()
    comm.Barrier()
    if rank==0:
        logger.info("\nTraining is done too")

    # Accumulate timing data for client and print summary
    client.collect_stats(comm, mpi_ops)
    if rank==0:
        logger.info("\nSummary of client timing data:")
        client.print_stats(logger)

    # Print FOM
    train_array_sz = train_array.itemsize * train_array.size / 1024 / 1024 / 1024
    if rank==0:
        logger.info("\nFOM:")
        utils.print_fom(logger, time_to_solution, train_array_sz, client.time_stats)

    client.destroy()
    mh.close()
    MPI.Finalize()

if __name__ == "__main__":
    main()
