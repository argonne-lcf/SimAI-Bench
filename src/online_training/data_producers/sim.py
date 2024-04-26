from argparse import ArgumentParser
from time import perf_counter, sleep, time

import mpi4py
mpi4py.rc.initialize = False
from mpi4py import MPI
mpi_ops = {
        "sum": MPI.SUM,
        "min": MPI.MINLOC,
        "max": MPI.MAXLOC
    }

from online_training.backends.ssim_client import SmartRedis_Sim_Client
from online_training.data_producers import utils

# Main data producer function
def main():
    """Emulate a data producing simulation for online training with SmartSim/SmartRedis
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
    parser.add_argument('--model', default="mlp", type=str, help='ML model identifier (mlp, quadconv, gnn)')
    parser.add_argument('--problem_size', default="debug", type=str, help='Size of problem to emulate (debug)')
    parser.add_argument('--tolerance', default=0.01, type=float, help='ML model convergence tolerance')
    parser.add_argument('--ppn', default=4, type=int, help='Number of processes per node')
    parser.add_argument('--logging', default='no', help='Level of performance logging (no, verbose)')
    parser.add_argument('--train_interval', type=int, default=5, help='Time step interval used to sync with ML training')
    parser.add_argument('--db_launch', default="colocated", type=str, help='Database deployment (colocated, clustered)')
    parser.add_argument('--db_nodes', default=1, type=int, help='Number of database nodes')
    parser.add_argument('--db_max_mem_size', default=1, type=float, help='Maximum size of DB in GB')
    args = parser.parse_args()

    rankl = rank % args.ppn
    if rank==0 and args.logging=="verbose":
        print(f"Hello from MPI rank {rank}/{size}, local rank {rankl} and node {name}\n")
    if rank==0:
        print(f'Running with {args.db_nodes} DB nodes', flush=True)
        print(f'and with {args.ppn} processes per node \n', flush=True)

    # Initialize client
    if args.backend=='smartredis':
        client = SmartRedis_Sim_Client(args, rank, size)
    client.init_client()
    comm.Barrier()
    if rank==0:
        print(f'All {args.backend} clients initialized \n', flush=True)

    # Generate synthetic data for the specific model
    train_array, coords, stats = utils.generate_training_data(args, (rank, size))

    # Send training metadata
    client.setup_training_problem(stats)
    if (args.model=="gnn"):
        client.setup_graph(coords, rank)
    comm.Barrier()
    if rank==0:
        print('Setup metadata for ML problem \n', flush=True)

    # Emulate integration of PDEs with a do loop
    numts = 1000
    success = 0
    tic_loop = perf_counter()
    for step in range(numts):
        # Sleep for a while to emulate the time required by PDE integration
        if rank==0:
            print(f"{step} \t {time()-t_start:>.2E}", flush=True)
        sleep(0.5)
        train_array, _, _ = utils.generate_training_data(args, (rank,size), step)

        if step>0 and step%60==0:
            args.train_interval = int(args.train_interval*1.2)
        if (step%args.train_interval==0):
            # Check if model exists to perform inference
            exists = client.model_exists(comm, args.model)
            if exists:
                if (args.problem_size=="debug" or args.problem_size=="small"):
                    inputs = train_array[:,0]
                    outputs = train_array[:,1]
                error = client.infer_model(comm, args.model, inputs, outputs)
                if (rank==0):
                    print(f"\tPerformed inference with error={error:>8e}", flush=True)
                if error <= args.tolerance:
                    success += 1
                else:
                    success = 0

            # Send training data
            local_free_mem = 1 if client.check_db_mem(train_array) else 0
            global_free_mem = comm.allreduce(local_free_mem)
            if global_free_mem==size:
                client.send_snapshot(train_array, step)
                comm.Barrier()
                if (rank==0):
                    print(f'\tAll ranks finished sending training data', flush=True)
                client.send_step(step)
            else:
                if (rank==0):
                    print(f'\tOut of memory in DB, did not send training data', flush=True)

            # Exit if model has converged to tolerence for 5 consecutive checks
            if success>=5: 
                if rank==0:
                    print("\nModel has converged to tolerence for 5 consecutive checks", flush=True)
                client.stop_train(comm)
                break

    toc_loop = perf_counter()
    time_to_solution = toc_loop - tic_loop

    # Accumulate timing data for client and print summary
    if rank==0:
        print("Summary of timing data:", flush=True)
    client.collect_stats(comm, mpi_ops)
    if (rank==0):
        client.print_stats()

    # Print FOM
    train_array_sz = train_array.itemsize * train_array.size / 1024 / 1024 / 1024
    if rank==0:
        print("\nFOM:")
        utils.print_fom(time_to_solution, train_array_sz, client.time_stats)

    MPI.Finalize()

if __name__ == "__main__":
    main()