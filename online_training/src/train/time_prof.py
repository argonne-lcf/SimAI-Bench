##### 
##### This script defines the class that is used to measure the performance
##### of the training and data transfer parts of the algorithm 
######

import sys
import numpy as np
import math
import torch

class timeStats:
    # General training metrics
    t_tot = 0.0 # local time spent training (loop over epochs, excluding first 2)
    t_train = 0.0 # local accumulated time spent in training loop
    tp_train = 0.0 # local accumulated training throughput
    i_train = 0 # local number of times training loop is called
    t_val = 0.0 # local accumulated time spent in validation loop
    tp_val = 0.0 # local accumulated validation throughput
    i_val = 0 # local number of times validation loop is called
    t_compMiniBatch = 0.0 # local accumulated time spent computing mini-batch
    i_compMiniBatch = 0 # local number of times computing mini-batch
    t_AveCompMiniBatch = 0.0 # local average time spent computing mini-batch

    # Online training metrics
    t_getBatch = 0.0 # local accumulated time spent grabbing training data for each batch
    i_getBatch = 0 # local number of times training data is grabbed
    t_AveGetBatch = 0.0 # local average time spent grabbing training data for each batch
    t_getBatch_v = 0.0 # local accumulated time spent grabbing validation data for each batch
    i_getBatch_v = 0 # local number of times validation data is grabbed
    t_AveGetBatch_v = 0.0 # local average time spent grabbing validation data for each batch
    t_init = 0.0 # local accumulated time spent initializing Redis clients
    i_init = 0 # local number of times initializing Redis clients
    t_meta = 0.0 # local accumulated time spent transfering metadata
    i_meta = 0 # local number of times metadata is transferred

    # Compute min, max, mean and standard deviation across all processes for a time measure
    def computeStats_f(self, comm, var):
        summ = comm.comm.allreduce(np.array(var),op=comm.sum)
        avg = summ / comm.size
        tmp = np.array((var - avg)**2) 
        std = comm.comm.allreduce(tmp,op=comm.sum)
        std = std / comm.size
        std = math.sqrt(std)
        min_loc = comm.comm.allreduce((var,comm.rank),op=comm.minloc)
        max_loc = comm.comm.allreduce((var,comm.rank),op=comm.maxloc)
        return avg, std, summ, [min_loc[0],min_loc[1]], [max_loc[0],max_loc[1]]

    # Compute min, max, mean and standard deviation across all processes for a counter
    def computeStats_i(self, comm, var):
        avg = comm.comm.allreduce(np.array(var),op=comm.sum)
        avg = avg / comm.size
        tmp = np.array((var - avg)**2) 
        std = comm.comm.allreduce(tmp)
        std = std / comm.size
        std = math.sqrt(std)
        return avg, std

    # Print the timing data
    def printTimeData(self, cfg, comm):
        
        # General training metrics
        avg, std, summ, min_arr, max_arr = self.computeStats_f(comm, self.t_tot)
        if comm.rank==0:
            stats_string = f": min = {min_arr[0]:>8e} , max = {max_arr[0]:>8e} , avg = {avg:>8e} , std = {std:>8e}"
            #stats_string_2 = f": min [{min_arr[0]:>8e},{min_arr[1]:>d}], max [{max_arr[0]:>8e},{max_arr[1]:>d}], avg [{avg:>8e},.], std [{std:>8e},.]"
            print("Total training time [s] " + stats_string + "\n")
        
        avg, std, summ, min_arr, max_arr = self.computeStats_f(comm, self.t_train)
        if comm.rank==0:
            stats_string = f": min = {min_arr[0]:>8e} , max = {max_arr[0]:>8e} , avg = {avg:>8e} , std = {std:>8e}"
            print("Total time spent in training loop [s] " + stats_string)
        avg, std, summ, min_arr, max_arr = self.computeStats_f(comm, self.tp_train/self.i_train)
        if comm.rank==0:
            stats_string = f": min = {min_arr[0]:>8e} , max = {max_arr[0]:>8e} , avg = {avg:>8e} , std = {std:>8e}, sum = {summ:>8e}"
            print("Total training throughput [samples/s] " + stats_string)
        avg, std = self.computeStats_i(comm, self.i_train)
        if comm.rank==0:
            print(f"Number of train loops executed : {int(avg)}\n")
        
        avg, std, summ, min_arr, max_arr = self.computeStats_f(comm, self.t_compMiniBatch)
        if comm.rank==0:
            stats_string = f": min = {min_arr[0]:>8e} , max = {max_arr[0]:>8e} , avg = {avg:>8e} , std = {std:>8e}"
            print(f"Total time spent in batch computation [s] " + stats_string)
        avg, std, summ, min_arr, max_arr = self.computeStats_f(comm, self.t_AveCompMiniBatch)
        if comm.rank==0:
            stats_string = f": min = {min_arr[0]:>8e} , max = {max_arr[0]:>8e} , avg = {avg:>8e} , std = {std:>8e}"
            print(f"Average time for a single batch computation [s] " + stats_string)
        avg, std = self.computeStats_i(comm, self.i_compMiniBatch)
        if comm.rank==0:
            print(f"Number of batches computed : {int(avg)}\n")

        avg, std, summ, min_arr, max_arr = self.computeStats_f(comm, self.t_val)
        if comm.rank==0:
            stats_string = f": min = {min_arr[0]:>8e} , max = {max_arr[0]:>8e} , avg = {avg:>8e} , std = {std:>8e}"
            print(f"Total time spent in validation loop [s] " + stats_string)
        if (self.i_val>0):
            avg, std, summ, min_arr, max_arr = self.computeStats_f(comm, self.tp_val/self.i_val)
            if comm.rank==0:
                stats_string = f": min = {min_arr[0]:>8e} , max = {max_arr[0]:>8e} , avg = {avg:>8e} , std = {std:>8e}, sum = {summ:>8e}"
                print(f"Total validation throughput [samples/s] " + stats_string)
        avg, std = self.computeStats_i(comm, self.i_val)
        if comm.rank==0:
            print(f"Number of validation loops executed : {int(avg)}\n")

        # Online training metrics
        if cfg.online.driver:
            avg, std, summ, min_arr, max_arr = self.computeStats_f(comm, self.t_init)
            if comm.rank==0:
                stats_string = f": min = {min_arr[0]:>8e} , max = {max_arr[0]:>8e} , avg = {avg:>8e} , std = {std:>8e}, sum = {summ:>8e}"
                print(f"SmartRedis client initialization [s] " + stats_string)

            avg, std, summ, min_arr, max_arr = self.computeStats_f(comm, self.t_meta)
            if comm.rank==0:
                stats_string = f": min = {min_arr[0]:>8e} , max = {max_arr[0]:>8e} , avg = {avg:>8e} , std = {std:>8e}, sum = {summ:>8e}"
                print(f"SmartRedis metadata transfer [s] " + stats_string)
            avg, std = self.computeStats_i(comm, self.i_meta)
            if comm.rank==0:
                print(f"SmartRedis calls for metadata transfer : {int(avg)}")

            avg, std, summ, min_arr, max_arr = self.computeStats_f(comm, self.t_getBatch)
            if comm.rank==0:
                stats_string = f": min = {min_arr[0]:>8e} , max = {max_arr[0]:>8e} , avg = {avg:>8e} , std = {std:>8e}, sum = {summ:>8e}"
                print(f"SmartRedis training batch data transfer [s] " + stats_string)
            avg, std, summ, min_arr, max_arr = self.computeStats_f(comm, self.t_AveGetBatch)
            if comm.rank==0:
                stats_string = f": min = {min_arr[0]:>8e} , max = {max_arr[0]:>8e} , avg = {avg:>8e} , std = {std:>8e}, sum = {summ:>8e}"
                print(f"SmartRedis average training batch data transfer [s] " + stats_string)
            avg, std = self.computeStats_i(comm, self.i_getBatch)
            if comm.rank==0:
                print(f"SmartRedis calls for training batch data transfer : {int(avg)}")
            
            avg, std, summ, min_arr, max_arr = self.computeStats_f(comm, self.t_getBatch_v)
            if comm.rank==0:
                stats_string = f": min = {min_arr[0]:>8e} , max = {max_arr[0]:>8e} , avg = {avg:>8e} , std = {std:>8e}, sum = {summ:>8e}"
                print(f"SmartRedis validation batch data transfer [s] " + stats_string)
            avg, std, summ, min_arr, max_arr = self.computeStats_f(comm, self.t_AveGetBatch_v)
            if comm.rank==0:
                stats_string = f": min = {min_arr[0]:>8e} , max = {max_arr[0]:>8e} , avg = {avg:>8e} , std = {std:>8e}, sum = {summ:>8e}"
                print(f"SmartRedis average validation batch data transfer [s] " + stats_string)
            avg, std = self.computeStats_i(comm, self.i_getBatch_v)
            if comm.rank==0:
                print(f"SmartRedis calls for validation batch data transfer : {int(avg)}")

        if comm.rank==0: print("")
    
        

