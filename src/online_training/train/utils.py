##### 
##### This script contains general utilities that can be useful
##### to many training applications and driver scripts 
#####
import os
from os.path import exists, abspath
import sys
import numpy as np
import logging

import mpi4py
mpi4py.rc.initialize = False
from mpi4py import MPI

### MPI Communicator class
class MPI_COMM:
    def __init__(self):
        """
        MPI Communicator class
        """
        self.comm = None
        self.size = None
        self.rank = None
        self.name = None
        self.rankl = None
        self.sum = None
        self.minloc = None
        self.maxloc = None

    def init(self, cfg):
        MPI.Init()
        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        self.name = MPI.Get_processor_name()
        self.rankl = self.rank % (cfg.ppn*cfg.ppd)
        self.sum = MPI.SUM
        self.min = MPI.MIN
        self.max = MPI.MAX
        self.minloc = MPI.MINLOC
        self.maxloc = MPI.MAXLOC

    def finalize(self):
        MPI.Finalize()

### Compute the average of a quantity across all ranks
def metric_average(comm, val):
    avg_val = comm.comm.allreduce(val, op=comm.sum)
    avg_val = avg_val / comm.size
    return avg_val


### Compute the correlation coefficient between predicted and target outputs
def comp_corrCoeff(output_tensor, target_tensor):
    target = target_tensor.numpy()
    target = np.ndarray.flatten(target)
    output = output_tensor.detach().numpy()
    output = np.ndarray.flatten(output)
    corrCoeff = np.corrcoef([output,target])
    return corrCoeff


### Count the number of trainable parameters in a model
def count_weights(model):
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return n_params

### Print FOM
def print_fom(logger: logging.Logger, time2sol: float, train_data_sz: float, ssim_stats: dict):
    logger.info(f"Time to solution [s]: {time2sol:>.3f}")
    total_sr_time = ssim_stats["tot_meta"]["max"][0] \
                    + ssim_stats["tot_train"]["max"][0]
    rel_sr_time = total_sr_time/time2sol*100
    rel_meta_time = ssim_stats["tot_meta"]["max"][0]/time2sol*100
    rel_train_time = ssim_stats["tot_train"]["max"][0]/time2sol*100
    logger.info(f"Relative total overhead [%]: {rel_sr_time:>.3f}")
    logger.info(f"Relative meta data overhead [%]: {rel_meta_time:>.3f}")
    logger.info(f"Relative train overhead [%]: {rel_train_time:>.3f}")
    string = f": min = {train_data_sz/ssim_stats['train']['max'][0]:>4e} , " + \
             f"max = {train_data_sz/ssim_stats['train']['min'][0]:>4e} , " + \
             f"avg = {train_data_sz/ssim_stats['train']['avg']:>4e}"
    logger.info(f"Train data throughput [GB/s] " + string)

### MPI file handler for parallel logging
# adapted from https://gist.github.com/muammar/2baec60fa8c7e62978720686895cdb9f
class MPIFileHandler(logging.FileHandler):                                      
    def __init__(self,
                 filename,
                 mode=MPI.MODE_WRONLY|MPI.MODE_CREATE|MPI.MODE_APPEND ,
                 encoding='utf-8',  
                 delay=False,
                 comm=MPI.COMM_WORLD ):                                                
        self.baseFilename = abspath('/'.join([os.getcwd(),filename]))                           
        self.mode = mode                                                        
        self.encoding = encoding                                            
        self.comm = comm                                                        
        if delay:                                                               
            #We don't open the stream, but we still need to call the            
            #Handler constructor to set level, formatter, lock etc.             
            logging.Handler.__init__(self)                                      
            self.stream = None                                                  
        else:                                                                   
           logging.StreamHandler.__init__(self, self._open())                   
                                                                                
    def _open(self):                                                            
        stream = MPI.File.Open( self.comm, self.baseFilename, self.mode )     
        stream.Set_atomicity(True)                                              
        return stream
                                                    
    def emit(self, record):
        """
        Emit a record.
        If a formatter is specified, it is used to format the record.
        The record is then written to the stream with a trailing newline.  If
        exception information is present, it is formatted using
        traceback.print_exception and appended to the stream.  If the stream
        has an 'encoding' attribute, it is used to determine how to do the
        output to the stream.
        
        Modification:
            stream is MPI.File, so it must use `Write_shared` method rather
            than `write` method. And `Write_shared` method only accept 
            bytestring, so `encode` is used. `Write_shared` should be invoked
            only once in each all of this emit function to keep atomicity.
        """
        try:
            msg = self.format(record)
            stream = self.stream
            stream.Write_shared((msg+self.terminator).encode(self.encoding))
            #self.flush()
        except Exception:
            self.handleError(record)
                                                         
    def close(self):                                                            
        if self.stream:                                                         
            self.stream.Sync()                                                  
            self.stream.Close()                                                 
            self.stream = None    
