import numpy as np
import time
import os
import sys
from abc import ABC, abstractmethod
from typing import Callable, Tuple, Union, Any

try:
    import cupy as cp
    from cupy.cuda import nccl
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

try:
    import dpnp as dnp
    import dpctl
    devices = dpctl.get_devices()
    default_queues = [dpctl.SyclQueue(d,property="enable_profiling") for d in devices]
    DPNP_AVAILABLE = True
except ImportError:
    DPNP_AVAILABLE = False

try:
    import mpi4py
    mpi4py.rc.initialize = False
    from mpi4py import MPI
    MPI4PY_AVAILABLE = True
except ImportError:
    MPI4PY_AVAILABLE = False

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False


#################
#misc 
#################

def sleep(seconds):
    time.sleep(seconds)

def get_device_module(device):
    if device == "cuda":
        if not CUPY_AVAILABLE:
            raise ImportError("CuPy is not installed. Install CuPy to use cuda capabilities.")
        return cp
    elif device == "xpu":
        if not DPNP_AVAILABLE:
            raise ImportError("DPNP not installed")
        return dnp
    else:
        return np

def init_mpi():
    if not MPI.Is_initialized():
        MPI.Init()
    
    
#################
#io
#################


class IOKernel(ABC):
    """Base class for I/O operations."""
    
    @abstractmethod
    def __call__(self, num_bytes: int, data_root_dir: str, filename_suffix=None, **kwargs):
        """
        Execute an I/O operation.
        
        :param num_bytes: Number of bytes to read/write
        :param data_root_dir: Directory to read from or write to
        :param filename_suffix: Optional suffix for filenames
        :param kwargs: Additional arguments
        :return: Result of the I/O operation
        """
        pass


class WriteSingleRank(IOKernel):
    def __call__(self, num_bytes: int, data_root_dir: str, filename_suffix=None, **kwargs):
        if not MPI4PY_AVAILABLE:
            raise ImportError("mpi4py is not installed. Install mpi4py to use multi-process read/write.")
        elif not H5PY_AVAILABLE:
            raise ImportError("h5py is not installed. Install h5py to use read/write.")
        else:
            if not MPI.Is_initialized():
                raise RuntimeError("MPI is not initialized. Please initialize MPI before calling this function.")
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()

            if rank == 0:
                filename = os.path.join(data_root_dir, "data.h5")
                
                num_elem = num_bytes // 4
                data = np.empty(num_elem, dtype=np.float32)
        
                with h5py.File(filename, 'w') as f:
                    dset = f.create_dataset("data", data=data)


class WriteNonMPI(IOKernel):
    def __call__(self, num_bytes: int, data_root_dir: str, filename_suffix=None, **kwargs):
        if not MPI4PY_AVAILABLE:
            raise ImportError("mpi4py is not installed. Install mpi4py to use multi-process read/write.")
        elif not H5PY_AVAILABLE:
            raise ImportError("h5py is not installed. Install h5py to use read/write.")
        else:
            if not MPI.Is_initialized():
                raise RuntimeError("MPI is not initialized. Please initialize MPI before calling this function.")
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()

            if filename_suffix == None:
                filename = os.path.join(data_root_dir, f"data_{rank}.h5")
            else:
                filename = os.path.join(data_root_dir, f"data_{rank}_{filename_suffix}.h5")
            print("In writeNonMPI, rank = ", rank, " filename = ", filename)
            
            num_elem = num_bytes // 4
            data = np.empty(num_elem, dtype=np.float32)

            with h5py.File(filename, 'w') as f:
                dset = f.create_dataset("data", data=data)


class WriteWithMPI(IOKernel):
    def __call__(self, num_bytes: int, data_root_dir: str, filename_suffix=None, **kwargs):
        if not MPI4PY_AVAILABLE:
            raise ImportError("mpi4py is not installed. Install mpi4py to use multi-process read/write.")
        elif not H5PY_AVAILABLE:
            raise ImportError("h5py is not installed. Install h5py to use read/write.")
        else:
            if not MPI.Is_initialized():
                raise RuntimeError("MPI is not initialized. Please initialize MPI before calling this function.")
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            size = comm.Get_size()

            num_elem = num_bytes // 4
            num_elem_tot = num_elem * size
            data = np.empty(num_elem, dtype=np.float32)

            if filename_suffix == None:
                filename = os.path.join(data_root_dir, 'data.h5')
            else:
                filename = os.path.join(data_root_dir, f"data_{filename_suffix}.h5")
            print("In writeWithMPI, rank = ", rank, " filename = ", filename)

            with h5py.File(filename, 'w', driver='mpio', comm=comm) as f:
                dset = f.create_dataset("data", (num_elem_tot, ), dtype=np.float32)
                offset = rank * num_elem
                dset[offset:offset+num_elem] = data


class ReadNonMPI(IOKernel):
    def __call__(self, num_bytes: int, data_root_dir: str, filename_suffix=None, **kwargs):
        if not MPI4PY_AVAILABLE:
            raise ImportError("mpi4py is not installed. Install mpi4py to use multi-process read/write.")
        elif not H5PY_AVAILABLE:
            raise ImportError("h5py is not installed. Install h5py to use read/write.")
        else:
            if not MPI.Is_initialized():
                raise RuntimeError("MPI is not initialized. Please initialize MPI before calling this function.")
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()

            if filename_suffix == None:
                filename = os.path.join(data_root_dir, f"data_{rank}.h5")
            else:
                filename = os.path.join(data_root_dir, f"data_{rank}_{filename_suffix}.h5")
            print("In readNonMPI, rank = ", rank, " filename = ", filename)
            
            num_elem = num_bytes // 4

            with h5py.File(filename, 'r') as f:
                data = f['data'][0:num_elem]
                return data


class ReadWithMPI(IOKernel):
    def __call__(self, num_bytes: int, data_root_dir: str, filename_suffix=None, **kwargs):
        if not MPI4PY_AVAILABLE:
            raise ImportError("mpi4py is not installed. Install mpi4py to use multi-process read/write.")
        elif not H5PY_AVAILABLE:
            raise ImportError("h5py is not installed. Install h5py to use read/write.")
        else:
            if not MPI.Is_initialized():
                raise RuntimeError("MPI is not initialized. Please initialize MPI before calling this function.")
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            size = comm.Get_size()

            num_elem = num_bytes // 4
            num_elem_tot = num_elem * size
            data = np.empty(num_elem, dtype=np.float32)

            if filename_suffix == None:
                filename = os.path.join(data_root_dir, 'data.h5')
            else:
                filename = os.path.join(data_root_dir, f"data_{filename_suffix}.h5")
            print("In readWithMPI, rank = ", rank, " filename = ", filename)

            with h5py.File(filename, 'r', driver='mpio', comm=comm) as f:
                dset = f['data']
                offset = rank * num_elem
                dset.read_direct(data, np.s_[offset:offset+num_elem])
                return data


def get_io_kernel_from_string(kernel_name: str):
    kernel_name_lower = kernel_name.lower()
    for kernel_class in IOKernel.__subclasses__():
        if kernel_name_lower == kernel_class.__name__.lower():
            return kernel_class()
    
    raise ValueError(f"Unknown I/O kernel name {kernel_name}.")

def get_all_io_kernel_names():
    """
    Returns a list of all I/O kernel names.
    """
    return [kernel_class.__name__ for kernel_class in IOKernel.__subclasses__()]

def get_all_io_kernels():
    """
    Returns a list of all I/O kernels.
    """
    return [kernel_class() for kernel_class in IOKernel.__subclasses__()]


# Legacy functions for backward compatibility
def writeSingleRank(num_bytes, data_root_dir):
    kernel = WriteSingleRank()
    return kernel(num_bytes, data_root_dir)


def writeNonMPI(num_bytes, data_root_dir, filename_suffix=None):
    kernel = WriteNonMPI()
    return kernel(num_bytes, data_root_dir, filename_suffix)


def writeWithMPI(num_bytes, data_root_dir, filename_suffix=None):
    kernel = WriteWithMPI()
    return kernel(num_bytes, data_root_dir, filename_suffix)


def readNonMPI(num_bytes, data_root_dir, filename_suffix=None):
    kernel = ReadNonMPI()
    return kernel(num_bytes, data_root_dir, filename_suffix)


def readWithMPI(num_bytes, data_root_dir, filename_suffix=None):
    kernel = ReadWithMPI()
    return kernel(num_bytes, data_root_dir, filename_suffix)


#################
#comm 
#################

class CollectiveCommKernel(ABC):
    """Base class for collective communication kernels."""
    @abstractmethod
    def __call__(self, device: str, data_size: tuple = (32, 32, 32), backend: str = "mpi", **kwargs):
        """
        Execute a collective communication operation.
        
        :param device: The device to use ('cpu', 'cuda', 'xpu')
        :param data_size: The size of the data to communicate
        :param backend: Communication backend ('mpi', 'nccl')
        :param kwargs: Additional arguments
        :return: The result of the communication operation
        """
        pass

class AllReduce(CollectiveCommKernel):
    def __call__(self, device: str, data_size: tuple = (32, 32), backend: str = "mpi", **kwargs):
        xp = get_device_module(device)
        if not MPI4PY_AVAILABLE:
            raise ImportError("mpi4py is not installed. Install mpi4py to perform allreduce.")
        else:
            if not MPI.Is_initialized():
                raise RuntimeError("MPI is not initialized. Please initialize MPI before calling this function.")
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            size = comm.Get_size()

            if backend == "mpi":
                sendbuf = np.empty(data_size, dtype=np.float32)
                recvbuf = np.empty(data_size, dtype=np.float32)
            else:
                sendbuf = xp.empty(data_size, dtype=xp.float32)
                recvbuf = xp.empty(data_size, dtype=xp.float32)
            
            if device == "cpu":
                comm.Allreduce(sendbuf, recvbuf, op=MPI.SUM)
        
            elif device == "cuda":
                if backend == "nccl":
                    uid = nccl.get_unique_id()
                    comm_nccl = nccl.NcclCommunicator(size, uid, rank)
                    comm_nccl.allReduce(sendbuf.data.ptr, recvbuf.data.ptr, data_size, nccl.NCCL_FLOAT32, nccl.NCCL_SUM, cp.cuda.Stream.null)
                    cp.cuda.Stream.null.synchronize()
                elif backend == "mpi":
                    comm.Allreduce(sendbuf, recvbuf, op=MPI.SUM)
                else:
                    raise ValueError(f"Invalid backend {backend}. Choose either 'nccl' or 'mpi'.")
            elif device == "xpu":
                if backend == "mpi":
                    comm.Allreduce(sendbuf, recvbuf, op=MPI.SUM)
                else:
                    raise ValueError(f"Invalid backend {backend}. Choose 'mpi'.")
            else:
                raise ValueError(f"Invalid device {device}. Choose either 'cpu', 'cuda', or 'xpu'.")
            
            return recvbuf

class AllGather(CollectiveCommKernel):
    def __call__(self, device: str, data_size: tuple = (32, 32, 32), backend: str = "mpi", **kwargs):
        xp = get_device_module(device)
        if not MPI4PY_AVAILABLE:
            raise ImportError("mpi4py is not installed. Install mpi4py to perform allgather.")
        else:
            if not MPI.Is_initialized():
                raise RuntimeError("MPI is not initialized. Please initialize MPI before calling this function.")
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            size = comm.Get_size()

            if backend == "mpi":
                sendbuf = np.empty(data_size, dtype=np.float32)
                recvbuf = np.empty(tuple([data_size[i] for i in range(len(data_size)-1)] + [data_size[-1] * size]), dtype=np.float32)
            else:
                sendbuf = xp.empty(data_size, dtype=xp.float32)
                recvbuf = xp.empty(tuple([data_size[i] for i in range(len(data_size)-1)] + [data_size[-1] * size]), dtype=xp.float32)

            if device == "cpu":
                comm.Allgather(sendbuf, recvbuf)

            elif device == "cuda":
                if backend == "mpi":
                    comm.Allgather(sendbuf, recvbuf)
                elif backend == "nccl":
                    uid = nccl.get_unique_id()
                    comm_nccl = nccl.NcclCommunicator(size, uid, rank)
                    comm_nccl.allGather(sendbuf.data.ptr, recvbuf.data.ptr, data_size, nccl.NCCL_FLOAT32, cp.cuda.Stream.null)
                    cp.cuda.Stream.null.synchronize()
                else:
                    raise ValueError(f"Invalid backend {backend}. Choose either 'nccl' or 'mpi'.")
            elif device == "xpu":
                if backend == "mpi":
                    comm.Allgather(sendbuf, recvbuf)
                else:
                    raise ValueError(f"Invalid backend {backend}. Choose 'mpi'.")
            else:   
                raise ValueError(f"Invalid device {device}. Choose either 'cpu', 'cuda', or 'xpu'.")
                
            return recvbuf

def get_comm_kernel_from_string(kernel_name: str):
    kernel_name_lower = kernel_name.lower()
    for kernel_class in CollectiveCommKernel.__subclasses__():
        if kernel_name_lower == kernel_class.__name__.lower():
            return kernel_class()
    
    raise ValueError(f"Unknown communication kernel name {kernel_name}.")

def get_all_comm_kernel_names():
    """
    Returns a list of all collective communication kernel names.
    """
    return [kernel_class.__name__ for kernel_class in CollectiveCommKernel.__subclasses__()]

def get_all_comm_kernels():
    """
    Returns a list of all collective communication kernels.
    """
    return [kernel_class() for kernel_class in CollectiveCommKernel.__subclasses__()]

# Legacy functions for backward compatibility
def MPIallReduce(device: str, data_size: tuple = (32, 32), backend: str = "mpi"):
    kernel = AllReduce()
    return kernel(device, data_size, backend)

def MPIallGather(device: str, data_size: tuple = (32, 32, 32), backend: str = "mpi"):
    kernel = AllGather()
    return kernel(device, data_size, backend)


#################
#data movement
#################

class CopyKernel(ABC):
    """Base class for data movement operations between host and device."""
    
    @abstractmethod
    def __call__(self, data_size: tuple = (32, 32, 32), **kwargs):
        """
        Execute a data copy operation.
        
        :param data_size: The size of the data to copy
        :param kwargs: Additional arguments
        :return: Result of the copy operation
        """
        pass


class CopyHostToDevice(CopyKernel):
    def __call__(self, data_size: tuple = (32, 32, 32), **kwargs):
        if not CUPY_AVAILABLE and not DPNP_AVAILABLE:
            raise ImportError("CuPy or DPNP is not installed.")

        # Allocate array on the host (CPU)
        # Then transfer to the selected device
        data_h = np.empty(data_size, dtype=np.float32)
        if DPNP_AVAILABLE:
            data_d = dnp.array(data_h)
        else:
            data_d = cp.asarray(data_h)
        return data_d


class CopyDeviceToHost(CopyKernel):
    def __call__(self, data_size: tuple = (32, 32, 32), **kwargs):
        if not CUPY_AVAILABLE and not DPNP_AVAILABLE:
            raise ImportError("CuPy or DPNP is not installed.")

        if DPNP_AVAILABLE:
            data_d = dnp.empty(data_size, dtype=dnp.float32)
            data_h = dnp.asnumpy(data_d)
        else:
            data_d = cp.empty(data_size, dtype=cp.float32)
            data_h = cp.asnumpy(data_d)
        return data_h


def get_copy_kernel_from_string(kernel_name: str):
    kernel_name_lower = kernel_name.lower()
    for kernel_class in CopyKernel.__subclasses__():
        if kernel_name_lower == kernel_class.__name__.lower():
            return kernel_class()
    
    raise ValueError(f"Unknown copy kernel name {kernel_name}.")


def get_all_copy_kernel_names():
    """
    Returns a list of all copy kernel names.
    """
    return [kernel_class.__name__ for kernel_class in CopyKernel.__subclasses__()]

def get_all_copy_kernels():
    """
    Returns a list of all copy kernels.
    """
    return [kernel_class() for kernel_class in CopyKernel.__subclasses__()]


# Legacy functions for backward compatibility
def dataCopyH2D(data_size: tuple = (32, 32, 32)):
    kernel = CopyHostToDevice()
    return kernel(data_size)


def dataCopyD2H(data_size: tuple = (32, 32, 32)):
    kernel = CopyDeviceToHost()
    return kernel(data_size)


#################
#computation
#################

class ComputeKernel(ABC):
    """ This ia base class for all compute kernels."""
    @abstractmethod
    def __call__(self, device:str, data_size:tuple=(32,32,32),**kwargs) -> tuple:
        """
        This method should be implemented by subclasses to define the kernel's behavior.
        :param device: The device to use for computation (e.g., 'cpu', 'cuda', 'xpu').
        :param data_size: The size of the data to be processed.
        :param kwargs: Additional arguments for the kernel.
        :return: The result of the computation.
        """
        pass

    def sync(self,device:str):
        if device=="xpu":
            for q in default_queues:
                q.wait()
        else:
            pass

class MatMulSimple2D(ComputeKernel):
    def __call__(self, device: str, data_size: tuple = (32, 32, 32), **kwargs):
        xp = get_device_module(device)
        if device == "xpu":
            default_timers = [dpctl.SyclTimer() for _ in range(len(default_queues))]
            for qid,q in enumerate(default_queues):
                with default_timers[qid](queue=q):
                    matrix_a = xp.empty(data_size, dtype=xp.float32,sycl_queue=q)
                    matrix_b = xp.empty(data_size, dtype=xp.float32,sycl_queue=q)
                    c = xp.matmul(matrix_a, matrix_b)
                    ###this is included to make sure that host_dt also includes the device_dt. See the link below
                    q.wait()
            #timer.dt syncs the tasks by default
            #https://intelpython.github.io/dpctl/latest/api_reference/dpctl/generated/dpctl.SyclTimer.html#dpctl.SyclTimer
            ##Note that host timer only gives the submission time and not the execution time
            ##take the mean of all the default queues
            return tuple(np.mean([(timer.dt.host_dt,timer.dt.device_dt) for timer in default_timers],axis=0))
        else:
            tic = time.time()
            matrix_a = xp.empty(data_size, dtype=xp.float32)
            matrix_b = xp.empty(data_size, dtype=xp.float32)
            c = xp.matmul(matrix_a, matrix_b)
            return tuple([time.time()-tic,time.time()-tic])
        

class MatMulGeneral(ComputeKernel):
    def __call__(self, device: str, data_size: tuple = (32, 32, 32), axis: int | tuple = 2, **kwargs):
        xp = get_device_module(device)
        if device == "xpu":
            default_timers = [dpctl.SyclTimer() for _ in range(len(default_queues))]
            for qid, q in enumerate(default_queues):
                with default_timers[qid](queue=q):
                    matrix_a = xp.empty(data_size, dtype=xp.float32, sycl_queue=q)
                    matrix_b = xp.empty(data_size, dtype=xp.float32, sycl_queue=q)
                    c = xp.tensordot(matrix_a, matrix_b, axis)
                    q.wait()
            # Return mean of all default queues
            return tuple(np.mean([(timer.dt.host_dt,timer.dt.device_dt) for timer in default_timers], axis=0))
        else:
            tic = time.time()
            matrix_a = xp.empty(data_size, dtype=xp.float32)
            matrix_b = xp.empty(data_size, dtype=xp.float32)
            c = xp.tensordot(matrix_a, matrix_b, axis)
            return tuple([time.time()-tic, time.time()-tic])

class FFT(ComputeKernel):
    def __call__(self, device: str, data_size: tuple = (32, 32, 32), type_in: str = "float", transform_dim: int = -1, **kwargs):
        xp = get_device_module(device)
        if device == "xpu":
            default_timers = [dpctl.SyclTimer() for _ in range(len(default_queues))]
            for qid, q in enumerate(default_queues):
                with default_timers[qid](queue=q):
                    if type_in == "float":
                        data_in = xp.empty(data_size, dtype=xp.float32, sycl_queue=q)
                    elif type_in == "double":
                        data_in = xp.empty(data_size, dtype=xp.float64, sycl_queue=q)
                    elif type_in == "complexF":
                        data_in = xp.empty(data_size, dtype=xp.complex64, sycl_queue=q)
                    elif type_in == "complexD":
                        data_in = xp.empty(data_size, dtype=xp.complex128, sycl_queue=q)
                    else:
                        raise TypeError("In fft call, type_in must be one of the following: [float, double, complexF, complexD]")
                    
                    c = xp.fft.fft(data_in, axis=transform_dim)
                    q.wait()
            return tuple(np.mean([(timer.dt.host_dt,timer.dt.device_dt) for timer in default_timers], axis=0))
        else:
            tic = time.time()
            if type_in == "float":
                data_in = xp.empty(data_size, dtype=xp.float32)
            elif type_in == "double":
                data_in = xp.empty(data_size, dtype=xp.float64)
            elif type_in == "complexF":
                data_in = xp.empty(data_size, dtype=xp.complex64)
            elif type_in == "complexD":
                data_in = xp.empty(data_size, dtype=xp.complex128)
            else:
                raise TypeError("In fft call, type_in must be one of the following: [float, double, complexF, complexD]")
            
            c = xp.fft.fft(data_in, axis=transform_dim)
            return tuple([time.time()-tic, time.time()-tic])

class AXPY(ComputeKernel):
    def __call__(self, device: str, data_size: tuple = (32, 32, 32), **kwargs):
        xp = get_device_module(device)
        if device == "xpu":
            default_timers = [dpctl.SyclTimer() for _ in range(len(default_queues))]
            for qid, q in enumerate(default_queues):
                with default_timers[qid](queue=q):
                    x = xp.empty(data_size, dtype=xp.float32, sycl_queue=q)
                    y = xp.empty(data_size, dtype=xp.float32, sycl_queue=q)
                    y += 1.01 * x
                    q.wait()
            return tuple(np.mean([(timer.dt.host_dt,timer.dt.device_dt) for timer in default_timers], axis=0))
        else:
            tic = time.time()
            x = xp.empty(data_size, dtype=xp.float32)
            y = xp.empty(data_size, dtype=xp.float32)
            y += 1.01 * x
            return tuple([time.time()-tic, time.time()-tic])

class InplaceCompute(ComputeKernel):
    def __call__(self, device: str, data_size: tuple = (32, 32, 32), op="exp", **kwargs):
        xp = get_device_module(device)
        if device == "xpu":
            default_timers = [dpctl.SyclTimer() for _ in range(len(default_queues))]
            for qid, q in enumerate(default_queues):
                with default_timers[qid](queue=q):
                    x = xp.empty(data_size, dtype=xp.float32, sycl_queue=q)
                    # op can be either a string identifier or a Python callable
                    if isinstance(op, str):
                        if op == "exp":
                            op_func = xp.exp
                        else:
                            raise ValueError(f"Unknown operator {op}.")
                    else:
                        if not callable(op):
                            raise ValueError("Operator must be a callable function.")
                        else:
                            op_func = op
                    y = op_func(x)
                    q.wait()
            return tuple(np.mean([(timer.dt.host_dt,timer.dt.device_dt) for timer in default_timers], axis=0))
        else:
            tic = time.time()
            x = xp.empty(data_size, dtype=xp.float32)
            # op can be either a string identifier or a Python callable
            if isinstance(op, str):
                if op == "exp":
                    op_func = xp.exp
                else:
                    raise ValueError(f"Unknown operator {op}.")
            else:
                if not callable(op):
                    raise ValueError("Operator must be a callable function.")
                else:
                    op_func = op
            y = op_func(x)
            return tuple([time.time()-tic, time.time()-tic])

class GenerateRandomNumber(ComputeKernel):
    def __call__(self, device: str, data_size: tuple = (32, 32, 32), **kwargs):
        xp = get_device_module(device)
        if device == "xpu":
            default_timers = [dpctl.SyclTimer() for _ in range(len(default_queues))]
            for qid, q in enumerate(default_queues):
                with default_timers[qid](queue=q):
                    x = xp.random.rand(*data_size)
                    q.wait()
            return tuple(np.mean([(timer.dt.host_dt,timer.dt.device_dt) for timer in default_timers], axis=0))
        else:
            tic = time.time()
            x = xp.random.rand(*data_size)
            return tuple([time.time()-tic, time.time()-tic])

class ScatterAdd(ComputeKernel):
    def __call__(self, device: str, data_size: tuple = (32, 32, 32), **kwargs):
        tic = time.time()
        xp = get_device_module(device)
        y = xp.empty(np.prod(data_size), dtype=xp.float32)
        x = xp.empty(np.prod(data_size), dtype=xp.float32)
        idx = xp.random.randint(0, int(np.prod(data_size)), size=int(np.prod(data_size)), dtype=xp.int32)
        if device.lower() == "cpu":
            y += x[idx]
        elif device.lower() == "cuda":
            scatter_add_kernel = cp.RawKernel(r'''
            extern "C" __global__
            void my_scatter_add_kernel(const float *x, const float *y, const int *idx)
            {
                int tid = blockDim.x * blockIdx.x + threadIdx.x;
                // Implementation needed
            }
            ''', 'my_scatter_add_kernel')
            # Implementation needed
        return tuple([time.time()-tic,time.time()-tic])

def get_compute_kernel_from_string(kernel_name: str):
    kernel_name_lower = kernel_name.lower()
    # Loop through all compute kernel subclasses
    for kernel_class in ComputeKernel.__subclasses__():
        if kernel_name_lower == kernel_class.__name__.lower():
            return kernel_class()
    
    # If no match found
    raise ValueError(f"Unknown kernel name {kernel_name}.")

def get_all_compute_kernel_names():
    """
    Returns a list of all compute kernel names.
    """
    return [kernel_class.__name__ for kernel_class in ComputeKernel.__subclasses__()]

def get_all_compute_kernels():
    """
    Returns a list of all compute kernels.
    """
    return [kernel_class() for kernel_class in ComputeKernel.__subclasses__()]


def get_kernel_from_string(kernel_name: str):
    """
    Returns a kernel object based on the kernel name.
    """
    compute_kernel_names = [i.lower() for i in get_all_compute_kernel_names()]
    io_kernel_names = [i.lower() for i in get_all_io_kernel_names()]
    comm_kernel_names = [i.lower() for i in get_all_comm_kernel_names()]
    copy_kernel_names = [i.lower() for i in get_all_copy_kernel_names()]

    for kernal_names,f in zip([compute_kernel_names, io_kernel_names, comm_kernel_names, copy_kernel_names], 
                                    [get_compute_kernel_from_string, get_io_kernel_from_string, get_comm_kernel_from_string, get_copy_kernel_from_string]):
        if kernel_name.lower() in kernal_names:
            return f(kernel_name)
    raise ValueError(f"Unknown kernel name {kernel_name}. Available kernels are: {compute_kernel_names + io_kernel_names + comm_kernel_names + copy_kernel_names}")