from SimAIBench.kernel import *
import subprocess
import shutil
import h5py


def test_all_compute_kernels():
    kernels = get_all_compute_kernels()
    device = "cpu"
    for kernel in kernels:
        kernel(device)
    
    if DPNP_AVAILABLE:
        device="xpu"
        for kernel in kernels:
            kernel(device)
        executed = True
    elif CUPY_AVAILABLE:
        device="cuda"
        for kernel in kernels:
            kernel(device)
        executed = True
    else:
        print("No GPU support available. Skipping GPU tests.")

def test_io():
    # Test I/O kernels
    kernels = get_all_io_kernel_names()
    dirname = os.path.join(os.path.dirname(__file__),"test_dir")
    os.makedirs(dirname, exist_ok=True)
    for kernel_name in kernels:
        if "read" in kernel_name.lower():
            filename = os.path.join(dirname, "data.h5")
            if os.path.exists(filename):
                os.remove(filename)
            with h5py.File(filename, "w") as f:
                f.create_dataset("data", data=np.empty(1024, dtype=np.byte))
        cmd = f"mpirun -n 4 python3 -c 'from mpi4py import MPI; from SimAIBench.kernel import *; k = get_io_kernel_from_string(\"{kernel_name}\");k(1024,\"{dirname}\")'"
        if "withmpi" in kernel_name.lower():
            if not h5py.get_config().mpi:
                print("H5PY is not built with MPI support. Skipping test.")
                continue
        result = subprocess.run(cmd, shell=True, capture_output=True)
        assert result.returncode == 0, f"Failed to execute {kernel_name}. Error: {result.stderr}"
    shutil.rmtree(dirname)

def test_copy():
    if not DPNP_AVAILABLE and not CUPY_AVAILABLE:
        print("No GPU support available. Skipping copy tests.")
        return
    kernels = get_all_copy_kernels()
    for kernel in kernels:
        kernel()

def test_mpi_global():
    def check_mpi(device):
        cmd = "mpirun -n 4 python3 -c " + f"'from mpi4py import MPI; from SimAIBench.kernel import MPIallReduce; MPIallReduce(\"{device}\")'"
        result = subprocess.run(cmd, shell=True, capture_output=True)
        assert result.returncode == 0, f"MPIallReduce failed to execute. Error: {result.stderr}"
        cmd = "mpirun -n 4 python3 -c " + f"'from mpi4py import MPI; from SimAIBench.kernel import MPIallGather; MPIallGather(\"{device}\")'"
        result = subprocess.run(cmd, shell=True, capture_output=True)
        assert result.returncode == 0, f"MPIallGather failed to execute. Error: {result.stderr}"

    check_mpi("cpu")
    if DPNP_AVAILABLE:
        check_mpi("xpu")
    elif CUPY_AVAILABLE:
        check_mpi("cuda")
    else:
        print("No GPU support available. Skipping GPU-MPI tests.")
        return

if __name__ == "__main__":
    test_all_compute_kernels()
    test_copy()
    test_io()
    test_mpi_global()
    print("All compute kernels executed successfully.")