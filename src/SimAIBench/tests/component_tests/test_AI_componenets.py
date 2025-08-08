from SimAIBench.training import AI
import subprocess
import torch


def test_ai_cpu():
    ai_component = AI(num_hidden_layers=2)
    ai_component.train(run_count=1)
    ai_component.infer(run_count=1)
    return

def test_ai_gpu():
    try:
        if torch.cuda.is_available():
            ai_component = AI(num_hidden_layers=2,device="cuda")
            ai_component.train(run_count=1)
            ai_component.infer(run_count=1)
        else:
            print("CUDA is not available. Skipping GPU tests.")
    except:
        if not hasattr(torch, 'xpu'):
            print("Intel XPU is not available in this PyTorch installation. Skipping XPU tests.")
            return
        if torch.xpu.is_available():
            ai_component = AI(num_hidden_layers=2,device="xpu")
            ai_component.train(run_count=1)
            ai_component.infer(run_count=1)
        else:
            print("Intel XPU is not available. Skipping XPU tests.")
            return
    return

def test_ai_ddp():
    cmd = "mpirun -n 4 python3 -c " + f"'from mpi4py import MPI;from SimAIBench.training import AI;comm = MPI.COMM_WORLD;AI(num_hidden_layers=2,ddp=True,comm=comm).train(run_count=1)'"
    result = subprocess.run(cmd, shell=True, capture_output=True)
    assert result.returncode == 0, f"Distributed AI component failed to execute. Error: {result.stderr}"

    try:
        if torch.cuda.is_available():
            cmd = "mpirun -n 4 python3 -c " + f"'from mpi4py import MPI;from SimAIBench.training import AI;comm = MPI.COMM_WORLD;AI(num_hidden_layers=2,ddp=True,device=\"cuda\",comm=comm).train(run_count=1)'"
            result = subprocess.run(cmd, shell=True, capture_output=True)
            assert result.returncode == 0, f"Distributed AI component failed to execute. Error: {result.stderr}"
        else:
            print("CUDA is not available. Skipping GPU tests.")
    except:
        if not hasattr(torch, 'xpu'):
            print("Intel XPU is not available in this PyTorch installation. Skipping XPU tests.")
            return
        if torch.xpu.is_available():
            cmd = "mpirun -n 4 python3 -c " + f"'from mpi4py import MPI;from SimAIBench.training import AI;comm = MPI.COMM_WORLD;AI(num_hidden_layers=2,ddp=True,device=\"xpu\",comm=comm).train(run_count=1)'"
            result = subprocess.run(cmd, shell=True, capture_output=True)
            assert result.returncode == 0, f"Distributed AI component failed to execute. Error: {result.stderr}"
        else:
            print("Intel XPU is not available. Skipping XPU tests.")

    return

if __name__ == "__main__":
    test_ai_cpu()
    test_ai_gpu()
    test_ai_ddp()
    print("AI component tests passed.")