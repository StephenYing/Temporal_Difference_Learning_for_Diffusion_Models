"""
Helpers for distributed training.
"""

import io
import os
import socket

import blobfile as bf
from mpi4py import MPI
import torch as th
import torch.distributed as dist

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 8

SETUP_RETRY_COUNT = 3


def setup_dist():
    """
    Setup a distributed process group.
    """
    if dist.is_initialized():
        return
    
    comm = MPI.COMM_WORLD
    
    # Do not modify CUDA_VISIBLE_DEVICES if it is already set by the scheduler.
    # Each rank will use torch.device(f"cuda:{rank}") which maps to
    # the corresponding GPU in CUDA_VISIBLE_DEVICES
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        # Otherwise, assign one GPU per MPI rank.
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{comm.rank % GPUS_PER_NODE}"
    
    backend = "gloo" if not th.cuda.is_available() else "nccl"
    
    # Increase NCCL timeout to handle slow checkpoint loading via MPI broadcast
    # Default is 10 minutes (600s), increase to 30 minutes (1800s)
    if backend == "nccl":
        os.environ["NCCL_TIMEOUT"] = "1800"

    if backend == "gloo":
        hostname = "localhost"
    else:
        hostname = socket.gethostbyname(socket.getfqdn())
    os.environ["MASTER_ADDR"] = comm.bcast(hostname, root=0)
    os.environ["RANK"] = str(comm.rank)
    os.environ["WORLD_SIZE"] = str(comm.size)

    port = comm.bcast(_find_free_port(), root=0)
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend=backend, init_method="env://")


def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        # When using DDP, each rank should use a different GPU
        # This works correctly when CUDA_VISIBLE_DEVICES is set externally.
        return th.device(f"cuda:{dist.get_rank()}")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file. Each rank loads independently from shared filesystem
    to avoid slow MPI broadcast that can cause NCCL timeout during resume.
    """
    # All ranks read the file directly - Lustre can handle parallel reads
    with bf.BlobFile(path, "rb") as f:
        data = f.read()
    return th.load(io.BytesIO(data), **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
