"""
Helpers for distributed training.
"""

import io
import os
import socket

import blobfile as bf
#from mpi4py import MPI
import torch as th
import torch.distributed as dist
import torch
# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 2   #8

SETUP_RETRY_COUNT = 3


# def setup_dist():
#     """
#     Setup a distributed process group.
#     """
#     backend = "nccl"
#     if dist.is_initialized():
#         return
#
#
#     backend = "nccl"
#     hostname = "localhost"
#     if backend == "nccl":
#         hostname = "localhost"
#     else:
#         hostname = socket.gethostbyname(socket.getfqdn())



########################################################
def setup_dist():
    """
    Setup a distributed process group.
    """
    if dist.is_initialized():
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'

    backend = "gloo" if not th.cuda.is_available() else "nccl"

    if backend == "gloo":
        hostname = "localhost"
    else:
        hostname = socket.gethostbyname(socket.getfqdn())
    os.environ["MASTER_ADDR"] = '127.0.1.1'#comm.bcast(hostname, root=0)
    os.environ["RANK"] = '0'#str(comm.rank)
    os.environ["WORLD_SIZE"] = '1'#str(comm.size)

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    s.listen(1)
    port = s.getsockname()[1]
    s.close()
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend=backend, init_method="env://")

# #
def setup_single_gpu(gpu_index=0):
    """
    设置单 GPU 环境。
    :param gpu_index: 要使用的 GPU 索引号，默认为 0。
    """
    # 确保 CUDA 可用
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check your installation.")

    # 设置 PyTorch 使用的 GPU
    torch.cuda.set_device(gpu_index)

    # 打印当前使用的 GPU 信息
    print(f"Using GPU: {torch.cuda.get_device_name(gpu_index)}")

# 调用函数来设置 GPU
setup_single_gpu(1)  # 假设您想使用的是 GPU 索引为 1 的卡


# 调用函数来设置 GPU
# setup_single_gpu(1)  # 假设您想使用的是 GPU 索引为 1 的卡

##########################################################


def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        return th.device(f"cuda")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    mpigetrank=0
    if mpigetrank==0:
        with bf.BlobFile(path, "rb") as f:
            data = f.read()
    else:
        data = None
    return th.load(io.BytesIO(data), **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with th.no_grad():
            dist.broadcast(p.clone(), 0)
            # dist.broadcast(p.clone(), 0)

def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()




