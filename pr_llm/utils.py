import os
from pathlib import Path

import hostlist


def get_env(env_variable):
    try:
        return Path(os.environ[env_variable])
    except KeyError:
        raise NameError(f"Make sure to define the env variable {env_variable}.")


def read_slurm_env():
    rank = int(os.environ.get("SLURM_PROCID", "0"))
    local_rank = int(os.environ.get("SLURM_LOCALID", "0"))
    world_size = int(os.environ.get("SLURM_NTASKS", "1"))
    devices = int(os.environ.get("SLURM_GPUS_ON_NODE", "0"))
    num_nodes = int(os.environ.get("SLURM_NNODES", "1"))
    return rank, local_rank, world_size, devices, num_nodes


def setup_slurm():
    # get node list from slurm
    hostnames = hostlist.expand_hostlist(os.environ.get("SLURM_JOB_NODELIST", ""))

    # get IDs of reserved GPU
    gpu_ids = os.environ.get("SLURM_STEP_GPUS", None)
    if gpu_ids is not None:
        gpu_ids = gpu_ids.split(",")
        port_complement = int(min(gpu_ids))
    else:
        port_complement = 0

    # define MASTER_ADD & MASTER_PORT
    if len(hostnames) > 1:
        os.environ["MASTER_ADDR"] = hostnames[0]
        os.environ["MASTER_PORT"] = str(12345 + port_complement)
    else:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(12345 + port_complement)
