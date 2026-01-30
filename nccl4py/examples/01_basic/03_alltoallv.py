#!/usr/bin/env python3
"""
NCCL4Py Basic Example: AlltoAllv
=======================================

This example demonstrates calling the NCCL4Py alltoallv API. The backend
ncclAlltoAllv implementation in VCCL is currently a stub that only prints a
message, so this example focuses on invoking the call rather than validating
data movement.

USAGE:
mpirun -np 4 python 03_alltoallv.py
"""

import sys

try:
    from mpi4py import MPI
except ImportError:
    print("ERROR: mpi4py required. Install with: pip install mpi4py")
    sys.exit(1)

try:
    import torch
except ImportError:
    print("ERROR: PyTorch required. Install with: pip install torch")
    sys.exit(1)

import nccl.core as nccl


def _prefix_sum(counts):
    displs = [0]
    for count in counts[:-1]:
        displs.append(displs[-1] + int(count))
    return displs


def main():
    # Initialize MPI
    comm_mpi = MPI.COMM_WORLD
    rank = comm_mpi.Get_rank()
    nranks = comm_mpi.Get_size()

    # Assign GPU to each process
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    torch.cuda.set_device(device)

    # [NCCL4Py] Generate unique ID on rank 0
    unique_id = nccl.get_unique_id() if rank == 0 else None

    # Broadcast unique ID to all ranks
    unique_id = comm_mpi.bcast(unique_id, root=0)

    # [NCCL4Py] Initialize NCCL communicator
    nccl_comm = nccl.Communicator.init(nranks=nranks, rank=rank, unique_id=unique_id)

    if rank == 0:
        print(f"Running AlltoAllv (stub) with {nranks} ranks...")

    sendcounts = [1] * (nranks * nranks)
    sdispls = []
    for i in range(nranks):
        # For each rank i, calculate displacements starting from 0
        row_counts = sendcounts[i * nranks:(i + 1) * nranks]
        row_displs = _prefix_sum(row_counts)
        sdispls.extend(row_displs)

    recvcounts = [1] * (nranks * nranks)
    rdispls = []
    for i in range(nranks):
        # For each rank i, calculate displacements starting from 0
        row_counts = recvcounts[i * nranks:(i + 1) * nranks]
        row_displs = _prefix_sum(row_counts)
        rdispls.extend(row_displs)

    # Calculate totals for current rank
    sendcounts_row = sendcounts[rank * nranks:(rank + 1) * nranks]
    recvcounts_row = recvcounts[rank * nranks:(rank + 1) * nranks]
    send_total = sum(sendcounts_row)
    recv_total = sum(recvcounts_row)

    # Create one contiguous symmetric buffer: [send | relay | recv]
    # Relay uses 2x max(send, recv) elements to match double-buffer use in RMA path.
    relay_total = max(send_total, recv_total) * 2
    total_elems = send_total + relay_total + recv_total
    sym_buf = nccl.torch.empty(total_elems, device=device, dtype=torch.float32)

    sendbuf = sym_buf[:send_total]
    relaybuf = sym_buf[send_total:send_total + relay_total]
    recvbuf = sym_buf[send_total + relay_total:]

    # Initialize buffers
    sendbuf.copy_(torch.arange(send_total, device=device, dtype=torch.float32) + rank * 1000.0)
    relaybuf.fill_(0.0)
    recvbuf.fill_(-1.0)

    # Collectively register the symmetric window (all ranks must call this)
    window = nccl_comm.register_window(sym_buf, flags=nccl.WindowFlag.CollSymmetric)
    if window is None:
        raise RuntimeError("register_window returned None; symmetric window registration failed")
    print(f"Rank {rank} window: {window}")
    # [NCCL4Py] AlltoAllv
    nccl_comm.alltoallv(sendbuf, recvbuf, sendcounts, sdispls, recvcounts, rdispls, relaybuf)

    torch.cuda.synchronize()

    print(f"Rank {rank}: alltoallv call issued (recvbuf size={recv_total})")

    # [NCCL4Py] Destroy NCCL communicator (collective call)
    nccl_comm.destroy()

    return 0


if __name__ == "__main__":
    sys.exit(main())

