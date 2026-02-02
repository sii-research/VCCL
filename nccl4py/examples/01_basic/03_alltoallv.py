#!/usr/bin/env python3
"""
NCCL4Py Basic Example: AlltoAllv
=======================================

This example demonstrates calling the NCCL4Py alltoallv API. 

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


def prefix_sum(counts):
    """Return displacements [0, c0, c0+c1, ...] (length len(counts))."""
    out = [0]
    for c in counts:
        out.append(out[-1] + int(c))
    return out[:-1]

def _row(mat, r, n):
    return mat[r * n : (r + 1) * n]

def _fmt_rows(mat, n):
    return "[" + ", ".join(str(_row(mat, r, n)) for r in range(n)) + "]"

def main():
    # Initialize MPI
    comm_mpi = MPI.COMM_WORLD
    rank = comm_mpi.Get_rank()
    nranks = comm_mpi.Get_size()

    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    torch.cuda.set_device(device)

    unique_id = nccl.get_unique_id() if rank == 0 else None
    unique_id = comm_mpi.bcast(unique_id, root=0)
    nccl_comm = nccl.Communicator.init(nranks=nranks, rank=rank, unique_id=unique_id)

    # Uneven sendcounts: sendcounts[r*nranks+p] = elements rank r sends to rank p (1..3)
    sendcounts = [(r + p) % 3 + 1 for r in range(nranks) for p in range(nranks)]
    sdispls = []
    for r in range(nranks):
        sdispls.extend(prefix_sum([sendcounts[r * nranks + p] for p in range(nranks)]))

    # recvcounts[r*nranks+s] = what r receives from s = what s sends to r
    recvcounts = [sendcounts[s * nranks + r] for r in range(nranks) for s in range(nranks)]
    rdispls = []
    for r in range(nranks):
        rdispls.extend(prefix_sum([recvcounts[r * nranks + s] for s in range(nranks)]))

    send_total = sum(sendcounts[rank * nranks : (rank + 1) * nranks])
    recv_total = sum(recvcounts[rank * nranks : (rank + 1) * nranks])
    relay_total = 1024 * 1024  # brutely make it large enough. FIXME

    sendbuf = nccl.torch.empty(send_total, device=device, dtype=torch.float32)
    recvbuf = nccl.torch.empty(recv_total, device=device, dtype=torch.float32)
    relaybuf = nccl.torch.empty(relay_total, device=device, dtype=torch.float32)

    # Fill send data: encode source rank + global offset for easy validation
    for p in range(nranks):
        disp = sdispls[rank * nranks + p]
        cnt = sendcounts[rank * nranks + p]
        sendbuf[disp : disp + cnt].copy_(
            torch.arange(cnt, device=device, dtype=torch.float32) + rank * 10.0 + disp
        )
    relaybuf.fill_(0.0)
    recvbuf.fill_(-1.0)

    if rank == 0:
        print("=== Before alltoallv (rank 0 send data) ===")
        print("sendcounts:", _fmt_rows(sendcounts, nranks))
        print("recvcounts:", _fmt_rows(recvcounts, nranks))
        print("sdispls:", _fmt_rows(sdispls, nranks))
        print("rdispls:", _fmt_rows(rdispls, nranks))
    print(f"Rank {rank}: sendbuf:", sendbuf.cpu().tolist())

    window = nccl_comm.register_window(sendbuf, flags=nccl.WindowFlag.CollSymmetric)
    if window is None:
        raise RuntimeError("register_window returned None")
    window = nccl_comm.register_window(recvbuf, flags=nccl.WindowFlag.CollSymmetric)
    if window is None:
        raise RuntimeError("register_window returned None")
    window = nccl_comm.register_window(relaybuf, flags=nccl.WindowFlag.CollSymmetric)
    if window is None:
        raise RuntimeError("register_window returned None")

    nccl_comm.alltoallv(sendbuf, recvbuf, sendcounts, sdispls, recvcounts, rdispls, relaybuf)
    torch.cuda.synchronize()

    recv_host = recvbuf.cpu()
    if rank == 0:
        print("=== After alltoallv (rank 0 recv data) ===")
        print("recvbuf:", recv_host.tolist())
        expected_full = torch.full((recv_total,), -1.0, dtype=torch.float32)
        for src in range(nranks):
            cnt = recvcounts[rank * nranks + src]
            if cnt == 0:
                continue
            rdisp = rdispls[rank * nranks + src]
            sdisp_src = sdispls[src * nranks + rank]
            expected_full[rdisp : rdisp + cnt] = (
                torch.arange(cnt, dtype=torch.float32) + (src * 10.0 + sdisp_src)
            )
        print("expected recvbuf:", expected_full.tolist())

    # Check: rank r receives from src recvcounts[r*nranks+src] elements at rdispls[r*nranks+src]
    ok = True
    for src in range(nranks):
        cnt = recvcounts[rank * nranks + src]
        if cnt == 0:
            continue
        disp = rdispls[rank * nranks + src]
        sdisp_src = sdispls[src * nranks + rank]
        expected = torch.arange(cnt, dtype=torch.float32) + (src * 10.0 + sdisp_src)
        if not torch.equal(recv_host[disp : disp + cnt], expected):
            ok = False
            print(f"Rank {rank}: mismatch from src {src}")

    print(f"Rank {rank}: alltoallv {'OK' if ok else 'FAIL'} (send_total={send_total}, recv_total={recv_total})")
    nccl_comm.destroy()
    return 0


if __name__ == "__main__":
    sys.exit(main())

