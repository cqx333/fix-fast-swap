"""MPI-launched Python process only; does not import mpi4py or LAMMPS."""
import json
import os
import socket

rank = next((os.environ[k] for k in ("PMI_RANK", "PMIX_RANK", "OMPI_COMM_WORLD_RANK") if k in os.environ), None)
print(json.dumps({"rank": int(rank) if rank is not None else None, "host": socket.gethostname(),
                  "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None}), flush=True)
