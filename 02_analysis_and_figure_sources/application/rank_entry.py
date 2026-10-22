#!/usr/bin/env python3
"""Pin one local MPI rank to one assigned OS CPU, then exec LAMMPS."""
import argparse
import os
import sys


def rank_index(environment, count):
    for name in ("OMPI_COMM_WORLD_RANK", "PMI_RANK", "PMIX_RANK"):
        if name in environment:
            rank = int(environment[name])
            if not 0 <= rank < count:
                raise ValueError("MPI rank outside CPU group; this workflow is single-host only")
            return rank
    if count == 1:
        return 0
    raise ValueError("MPI launcher did not expose a supported rank variable")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpus", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cpus = [int(value) for value in args.cpus.split(",")]
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or len(set(cpus)) != len(cpus):
        parser.error("A command and a list of distinct CPU IDs are required")
    rank = rank_index(os.environ, len(cpus))
    if not hasattr(os, "sched_setaffinity"):
        raise ValueError("Pinned concurrent runs require Linux CPU-affinity support")
    os.sched_setaffinity(0, {cpus[rank]})
    if os.sched_getaffinity(0) != {cpus[rank]}:
        raise ValueError("Failed to apply the requested CPU affinity")
    os.execv(command[0], command)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as error:
        print(f"Rank launch failed: {error}", file=sys.stderr)
        sys.exit(1)
