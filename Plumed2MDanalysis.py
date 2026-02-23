#!/usr/bin/env python3
import argparse
import re
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from collections import OrderedDict
import MDAnalysis as mda


def parse_plumed_distances(plumed_path):
    """
    Parse a PLUMED distances.dat file.
    Returns list of (dnum, a1, a2, label1, label2, line)
    """
    distances = []
    other_lines = []
    pattern = re.compile(
        r'd(\d+):\s*DISTANCE\s+ATOMS=(\d+),(\d+)\s*(?:#\s*(.+?)\s*-\s*(.+))?',
        re.IGNORECASE
    )
    with open(plumed_path, 'r') as fh:
        for line in fh:
            m = pattern.search(line)
            if m:
                dnum = int(m.group(1))
                a1 = int(m.group(2)) - 1  # PLUMED 1-based -> 0-based
                a2 = int(m.group(3)) - 1
                label1 = (m.group(4) or f"atom{a1+1}").strip()
                label2 = (m.group(5) or f"atom{a2+1}").strip()
                distances.append((dnum, a1, a2, label1, label2, line.strip()))
            else:
                other_lines.append(line)
    return distances, other_lines


def canonical_mapping_from_distances(distances_list_of_files):
    """
    Build canonical ordering for all unique label pairs across input lists.
    Returns OrderedDict: {(labelA,labelB): canonical_index}
    """
    all_pairs = set()
    for distances, _ in distances_list_of_files:
        for _, a1, a2, label1, label2, _ in distances:
            pair = tuple(sorted([label1, label2]))
            all_pairs.add(pair)
    sorted_pairs = sorted(all_pairs)
    mapping = OrderedDict((pair, i + 1) for i, pair in enumerate(sorted_pairs))
    return mapping


def build_distance_definitions(distances, canonical_map):
    """
    Map all distance definitions to canonical indices.
    """
    out = []
    for _, a1, a2, label1, label2, _ in distances:
        pair = tuple(sorted([label1, label2]))
        idx = canonical_map[pair]
        out.append((idx, a1, a2, label1, label2))
    out.sort(key=lambda x: x[0])
    return out


def compute_distances_vectorized(positions, atom_pairs):
    """
    Vectorized distance calculation using NumPy.
    """
    pairs_array = np.array(atom_pairs, dtype=np.int32)
    deltas = positions[pairs_array[:, 0]] - positions[pairs_array[:, 1]]
    distances = np.linalg.norm(deltas, axis=1)
    return distances


# Don't use multiple universes, as this could kill the RAM! 
def compute_distances_memory_safe(top, traj, plumed_path, out_prefix,
                                  chunk_size=1000, frames_subset=None):
    
    print("\nParsing PLUMED distances file...")
    distances, _ = parse_plumed_distances(plumed_path)
    if len(distances) == 0:
        raise RuntimeError("No DISTANCE lines found in PLUMED file.")

    canonical = canonical_mapping_from_distances([(distances, None)])
    distance_defs = build_distance_definitions(distances, canonical)

    n_dist = len(distance_defs)
    atom_pairs = [(a1, a2) for _, a1, a2, *_ in distance_defs]

    print("Loading trajectory...")
    u = mda.Universe(top, traj)
    total_frames = len(u.trajectory)
    times = np.array([ts.time for ts in u.trajectory])

    if frames_subset is not None:
        frame_indices_all = np.array(frames_subset, dtype=int)
    else:
        frame_indices_all = np.arange(total_frames)

    total_frames_to_compute = len(frame_indices_all)
    print(f"Total frames to compute: {total_frames_to_compute}")
    print(f"Total distances per frame: {n_dist}")

    npy_out = out_prefix + ".npy"
    print(f"Creating memmap file: {npy_out}")
    mm = np.memmap(npy_out, dtype=np.float32, mode='w+', shape=(total_frames_to_compute, n_dist))

    start_time = time.time()

    for chunk_start in tqdm(range(0, total_frames_to_compute, chunk_size), desc="Processing chunks"):
        chunk_end = min(chunk_start + chunk_size, total_frames_to_compute)
        chunk_indices = frame_indices_all[chunk_start:chunk_end]

        chunk_data = np.empty((len(chunk_indices), n_dist), dtype=np.float32)

        for i, frame_idx in enumerate(chunk_indices):
            u.trajectory[frame_idx]
            positions = u.atoms.positions
            chunk_data[i, :] = compute_distances_vectorized(positions, atom_pairs)

        mm[chunk_start:chunk_end, :] = chunk_data
        mm.flush()

    elapsed_time = time.time() - start_time
    print(f"Computation complete!")
    print(f"Elapsed time: {elapsed_time:.2f} seconds ({total_frames_to_compute / elapsed_time:.2f} frames/sec)")

    result_array = np.memmap(npy_out, dtype=np.float32, mode='r', shape=(total_frames_to_compute, n_dist))
    colnames = [f"d{i}" for i in range(1, n_dist + 1)]

    df = pd.DataFrame(result_array, columns=colnames)

    print(f"Saved NumPy memmap -> {npy_out}")
    print(f"Array shape: {result_array.shape} (frames x distances)")

    return df, canonical



def main():
    parser = argparse.ArgumentParser(
        description="Compute PLUMED-like distances using MDAnalysis with memory-efficient processing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--top", required=True, help="Topology file (pdb/gro/tpr)")
    parser.add_argument("--traj", required=True, help="Trajectory file (xtc/trr/etc.)")
    parser.add_argument("--plumed", required=True, help="PLUMED distances file (distances.dat)")
    parser.add_argument("--out", default="distances_out", help="Output prefix (CSV and npy will be written)")
    parser.add_argument("--chunk", type=int, default=1000, help="Number of frames per chunk (default: 1000)")
    parser.add_argument("--subset", type=str, default=None,
                        help="Optional frame subset: 'start:stop:step' or '0,10,20'")
    args = parser.parse_args()

    frames_subset = None
    if args.subset:
        if ":" in args.subset:
            parts = [int(p) if p else None for p in args.subset.split(":")]
            frames_subset = list(range(*(parts)))
        else:
            frames_subset = [int(x) for x in args.subset.split(",")]

    df, canonical = compute_distances_memory_safe(
        top=args.top,
        traj=args.traj,
        plumed_path=args.plumed,
        out_prefix=args.out,
        chunk_size=args.chunk,
        frames_subset=frames_subset,
    )

    print("\nAll done!")


if __name__ == "__main__":
    main()
