#!/usr/bin/env python3
"""Build the oligomer prediction metrics shown on each PHROG's page.

Usage:
    scripts/build_metrics.py PICKLE_DIR OLIGOMER_PDB_DIR [--site metrics] [--full FULL_DIR] [--groups 0,5]

PICKLE_DIR holds Susie's phrog_<n>.pkl files. Each is a dict
{"<k>_subunits": {"rep_<r>": {metric: value}}} for k = 2..10 and ColabFold ranks r = 1..5, with
scalar metrics (iptm, ptm, max_pae, mean_plddt, median_plddt) and per-chain lists (pdockq2,
avgif_pae, plddt_lst, n_contacts_<d>A). The pickles are read with a restricted unpickler that only
accepts numpy float64 scalars, so loading them cannot run code.

OLIGOMER_PDB_DIR is a copy of oligomer_pdb/ from the Acacia bucket. The stoichiometry assigned to a
PHROG is the number of chains in its oligomer structure (the structure is rank 1 of that
stoichiometry); a PHROG with metrics but no oligomer structure was assigned a monomer.

--site writes two files per group, read by index.html. <site>/phrog_NNN.json holds rank 1:
    {"cols": [...], "phrogs": {"505": {"a": <assigned subunits, 1 = monomer>,
                                       "s": [<rank-1 row for 2 subunits>, ..., <10 subunits>]}}}
Each row holds the values named in "cols" (per-chain metrics averaged over chains), or null when
that stoichiometry was not predicted. <site>/ranks/phrog_NNN.json holds ranks 2-5, fetched only
when a reader expands a row: {"cols": [...], "phrogs": {"505": [[<rank 2 row>, ..., <rank 5 row>]
for 2..10 subunits, or null]}}.

--full writes every metric for every rank (floats rounded to 4 decimal places), as <full>/phrog_NNN/phrog_NNNNN.json, for upload to
metrics/ in the public bucket.

--groups limits the build to some groups (e.g. 0,5 for phrog_000 and phrog_005). The dev site uses
this to keep its Pages deploy small; production builds every group.
"""
import argparse
import gzip
import json
import os
import pickle
import re
import struct
import sys

NGROUPS = 389
SUBUNITS = range(2, 11)
COLS = ["iptm", "ptm", "pdockq2", "mean_plddt", "avgif_pae"]
DIGITS = {"iptm": 2, "ptm": 2, "pdockq2": 3, "mean_plddt": 1, "avgif_pae": 3}


class _DType:
    def __init__(self, name, *args):
        self.name, self.state = name, None

    def __setstate__(self, state):
        self.state = state


def _scalar(dtype, data):
    if dtype.name != "f8" or len(data) != 8:
        raise pickle.UnpicklingError(f"unexpected numpy scalar {dtype.name}")
    order = dtype.state[1] if isinstance(dtype.state, tuple) else "<"
    return struct.unpack((">" if order == ">" else "<") + "d", data)[0]


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ("numpy", "dtype"):
            return _DType
        if module in ("numpy._core.multiarray", "numpy.core.multiarray") and name == "scalar":
            return _scalar
        raise pickle.UnpicklingError(f"forbidden global {module}.{name}")


def load(path):
    with open(path, "rb") as f:
        return SafeUnpickler(f).load()


def chain_count(path):
    chains = set()
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith("ATOM"):
                chains.add(line[21])
    return len(chains)


def rounded(x):
    if isinstance(x, float):
        return round(x, 4)
    if isinstance(x, dict):
        return {k: rounded(v) for k, v in x.items()}
    if isinstance(x, list):
        return [rounded(v) for v in x]
    return x


def summary_row(rep):
    row = []
    for c in COLS:
        v = rep[c]
        if isinstance(v, list):
            v = sum(v) / len(v)
        row.append(round(v, DIGITS[c]))
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("pickles")
    ap.add_argument("oligomer_pdb")
    ap.add_argument("--site", default="metrics")
    ap.add_argument("--full")
    ap.add_argument("--groups", type=lambda v: {int(g) for g in v.split(",")},
                    default=set(range(NGROUPS)))
    args = ap.parse_args()

    assigned = {}
    for dp, _, fs in os.walk(args.oligomer_pdb):
        for f in fs:
            m = re.fullmatch(r"phrog_(\d{5})\.pdb\.gz", f)
            if m and int(m.group(1)) // 100 in args.groups:
                assigned[int(m.group(1))] = chain_count(os.path.join(dp, f))

    groups = [{} for _ in range(NGROUPS)]
    ranks = [{} for _ in range(NGROUPS)]
    bad = []
    for f in sorted(os.listdir(args.pickles)):
        m = re.fullmatch(r"phrog_(\d+)\.pkl", f)
        if not m or int(m.group(1)) // 100 not in args.groups:
            continue
        n = int(m.group(1))
        try:
            d = load(os.path.join(args.pickles, f))
            rows = [summary_row(d[f"{k}_subunits"]["rep_1"]) if f"{k}_subunits" in d else None
                    for k in SUBUNITS]
            others = [[summary_row(d[f"{k}_subunits"][f"rep_{r}"]) for r in range(2, 6)
                       if f"rep_{r}" in d[f"{k}_subunits"]] if f"{k}_subunits" in d else None
                      for k in SUBUNITS]
        except (pickle.UnpicklingError, KeyError, TypeError, ValueError, ZeroDivisionError) as e:
            bad.append(f"{f}: {e}")
            continue
        a = assigned.get(n, 1)
        if a > 1 and rows[a - 2] is None:
            bad.append(f"{f}: assigned {a} subunits but has no {a}_subunits metrics")
        groups[n // 100][str(n)] = {"a": a, "s": rows}
        ranks[n // 100][str(n)] = others
        if args.full:
            out = os.path.join(args.full, f"phrog_{n // 100:03d}", f"phrog_{n:05d}.json")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out, "w") as h:
                json.dump({"phrog": n, "assigned_subunits": a, "predictions": rounded(d)}, h, separators=(",", ":"))

    for sub, data in (("", groups), ("ranks", ranks)):
        os.makedirs(os.path.join(args.site, sub), exist_ok=True)
        for g, phrogs in enumerate(data):
            path = os.path.join(args.site, sub, f"phrog_{g:03d}.json")
            if phrogs:
                with open(path, "w") as h:
                    json.dump({"cols": COLS, "phrogs": phrogs}, h, separators=(",", ":"))
                    h.write("\n")
            elif os.path.exists(path):
                os.remove(path)

    with_metrics = {int(p) for grp in groups for p in grp}
    missing = sorted(set(assigned) - with_metrics)
    print(f"{len(with_metrics)} PHROGs with metrics ({sum(1 for p in with_metrics if p in assigned)} oligomers); "
          f"{len(missing)} oligomer structures without metrics {missing[:10]}; {len(bad)} problems",
          file=sys.stderr)
    for b in bad:
        print("  " + b, file=sys.stderr)
    return 1 if bad or missing else 0


if __name__ == "__main__":
    sys.exit(main())
