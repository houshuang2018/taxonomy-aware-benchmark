#!/usr/bin/env python3
"""
Compute sequence and biophysical features for a UniProt list.

Inputs
------
- UniProt ID list (one ID per line; optional extra columns — first column is used)
- Directory of per-ID FASTA files: {fasta_dir}/{uniprot_id}.fasta
- AIUPred binding output (-b -g 0), e.g.:
    python3 AIUPred-master/aiupred.py -i proteins.fasta -o bindingScore.txt -b -g 0

Output
------
DataFrame indexed by uniprot_id; columns are human-readable feature names.

Requires: pandas, biopython, localcider (SequenceParameters for kappa and Uversky hydropathy)
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import pandas as pd


_POSITIVE_FLAG = "1"
_NEGATIVE_FLAG = "0"


def dilate(states: str, max_length: int) -> str:
    """String-based dilation as in MobiDB-lite."""
    states = f"{_POSITIVE_FLAG * max_length}{states}{_POSITIVE_FLAG * max_length}"

    for level in range(1, max_length + 1):
        old = f"{_POSITIVE_FLAG * level}{_NEGATIVE_FLAG * level}{_POSITIVE_FLAG * level}"
        new = f"{_POSITIVE_FLAG * level}{_POSITIVE_FLAG * level}{_POSITIVE_FLAG * level}"
        for _ in range(level + 1):
            states = states.replace(old, new)

    return states[max_length:-max_length]


def erode(states: str, max_length: int) -> str:
    """String-based erosion as in MobiDB-lite."""
    states = f"{_NEGATIVE_FLAG * max_length}{states}{_NEGATIVE_FLAG * max_length}"

    for level in range(1, max_length + 1):
        old = f"{_NEGATIVE_FLAG * level}{_POSITIVE_FLAG * level}{_NEGATIVE_FLAG * level}"
        new = f"{_NEGATIVE_FLAG * level}{_NEGATIVE_FLAG * level}{_NEGATIVE_FLAG * level}"
        for _ in range(level + 1):
            states = states.replace(old, new)

    return states[max_length:-max_length]


def _repl_struct_by_disord(match: re.Match) -> str:
    """Replace a matched long-IDR / short-gap / long-IDR span with all '1'."""
    return _POSITIVE_FLAG * len(match.group(0))


def merge_long_disordered_regions(states: str) -> str:
    """Same as merge_long_disordered_regions in consensus.py (MobiDB-lite)."""
    pattern = r"{p}{{21,}}{n}{{1,10}}{p}{{21,}}".format(
        p=_POSITIVE_FLAG,
        n=_NEGATIVE_FLAG,
    )

    while True:
        new_states = re.sub(
            pattern=pattern,
            repl=_repl_struct_by_disord,
            string=states,
        )
        if new_states == states:
            return new_states
        states = new_states


def get_regions(states: str, min_length: int) -> List[Tuple[int, int, str]]:
    """
    Same as get_regions in consensus.py: return 0-based (start, end, flag) runs where
    flag != '0' and run length >= min_length.
    """
    regions = []
    start = None
    current_flag = None

    for i, flag in enumerate(states):
        if flag != current_flag:
            if start is not None and current_flag != _NEGATIVE_FLAG:
                end = i - 1
                length = end - start + 1
                if length >= min_length:
                    regions.append((start, end, current_flag))
            start = i
            current_flag = flag

    if start is not None and current_flag != _NEGATIVE_FLAG:
        end = len(states) - 1
        length = end - start + 1
        if length >= min_length:
            regions.append((start, end, current_flag))

    return regions


def mobidb_lite_postproc_from_aiupred(
    scores: Sequence[float], thr: float = 0.5
) -> List[Tuple[int, int]]:
    """
    MobiDB-lite post-processing on AIUPred scores:
      1) score>=thr -> '1' / '0'
      2) dilate(max_length=3)
      3) erode(max_length=3)
      4) merge_long_disordered_regions
      5) get_regions(min_length=20) — keep long IDRs only
    Returns: list of (start, end) inclusive 0-based intervals.
    """
    states = "".join(
        _POSITIVE_FLAG if s >= thr else _NEGATIVE_FLAG
        for s in scores
    )

    states = dilate(states, max_length=3)
    states = erode(states, max_length=3)
    states = merge_long_disordered_regions(states)

    regions = get_regions(states, min_length=20)
    return [(start, end) for (start, end, _) in regions]


def parse_aiupred_multi_fasta(path: str) -> List[Dict[str, Any]]:
    """Parse AIUPred multi-record output: disorder score (col 3) and anchor (col 4) per residue."""
    proteins: List[Dict[str, Any]] = []
    current_id = None
    current_scores = []
    current_scores_anchor = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith("# "):
                continue

            if line.startswith("#>"):
                if current_id is not None:
                    proteins.append({
                        "uniprot": current_id,
                        "scores": current_scores,
                        "anchor": current_scores_anchor,
                    })
                parts = line.lstrip("#>").split("|")
                if len(parts) >= 2:
                    current_id = parts[1]
                else:
                    current_id = line.lstrip("#>").strip()
                current_scores = []
                current_scores_anchor = []
            else:
                cols = line.split()
                if len(cols) < 4:
                    continue
                score = float(cols[2])
                current_scores.append(score)
                current_scores_anchor.append(float(cols[3]))

    if current_id is not None:
        proteins.append({
            "uniprot": current_id,
            "scores": current_scores,
            "anchor": current_scores_anchor,
        })

    return proteins


Region = Tuple[int, int]  # (start, end), inclusive, 0-based


def anchor_summary_from_idr_regions(
    anchor_scores: Sequence[float],
    regions_0based: List[Region],
    *,
    threshold: float = 0.5,
    ignore_nan: bool = True,
) -> float:
    """
    Fraction of residues (over full length) where ANCHOR/binding score > threshold
    within MobiDB-lite IDR regions: n_above_in_idr / L.
    """
    L = len(anchor_scores)

    def is_nan(x: float) -> bool:
        try:
            return math.isnan(float(x))
        except (TypeError, ValueError):
            return False

    if L == 0:
        return float("nan")

    n_above_in_idr = 0
    for start, end in regions_0based:
        start, end = int(start), int(end)
        if end < start:
            raise ValueError(f"Invalid region (end < start): {(start, end)}")
        if start >= L:
            continue
        s = max(0, start)
        e = min(L - 1, end)
        for i in range(s, e + 1):
            a = float(anchor_scores[i])
            if ignore_nan and is_nan(a):
                continue
            if a > threshold:
                n_above_in_idr += 1

    return n_above_in_idr / L


try:
    from localcider.sequenceParameters import SequenceParameters
except ImportError as e:
    SequenceParameters = None  # type: ignore
    _LOCALCIDER_IMPORT_ERROR = e
else:
    _LOCALCIDER_IMPORT_ERROR = None

# Internal column keys (stable); exported columns use FEATURE_LABELS values.
FEATURE_LABELS: Dict[str, str] = {
    "lcs_fractions": "Low-complexity score",
    "idr_50": "IDR proportion",
    "Negatively charged (DE)": "Negatively charged group",
    "Positively charged (KRH)": "Positively charged group",
    "Polar uncharged (STNQC)": "Polar uncharged group",
    "kappa_max_idr": "IDR Kappa",
    "n_above_0p5_in_idr_over_all": "IDR binding",
    "uversky_hydropathy_idr": "IDR hydropathy",
    "ncpr5": "Net charge per residue",
    "Hydrophilic": "Hydrophilic group",
    "Hydrophobic": "Hydrophobic group",
    "Aromatic (FWY)": "Aromatic group",
    "Aliphatic (AVLIM)": "Aliphatic group",
    "alpha_helix": "Alpha helix group",
}

FEATURE_KEYS = list(FEATURE_LABELS.keys())

RESIDUES = list("ACDEFGHIKLMNPQRSTVWY")

AA_GROUPS = {
    "Aliphatic (AVLIM)": set("AVLIM"),
    "Aromatic (FWY)": set("FWY"),
    "Polar uncharged (STNQC)": set("STNQC"),
    "Positively charged (KRH)": set("KRH"),
    "Negatively charged (DE)": set("DE"),
}


def read_uniprot_ids(path: str) -> List[str]:
    ids: List[str] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            ids.append(line.split()[0].strip())
    return ids


def read_fasta_sequence(path: str) -> str:
    """Return first sequence in FASTA, uppercase one-letter."""
    from Bio import SeqIO

    for record in SeqIO.parse(path, "fasta"):
        return str(record.seq).upper().replace("U", "").replace("*", "")
    return ""


def binding_dict_from_aiupred(path: str) -> Dict[str, dict]:
    """uniprot -> {scores: list[float], anchor: list[float]}"""
    proteins = parse_aiupred_multi_fasta(path)
    out: Dict[str, dict] = {}
    for p in proteins:
        uid = p["uniprot"]
        out[uid] = {"scores": p["scores"], "anchor": p["anchor"]}
    return out


def lcs_fraction(seq: str) -> float:
    """Same definition as 4.calculate_PSAP_features.MakeMatrix.add_lowcomplexity_features."""
    n_window = 20
    n_halfwindow = n_window // 2
    seq = str(seq).upper()
    L = len(seq)
    if L == 0:
        return float("nan")

    lc_bool = [False] * L
    for i in range(L):
        if i < n_halfwindow:
            peptide = seq[:n_window]
        elif i + n_halfwindow > L:
            peptide = seq[-n_window:]
        else:
            peptide = seq[i - n_halfwindow : i + n_halfwindow]
        complexity = len(set(peptide))
        if complexity <= 7:
            for edge_idx in (i - n_halfwindow, i + n_halfwindow):
                try:
                    lc_bool[edge_idx] = True
                except IndexError:
                    pass

    lc_frame = pd.DataFrame({"is_low_complexity": lc_bool, "acid": list(seq)})
    n_lc = int(lc_frame["is_low_complexity"].sum())
    return n_lc / L


def aa_group_fractions(seq: str) -> Dict[str, float]:
    seq = seq.upper()
    L = len(seq)
    if L == 0:
        return {k: float("nan") for k in AA_GROUPS}
    counts = Counter(seq)
    return {
        name: sum(counts[c] for c in chars) / L for name, chars in AA_GROUPS.items()
    }


def biochemical_hydrophilic_hydrophobic_alpha(seq: str) -> Dict[str, float]:
    """Matches 4.calculate_PSAP_features.add_biochemical_combinations."""
    seq = seq.upper()
    L = len(seq)
    if L == 0:
        return {
            "Hydrophilic": float("nan"),
            "Hydrophobic": float("nan"),
            "alpha_helix": float("nan"),
        }
    fr = {res: seq.count(res) / L for res in RESIDUES}
    hydrophilic = (
        fr["S"]
        + fr["T"]
        + fr["H"]
        + fr["N"]
        + fr["Q"]
        + fr["E"]
        + fr["D"]
        + fr["K"]
        + fr["R"]
    )
    hydrophobic = (
        fr["V"] + fr["I"] + fr["L"] + fr["F"] + fr["W"] + fr["Y"] + fr["M"]
    )
    alpha_helix = (
        fr["V"] + fr["I"] + fr["Y"] + fr["F"] + fr["W"] + fr["L"]
    )
    return {
        "Hydrophilic": hydrophilic,
        "Hydrophobic": hydrophobic,
        "alpha_helix": alpha_helix,
    }


def ncpr5(seq: str) -> float:
    """Net charge per residue (K+R+H vs D+E), percent scale — benchmark_physicochemical_properties_analyses2_IDR."""
    seq = seq.upper()
    if len(seq) == 0:
        return float("nan")
    ncpr_count = (
        seq.count("K") + seq.count("R") + seq.count("H")
    ) - (seq.count("D") + seq.count("E"))
    return (ncpr_count / len(seq)) * 100.0


@dataclass
class KappaSummary:
    idr_concat: str
    kappa_max: float


def _slice_region(
    seq: str, start: int, end: int, *, end_inclusive: bool
) -> str:
    """Return substring for [start, end] or [start, end) depending on end_inclusive."""
    if start < 0 or end < 0 or start >= len(seq):
        return ""
    if end_inclusive:
        end2 = min(end + 1, len(seq))
        if end2 <= start:
            return ""
        return seq[start:end2]
    end2 = min(end, len(seq))
    if end2 <= start:
        return ""
    return seq[start:end2]


def compute_idr_kappas(
    sequence: str,
    regions_0based: List[Tuple[int, int]],
    *,
    end_inclusive: bool = True,
    min_len: int = 20,
) -> KappaSummary:
    seq = (sequence or "").strip().upper()
    if not seq:
        return KappaSummary("", float("nan"))

    idr_seqs: List[str] = []
    for (s, e) in regions_0based:
        sub = _slice_region(seq, int(s), int(e), end_inclusive=end_inclusive)
        if len(sub) >= min_len:
            idr_seqs.append(sub)

    if not idr_seqs:
        return KappaSummary("", float("nan"))

    if SequenceParameters is None:
        return KappaSummary("".join(idr_seqs), float("nan"))

    per_kappa: List[float] = []
    for sub in idr_seqs:
        k = SequenceParameters(sub).get_kappa()
        try:
            k = float(k)
        except (TypeError, ValueError):
            k = float("nan")
        per_kappa.append(k)

    idr_concat = "".join(idr_seqs)

    valid = [
        k
        for k in per_kappa
        if k is not None and not (isinstance(k, float) and math.isnan(k))
    ]
    kappa_max = max(valid) if valid else float("nan")

    return KappaSummary(
        idr_concat=idr_concat,
        kappa_max=kappa_max,
    )


def uversky_hydropathy_idr(idr_concat: str) -> float:
    if not idr_concat or SequenceParameters is None:
        return float("nan")
    try:
        return float(SequenceParameters(idr_concat).get_uversky_hydropathy())
    except Exception:
        return float("nan")


def compute_row(
    uid: str,
    seq: str,
    scores: Sequence[float] | None,
    anchor: Sequence[float] | None,
    *,
    disorder_thr: float = 0.5,
) -> Dict[str, float]:
    """Return internal-key feature dict for one protein."""
    row: Dict[str, float] = {k: float("nan") for k in FEATURE_KEYS}

    row["lcs_fractions"] = lcs_fraction(seq)
    ag = aa_group_fractions(seq)
    for k in (
        "Negatively charged (DE)",
        "Positively charged (KRH)",
        "Polar uncharged (STNQC)",
        "Aromatic (FWY)",
        "Aliphatic (AVLIM)",
    ):
        row[k] = ag[k]

    bio = biochemical_hydrophilic_hydrophobic_alpha(seq)
    row["Hydrophilic"] = bio["Hydrophilic"]
    row["Hydrophobic"] = bio["Hydrophobic"]
    row["alpha_helix"] = bio["alpha_helix"]

    # Full-sequence net charge (same convention as benchmark CSV column "ncpr5")
    row["ncpr5"] = ncpr5(seq)

    if scores is None or anchor is None:
        return row

    len_scores = len(scores)
    len_anchor = len(anchor)
    len_seq = len(seq)
    if len_scores == 0 or len_anchor != len_scores:
        return row
    if len_seq > 0 and len_scores != len_seq:
        return row

    regions = mobidb_lite_postproc_from_aiupred(list(scores), thr=disorder_thr)
    row["n_above_0p5_in_idr_over_all"] = anchor_summary_from_idr_regions(
        list(anchor), regions, threshold=disorder_thr
    )

    # idr_50: fraction of residues with disorder score > thr (full length)
    row["idr_50"] = sum(1 for s in scores if float(s) > disorder_thr) / len_scores

    kout = compute_idr_kappas(seq, regions, end_inclusive=True, min_len=20)
    row["kappa_max_idr"] = kout.kappa_max
    row["uversky_hydropathy_idr"] = uversky_hydropathy_idr(kout.idr_concat)

    return row


def build_feature_dataframe(
    uniprot_ids: List[str],
    fasta_dir: str,
    binding_path: str,
    *,
    disorder_thr: float = 0.5,
) -> pd.DataFrame:
    binding = binding_dict_from_aiupred(binding_path)
    rows: List[Dict[str, float]] = []
    index: List[str] = []

    for uid in uniprot_ids:
        fp = os.path.join(fasta_dir, f"{uid}.fasta")
        if not os.path.isfile(fp):
            raise FileNotFoundError(f"Missing FASTA for {uid}: {fp}")

        seq = read_fasta_sequence(fp)
        b = binding.get(uid)
        scores = b["scores"] if b else None
        anchor = b["anchor"] if b else None

        feat = compute_row(uid, seq, scores, anchor, disorder_thr=disorder_thr)
        rows.append(feat)
        index.append(uid)

    df = pd.DataFrame(rows, index=index)
    df.index.name = "uniprot_id"
    return df[FEATURE_KEYS]


def build_renamed_feature_dataframe(
    uniprot_ids: List[str],
    fasta_dir: str,
    binding_path: str,
    *,
    disorder_thr: float = 0.5,
) -> pd.DataFrame:
    """Same as build_feature_dataframe but columns use FEATURE_LABELS (human-readable)."""
    return build_feature_dataframe(
        uniprot_ids, fasta_dir, binding_path, disorder_thr=disorder_thr
    ).rename(columns=FEATURE_LABELS)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compute selected IDR / composition features for a UniProt list."
    )
    ap.add_argument(
        "--uniprot_list",
        required=True,
        help="Text file: one UniProt ID per line (first column if tab-separated).",
    )
    ap.add_argument(
        "--fasta_dir",
        required=True,
        help="Directory containing {uniprot_id}.fasta",
    )
    ap.add_argument(
        "--binding_score_txt",
        required=True,
        help="AIUPred output with binding/anchor (-b -g 0).",
    )
    ap.add_argument(
        "--out_csv",
        default="feature_matrix.csv",
        help="Output CSV path (index = uniprot_id).",
    )
    ap.add_argument(
        "--disorder_thr",
        type=float,
        default=0.5,
        help="Threshold for disorder and binding counts (default 0.5).",
    )
    args = ap.parse_args()

    if _LOCALCIDER_IMPORT_ERROR is not None:
        print(
            "Warning: localcider not importable; IDR Kappa and Uversky will be NaN. "
            f"({_LOCALCIDER_IMPORT_ERROR})",
            file=sys.stderr,
        )

    uids = read_uniprot_ids(args.uniprot_list)
    if not uids:
        sys.exit("No UniProt IDs read from --uniprot_list")

    df_out = build_renamed_feature_dataframe(
        uids,
        args.fasta_dir,
        args.binding_score_txt,
        disorder_thr=args.disorder_thr,
    )
    df_out.to_csv(args.out_csv, index=True)
    print(f"[Saved] {args.out_csv}  shape={df_out.shape}")


if __name__ == "__main__":
    main()
