#!/usr/bin/env python3
"""
Usage (examples)
----------------
  python3 predictor_evaluation.py \\
      --scores predictor_scores_excludeTraining.csv \\
      --output example_out.csv \\
      --organism Plants

  # Reproducibility and subsampling (defaults: n_repeats=10, sub_ratio=1.0, random_seed=42)
  python3 predictor_evaluation.py --n-repeats 10 --sub-ratio 1.0 --random-seed 42

  # Optional external binary predictors
  python3 predictor_evaluation.py \\
      --scores predictor_scores_excludeTraining.csv \\
      --output example_out.csv \\
      --organism Plants \\
      --flfb-csv FLFB_scores.csv \\
      --plaac-csv PLAAC_scores.csv

FLFB / PLAAC optional files (comma-separated CSV)
-------------------------------------------------
``--flfb-csv`` must include columns:

  - ``uniprot`` — UniProt accession
  - ``Class`` — label string; a row counts as positive when ``Class == "LLPS"``

``--plaac-csv`` must include columns:

  - ``uniprot`` — UniProt accession
  - ``COREscore`` — numeric or empty; a row counts as positive when ``COREscore`` is not NaN

What this does
--------------
Input: predictor_scores_excludeTraining.csv (wide table with uniprot, tag, organism,
       IDP_type, model score columns, ...).

Outputs
-------
1. The path given by ``--output``: summary CSV with columns matching example.csv (mean/std over
   repeats per Model).

2. ``neg_subsample_uids.csv`` — intermediate file listing subsampled negative ``uniprot`` IDs per
   organism and repeat (columns typically include ``Organism``, ``Repeat``, ``uniprot``). Written
   under ``--work-dir`` by default, or to the path set with ``--neg-csv``. Required for the MCC /
   metrics steps inside the same run.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# Negative subsample by organism (valid-set check + IDP-stratified downsampling).
# ---------------------------------------------------------------------------


def infer_model_columns(df: pd.DataFrame) -> List[str]:
    exclude = {
        "uniprot",
        "tag",
        "organism",
        "Organism",
        "IDP_type",
        "Repeat",
        "repeat",
    }
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def compute_model_performance_by_organism(
    data1: pd.DataFrame,
    *,
    id_col: str = "uniprot",
    tag_col: str = "tag",
    organism_col: str = "organism",
    organisms: Optional[List[str]] = None,
    models: Optional[List[str]] = None,
    n_repeats: int = 10,
    random_seed: int = 42,
    output_subsampled_neg_file: Optional[str] = None,
    idp_type_col: str = "IDP_type",
    idp_value: str = "IDP",
    non_idp_value: str = "non-IDP",
    sub_ratio: float = 1.0,
) -> None:
    """
    Cross-model valid-sample check, IDP-stratified negative subsampling, and optional
    ``output_subsampled_neg_file``.
    """
    if n_repeats <= 0:
        raise ValueError("n_repeats must be >= 1.")
    if sub_ratio <= 0:
        raise ValueError("sub_ratio must be > 0.")

    if organisms is None:
        organisms = sorted(data1[organism_col].dropna().astype(str).unique().tolist())

    if models is None:
        models = infer_model_columns(data1)
    else:
        models = [m for m in models if m in data1.columns]

    if len(models) == 0:
        raise ValueError("No numeric model columns found in data1 (or none left after filtering).")

    valid_sets_by_org_model: Dict[Tuple[str, str], set] = {}

    for org in organisms:
        d_org = data1.loc[data1[organism_col] == org, :].copy()
        if d_org.empty:
            continue

        y_all = pd.to_numeric(d_org[tag_col], errors="coerce")

        for m in models:
            s_all = pd.to_numeric(d_org[m], errors="coerce")
            valid = (~y_all.isna()) & (~s_all.isna()) & (y_all.isin([0, 1]))
            valid_uniprots = set(d_org.loc[valid, id_col].astype(str).tolist())
            valid_sets_by_org_model[(org, m)] = valid_uniprots

    for org in organisms:
        org_valid_sets = [valid_sets_by_org_model.get((org, m), set()) for m in models]
        if len(org_valid_sets) == 0:
            continue

        first_valid_set = org_valid_sets[0]
        for idx, valid_set in enumerate(org_valid_sets[1:], 1):
            if valid_set != first_valid_set:
                missing_in_model = first_valid_set - valid_set
                extra_in_model = valid_set - first_valid_set
                error_msg = f"\nError: valid samples for organism '{org}' differ across models.\n"
                error_msg += f"Model '{models[0]}' has {len(first_valid_set)} valid samples\n"
                error_msg += f"Model '{models[idx]}' has {len(valid_set)} valid samples\n"
                if missing_in_model:
                    error_msg += f"\nModel '{models[idx]}' is missing scores for {len(missing_in_model)} uniprot IDs:\n"
                    missing_list = sorted(list(missing_in_model))[:50]
                    error_msg += ", ".join(missing_list)
                    if len(missing_in_model) > 50:
                        error_msg += f"\n... and {len(missing_in_model) - 50} more not shown"
                if extra_in_model:
                    error_msg += f"\nModel '{models[idx]}' has {len(extra_in_model)} extra uniprot IDs:\n"
                    extra_list = sorted(list(extra_in_model))[:50]
                    error_msg += ", ".join(extra_list)
                    if len(extra_in_model) > 50:
                        error_msg += f"\n... and {len(extra_in_model) - 50} more not shown"
                raise ValueError(error_msg)

    rng = np.random.default_rng(random_seed)
    subsampled_neg_records: List[dict] = []

    for org in organisms:
        d_org = data1.loc[data1[organism_col] == org, :].copy()
        if d_org.empty:
            continue

        y_all = pd.to_numeric(d_org[tag_col], errors="coerce")

        if idp_type_col in d_org.columns:
            idp_all_raw = d_org[idp_type_col]
        else:
            idp_all_raw = pd.Series([np.nan] * len(d_org), index=d_org.index)

        if len(models) > 0:
            s_all_first = pd.to_numeric(d_org[models[0]], errors="coerce")
            valid_common = (~y_all.isna()) & (~s_all_first.isna()) & (y_all.isin([0, 1]))
        else:
            valid_common = pd.Series([False] * len(d_org), index=d_org.index)

        n_common = int(valid_common.sum())

        if n_common == 0:
            y_common = np.array([], dtype=int)
            idp_arr_common = np.array([], dtype=object)
        else:
            y_common = y_all.loc[valid_common].astype(int).to_numpy()
            idp_arr_common = idp_all_raw.loc[valid_common].fillna("").astype(str).to_numpy()

        idx_pos_idp_common = np.where((y_common == 1) & (idp_arr_common == idp_value))[0]
        idx_pos_non_common = np.where((y_common == 1) & (idp_arr_common == non_idp_value))[0]
        n_pos_idp_common = int(idx_pos_idp_common.size)
        n_pos_non_common = int(idx_pos_non_common.size)

        idx_neg_idp_common = np.where((y_common == 0) & (idp_arr_common == idp_value))[0]
        idx_neg_non_common = np.where((y_common == 0) & (idp_arr_common == non_idp_value))[0]
        n_neg_idp_common = int(idx_neg_idp_common.size)
        n_neg_non_common = int(idx_neg_non_common.size)

        d_org_valid_common = d_org.loc[valid_common].copy()

        n_neg_idp_target = int(np.round(n_pos_idp_common * float(sub_ratio))) if n_pos_idp_common > 0 else 0
        n_neg_non_target = int(np.round(n_pos_non_common * float(sub_ratio))) if n_pos_non_common > 0 else 0

        do_down_idp = (
            n_common > 0
            and n_pos_idp_common > 0
            and idx_neg_idp_common.size > 0
            and idx_neg_idp_common.size > n_neg_idp_target
        )
        do_down_non = (
            n_common > 0
            and n_pos_non_common > 0
            and idx_neg_non_common.size > 0
            and idx_neg_non_common.size > n_neg_non_target
        )

        if do_down_idp or do_down_non:
            if do_down_idp:
                n_neg_idp_keep_org = max(1, min(n_neg_idp_target, idx_neg_idp_common.size))
            else:
                n_neg_idp_keep_org = idx_neg_idp_common.size

            if do_down_non:
                n_neg_non_keep_org = max(1, min(n_neg_non_target, idx_neg_non_common.size))
            else:
                n_neg_non_keep_org = idx_neg_non_common.size

            idp_order = rng.permutation(idx_neg_idp_common) if idx_neg_idp_common.size > 0 else np.array([], dtype=int)
            non_order = rng.permutation(idx_neg_non_common) if idx_neg_non_common.size > 0 else np.array([], dtype=int)
            idp_cursor = 0
            non_cursor = 0

            for repeat_idx in range(int(n_repeats)):
                if idx_neg_idp_common.size > 0 and n_neg_idp_keep_org > 0:
                    take_idp = min(n_neg_idp_keep_org, idx_neg_idp_common.size)
                    if idp_order.size - idp_cursor >= take_idp:
                        chosen_neg_idp = idp_order[idp_cursor : idp_cursor + take_idp]
                        idp_cursor += take_idp
                    else:
                        idp_order = rng.permutation(idx_neg_idp_common)
                        idp_cursor = take_idp
                        chosen_neg_idp = idp_order[:take_idp]
                else:
                    chosen_neg_idp = np.array([], dtype=int)

                if idx_neg_non_common.size > 0 and n_neg_non_keep_org > 0:
                    take_non = min(n_neg_non_keep_org, idx_neg_non_common.size)
                    if non_order.size - non_cursor >= take_non:
                        chosen_neg_non = non_order[non_cursor : non_cursor + take_non]
                        non_cursor += take_non
                    else:
                        non_order = rng.permutation(idx_neg_non_common)
                        non_cursor = take_non
                        chosen_neg_non = non_order[:take_non]
                else:
                    chosen_neg_non = np.array([], dtype=int)

                chosen_neg = np.concatenate([chosen_neg_idp, chosen_neg_non])

                if output_subsampled_neg_file is not None:
                    chosen_neg_df = d_org_valid_common.iloc[chosen_neg]
                    uniprot_ids = chosen_neg_df[id_col].tolist()
                    for uniprot_id in uniprot_ids:
                        subsampled_neg_records.append(
                            {"Organism": org, "Repeat": repeat_idx + 1, id_col: uniprot_id}
                        )
        else:
            for repeat_idx in range(int(n_repeats)):
                chosen_neg = np.concatenate([idx_neg_idp_common, idx_neg_non_common])

                if output_subsampled_neg_file is not None:
                    chosen_neg_df = d_org_valid_common.iloc[chosen_neg]
                    uniprot_ids = chosen_neg_df[id_col].tolist()
                    for uniprot_id in uniprot_ids:
                        subsampled_neg_records.append(
                            {"Organism": org, "Repeat": repeat_idx + 1, id_col: uniprot_id}
                        )

    if output_subsampled_neg_file is not None and len(subsampled_neg_records) > 0:
        df_subsampled = pd.DataFrame(subsampled_neg_records)
        output_dir = os.path.dirname(output_subsampled_neg_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        df_subsampled.to_csv(output_subsampled_neg_file, index=False)


# ---------------------------------------------------------------------------
# MCC threshold helpers
# ---------------------------------------------------------------------------


def prepare_mcc_input_from_tag_wide(
    df: pd.DataFrame,
    *,
    tag_col: str = "tag",
    model_cols: Optional[List[str]] = None,
    tag_mapping: Optional[Mapping[Any, int]] = None,
    positive_default_values=("PSP", "PSPs", "positive", "pos", "1"),
    exclude_cols: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    default_exclude = {
        tag_col,
        "uniprot",
        "Organism",
        "organism",
        "Repeat",
        "repeat",
        "y_true",
        "y_true_tmp",
    }
    if exclude_cols is not None:
        default_exclude |= set(exclude_cols)

    if model_cols is None:
        candidate_cols = [c for c in df.columns if c not in default_exclude]
        model_cols = [c for c in candidate_cols if pd.api.types.is_numeric_dtype(df[c])]

    tag_series = df[tag_col]

    if tag_mapping is not None:
        y_true = tag_series.map(tag_mapping)
        if y_true.isna().any():
            unknown = tag_series[y_true.isna()].unique()
            raise ValueError(f"tag_mapping does not cover tag values: {unknown}")
        y_true = y_true.astype(int)
    else:
        if pd.api.types.is_numeric_dtype(tag_series):
            y_true = tag_series.astype(int)
        else:
            lower_pos = {str(v).lower() for v in positive_default_values}

            def auto_map(v: Any) -> Optional[int]:
                if pd.isna(v):
                    return None
                s = str(v).strip().lower()
                return 1 if s in lower_pos else 0

            y_true = tag_series.map(auto_map)
            if y_true.isna().any():
                raise ValueError(
                    "NaN while auto-mapping tag; unrecognized tag values may be present. "
                    "Provide an explicit tag_mapping."
                )
            y_true = y_true.astype(int)

    df_tmp = df.copy()
    df_tmp["y_true"] = y_true

    df_long = df_tmp.melt(
        id_vars=["y_true"],
        value_vars=model_cols,
        var_name="Model",
        value_name="score",
    ).dropna(subset=["score"])

    return df_long


def safe_mcc(y_true: Union[np.ndarray, List[Any]], y_pred: Union[np.ndarray, List[Any]]) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if np.unique(y_true).size < 2 or np.unique(y_pred).size < 2:
        return 0.0
    return float(matthews_corrcoef(y_true, y_pred))


def compute_mcc_max_threshold_wide(
    df_long: pd.DataFrame,
    *,
    model_col: str = "Model",
    y_col: str = "y_true",
    score_col: str = "score",
    threshold_grid: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Threshold that maximizes MCC on the grid (default: 0.01 to 0.99 with step 0.01).
    """
    if threshold_grid is None:
        threshold_grid = np.arange(0.01, 1.00, 0.01)
    rows: List[dict] = []
    for m in df_long[model_col].dropna().unique():
        sub = df_long[df_long[model_col] == m][[y_col, score_col]].dropna()
        if sub.empty:
            continue

        y_true = sub[y_col].values
        scores = sub[score_col].values

        best_mcc = -np.inf
        best_thr = np.nan
        for thr in threshold_grid:
            mcc_t = safe_mcc(y_true, (scores >= thr).astype(int))
            if mcc_t > best_mcc:
                best_mcc = mcc_t
                best_thr = float(thr)

        rows.append({"Model": m, "Thr_max": best_thr, "MCC_max": best_mcc})

    return pd.DataFrame(rows)


def mcc_wide_for_repeats(
    scores_org: pd.DataFrame,
    neg_sub_org: pd.DataFrame,
    tag_mapping: Mapping[Any, int],
) -> pd.DataFrame:
    repeats = sorted(neg_sub_org["Repeat"].unique())
    all_wide: List[pd.DataFrame] = []
    for r in repeats:
        neg_r = neg_sub_org[neg_sub_org["Repeat"] == r].copy()
        orgs_r = neg_r["Organism"].unique()

        neg_r_df = scores_org.merge(
            neg_r[["Organism", "uniprot"]],
            on=["Organism", "uniprot"],
            how="inner",
        )

        tmp = scores_org.copy()
        tmp["y_true_tmp"] = tmp["tag"].map(tag_mapping)
        pos_r_df = tmp[(tmp["y_true_tmp"] == 1) & (tmp["Organism"].isin(orgs_r))].drop(columns=["y_true_tmp"])

        df_r = pd.concat([pos_r_df, neg_r_df], ignore_index=True)
        df_r["Repeat"] = r

        df_r_long_input = prepare_mcc_input_from_tag_wide(
            df_r,
            tag_col="tag",
            model_cols=None,
            tag_mapping=tag_mapping,
        )
        df_wide_r = compute_mcc_max_threshold_wide(df_r_long_input)
        df_wide_r["Repeat"] = r
        all_wide.append(df_wide_r)

    return pd.concat(all_wide, ignore_index=True)


# ---------------------------------------------------------------------------
# All metrics — optional FLFB / PLAAC files
# ---------------------------------------------------------------------------

FIXED_THRESHOLD_MODELS: Tuple[str, ...] = ("PScore", "FuzDrop")
FIXED_THRESHOLD_BY_MODEL: Dict[str, float] = {"PScore": 4.0, "FuzDrop": 0.6}
FLFB_MODEL_NAME = "FLFB"
PLAAC_MODEL_NAME = "PLAAC"


def _safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return float("nan")


def _safe_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(average_precision_score(y_true, y_score))
    except ValueError:
        return float("nan")


def calc_auprc_ratio(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    prevalence = y_true.mean()
    if prevalence == 0:
        raise ValueError("y_true has no positive samples; cannot compute AUPRC ratio vs. prevalence.")
    auprc = average_precision_score(y_true, y_score)
    return auprc / prevalence


def _confusion_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    mcc = float(matthews_corrcoef(y_true, y_pred)) if len(y_true) > 0 else float("nan")
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))
    spec = (tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
    return {
        "MCC": mcc,
        "F1": f1,
        "Precision": prec,
        "Recall": rec,
        "Specificity": float(spec),
        "Accuracy": acc,
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
    }


def calc_positive_likelihood_ratio(
    y_true: np.ndarray,
    y_score: Optional[np.ndarray] = None,
    y_pred: Optional[np.ndarray] = None,
    threshold: float = 0.5,
) -> float:
    y_true = np.asarray(y_true)
    if y_pred is None:
        if y_score is None:
            raise ValueError("Provide either y_score or y_pred.")
        y_score = np.asarray(y_score)
        y_pred = (y_score >= threshold).astype(int)
    else:
        y_pred = np.asarray(y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    if (tp + fn) == 0:
        raise ValueError("y_true has no positive samples; cannot compute sensitivity.")
    sensitivity = tp / (tp + fn)
    if (tn + fp) == 0:
        raise ValueError("y_true has no negative samples; cannot compute specificity.")
    specificity = tn / (tn + fp)
    fpr = 1 - specificity
    if fpr == 0:
        return float("inf")
    return float(sensitivity / fpr)


def _make_uniprot_bool_map(df: pd.DataFrame, uniprot_col: str, is_pos: pd.Series) -> pd.Series:
    tmp = df[[uniprot_col]].copy()
    tmp["__pos__"] = is_pos.astype(int).values
    return tmp.groupby(uniprot_col, as_index=True)["__pos__"].max()


def load_optional_binary_maps(
    flfb_path: Optional[str],
    plaac_path: Optional[str],
) -> Tuple[Optional[pd.Series], Optional[pd.Series]]:
    flfb_map = None
    plaac_map = None
    if flfb_path and os.path.isfile(flfb_path):
        flfb_class_df = pd.read_csv(flfb_path)
        for col in ("uniprot", "Class"):
            if col not in flfb_class_df.columns:
                raise ValueError(f"FLFB CSV {flfb_path!r} missing required column {col!r}")
        flfb_map = _make_uniprot_bool_map(
            flfb_class_df,
            uniprot_col="uniprot",
            is_pos=(flfb_class_df["Class"] == "LLPS"),
        )
    if plaac_path and os.path.isfile(plaac_path):
        plaac_df = pd.read_csv(plaac_path)
        for col in ("uniprot", "COREscore"):
            if col not in plaac_df.columns:
                raise ValueError(f"PLAAC CSV {plaac_path!r} missing required column {col!r}")
        plaac_map = _make_uniprot_bool_map(
            plaac_df,
            uniprot_col="uniprot",
            is_pos=~plaac_df["COREscore"].isna(),
        )
    return flfb_map, plaac_map


def calculate_metrics(
    df_thr: pd.DataFrame,
    df_subsample: pd.DataFrame,
    df_scores: pd.DataFrame,
    flfb_uid2bin: Optional[pd.Series],
    plaac_uid2bin: Optional[pd.Series],
) -> pd.DataFrame:
    df_thr = df_thr.copy()
    df_subsample = df_subsample.copy()
    df_scores = df_scores.copy()

    df_thr["Repeat"] = df_thr["Repeat"].astype(str)
    df_subsample["Repeat"] = df_subsample["Repeat"].astype(str)

    df_pos_all = df_scores[df_scores["tag"] == 1][["uniprot", "tag"]].copy()

    results: List[dict] = []

    for rep, df_rep_neg_uids in df_subsample.groupby("Repeat"):
        neg_uids = df_rep_neg_uids["uniprot"].unique()

        df_neg_rep = df_scores[(df_scores["tag"] == 0) & (df_scores["uniprot"].isin(neg_uids))][
            ["uniprot", "tag"]
        ].copy()

        if df_neg_rep.empty:
            continue

        df_eval_base = pd.concat([df_pos_all, df_neg_rep], ignore_index=True)
        eval_uids = df_eval_base["uniprot"].unique()

        df_eval_truth = df_scores[df_scores["uniprot"].isin(eval_uids)][["uniprot", "tag"]].drop_duplicates("uniprot")
        df_eval_truth = df_eval_truth.set_index("uniprot").reindex(eval_uids)
        y_true = df_eval_truth["tag"].to_numpy()

        df_thr_rep = df_thr[df_thr["Repeat"] == rep]

        models_rep = set(df_thr_rep["Model"].unique())
        models_rep.update(FIXED_THRESHOLD_MODELS)
        if FLFB_MODEL_NAME in df_scores.columns and flfb_uid2bin is not None:
            models_rep.add(FLFB_MODEL_NAME)
        if PLAAC_MODEL_NAME in df_scores.columns and plaac_uid2bin is not None:
            models_rep.add(PLAAC_MODEL_NAME)

        for model in sorted(models_rep):
            if model in df_scores.columns:
                df_eval_score = df_scores[df_scores["uniprot"].isin(eval_uids)][["uniprot", model]].drop_duplicates(
                    "uniprot"
                )
                df_eval_score = df_eval_score.set_index("uniprot").reindex(eval_uids)
                y_score = df_eval_score[model].to_numpy()
                if np.all(pd.isna(y_score)):
                    auroc = float("nan")
                    auprc = float("nan")
                    auprc_ratio = float("nan")
                else:
                    mask = ~pd.isna(y_score) & ~pd.isna(y_true)
                    auroc = _safe_auroc(y_true[mask], y_score[mask])
                    auprc = _safe_auprc(y_true[mask], y_score[mask])
                    auprc_ratio = calc_auprc_ratio(y_true[mask], y_score[mask])
            else:
                auroc = float("nan")
                auprc = float("nan")
                auprc_ratio = float("nan")
                y_score = None

            pred_source = ""
            thr_used = float("nan")

            if model == FLFB_MODEL_NAME and flfb_uid2bin is not None:
                pred_source = "FLFB_classfile(Class==LLPS)"
                y_pred = pd.Series(flfb_uid2bin, dtype=float).reindex(eval_uids).fillna(0).astype(int).to_numpy()

            elif model == PLAAC_MODEL_NAME and plaac_uid2bin is not None:
                pred_source = "PLAAC_rule(COREscore_not_NaN)"
                y_pred = pd.Series(plaac_uid2bin, dtype=float).reindex(eval_uids).fillna(0).astype(int).to_numpy()

            else:
                if model not in df_scores.columns:
                    continue

                if model in FIXED_THRESHOLD_BY_MODEL:
                    thr_used = float(FIXED_THRESHOLD_BY_MODEL[model])
                    pred_source = "fixed_threshold"
                else:
                    row = df_thr_rep.loc[df_thr_rep["Model"] == model, "Thr_max"]
                    if row.empty:
                        continue
                    thr_used = float(row.iloc[0])
                    pred_source = "Thr_max_from_df_thr"

                df_eval_score = df_scores[df_scores["uniprot"].isin(eval_uids)][["uniprot", model]].drop_duplicates(
                    "uniprot"
                )
                df_eval_score = df_eval_score.set_index("uniprot").reindex(eval_uids)
                y_score_for_bin = df_eval_score[model].to_numpy()
                mask_bin = ~pd.isna(y_score_for_bin) & ~pd.isna(y_true)
                if mask_bin.sum() == 0:
                    continue
                y_pred = (y_score_for_bin[mask_bin] >= thr_used).astype(int)
                y_true_bin = y_true[mask_bin]

                cm = _confusion_metrics(y_true_bin, y_pred)
                try:
                    lr_plus = calc_positive_likelihood_ratio(y_true_bin, y_pred=y_pred)
                except ValueError:
                    lr_plus = float("nan")

                results.append(
                    {
                        "Repeat": rep,
                        "Model": model,
                        "PredSource": pred_source,
                        "Thr_used": thr_used,
                        "n_samples": int(mask_bin.sum()),
                        "n_pos": int((y_true_bin == 1).sum()),
                        "n_neg": int((y_true_bin == 0).sum()),
                        "AUROC": auroc,
                        "AUPRC": auprc,
                        "AUPRC ratio": auprc_ratio,
                        "LR+": lr_plus,
                        "MCC": cm["MCC"],
                        "F1": cm["F1"],
                        "Precision": cm["Precision"],
                        "Recall": cm["Recall"],
                        "Specificity": cm["Specificity"],
                        "Accuracy": cm["Accuracy"],
                        "TP": cm["TP"],
                        "FP": cm["FP"],
                        "TN": cm["TN"],
                        "FN": cm["FN"],
                    }
                )
                continue

            y_true_bin = y_true
            cm = _confusion_metrics(y_true_bin, y_pred)
            try:
                lr_plus = calc_positive_likelihood_ratio(y_true_bin, y_pred=y_pred)
            except ValueError:
                lr_plus = float("nan")

            results.append(
                {
                    "Repeat": rep,
                    "Model": model,
                    "PredSource": pred_source,
                    "Thr_used": thr_used,
                    "n_samples": int(len(y_true_bin)),
                    "n_pos": int((y_true_bin == 1).sum()),
                    "n_neg": int((y_true_bin == 0).sum()),
                    "AUROC": auroc,
                    "AUPRC": auprc,
                    "AUPRC ratio": auprc_ratio,
                    "LR+": lr_plus,
                    "MCC": cm["MCC"],
                    "F1": cm["F1"],
                    "Precision": cm["Precision"],
                    "Recall": cm["Recall"],
                    "Specificity": cm["Specificity"],
                    "Accuracy": cm["Accuracy"],
                    "TP": cm["TP"],
                    "FP": cm["FP"],
                    "TN": cm["TN"],
                    "FN": cm["FN"],
                }
            )

    df_perf = pd.DataFrame(results)
    col_order = [
        "Repeat",
        "Model",
        "PredSource",
        "Thr_used",
        "n_samples",
        "n_pos",
        "n_neg",
        "AUROC",
        "AUPRC",
        "MCC",
        "F1",
        "AUPRC ratio",
        "LR+",
        "Precision",
        "Recall",
        "Specificity",
        "Accuracy",
        "TP",
        "FP",
        "TN",
        "FN",
    ]
    col_order = [c for c in col_order if c in df_perf.columns]
    return df_perf[col_order]


def summary_like_example(df_perf: pd.DataFrame) -> pd.DataFrame:
    if df_perf.empty:
        return pd.DataFrame()

    metric_cols = [
        "AUROC",
        "AUPRC",
        "MCC",
        "F1",
        "AUPRC ratio",
        "LR+",
        "Precision",
        "Recall",
        "Specificity",
        "Accuracy",
    ]
    present = [c for c in metric_cols if c in df_perf.columns]
    df_summary = df_perf.groupby("Model")[present].agg(["mean", "std"]).reset_index()
    df_summary.columns = [f"{a}_{b}" if b else a for (a, b) in df_summary.columns.to_flat_index()]
    summary_order: List[str] = ["Model"]
    for m in present:
        summary_order.extend((f"{m}_mean", f"{m}_std"))
    return df_summary[summary_order]


def preprocess_scores_for_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Drop ``IDP_type``; ensure column name ``Organism`` (rename from ``organism`` if needed)."""
    out = df.copy()
    if "IDP_type" in out.columns:
        out = out.drop(columns=["IDP_type"])
    org_col = "Organism" if "Organism" in out.columns else "organism"
    if org_col == "organism":
        out = out.rename(columns={"organism": "Organism"})
    return out


def prepare_compute_frame(data1_raw: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    if "organism" not in data1_raw.columns and "Organism" in data1_raw.columns:
        data1_for_compute = data1_raw.rename(columns={"Organism": "organism"})
    else:
        data1_for_compute = data1_raw.copy()
    return data1_for_compute, "organism"


def run_subsample_and_summary(
    data1_raw: pd.DataFrame,
    organisms_eval: List[str],
    *,
    n_repeats: int,
    sub_ratio: float,
    random_seed: int,
    neg_csv_path: str,
    flfb_path: Optional[str],
    plaac_path: Optional[str],
) -> pd.DataFrame:
    """
    One subsample pass (all organisms in the table), then per-organism MCC + metrics summaries.
    """
    data1_for_compute, org_key = prepare_compute_frame(data1_raw)
    all_orgs = sorted(data1_for_compute[org_key].astype(str).unique())
    model_cols = infer_model_columns(data1_for_compute)
    if not model_cols:
        raise ValueError("No numeric model columns found in input CSV.")

    for org in organisms_eval:
        if org not in set(all_orgs):
            raise ValueError(f"Organism {org!r} not found in scores. Available: {all_orgs}")

    compute_model_performance_by_organism(
        data1_for_compute,
        organism_col=org_key,
        organisms=all_orgs,
        models=model_cols,
        n_repeats=n_repeats,
        random_seed=random_seed,
        sub_ratio=sub_ratio,
        output_subsampled_neg_file=neg_csv_path,
    )

    neg_sub = pd.read_csv(neg_csv_path)

    scores_base = (
        data1_raw.rename(columns={"organism": "Organism"}) if "organism" in data1_raw.columns else data1_raw.copy()
    )
    scores_metrics = preprocess_scores_for_metrics(scores_base)

    tag_mapping = {
        "PSPs": 1,
        "PSP": 1,
        "positive": 1,
        "pos": 1,
        "non-PSPs": 0,
        "Non-PSPs": 0,
        "negative": 0,
        "neg": 0,
        1: 1,
        0: 0,
    }

    flfb_map, plaac_map = load_optional_binary_maps(flfb_path, plaac_path)

    summaries: List[pd.DataFrame] = []
    for organism in organisms_eval:
        neg_sub_org = neg_sub[neg_sub["Organism"] == organism].copy()
        scores_org = scores_metrics[scores_metrics["Organism"] == organism].copy()
        df_thr = mcc_wide_for_repeats(scores_org, neg_sub_org, tag_mapping)
        df_perf = calculate_metrics(df_thr, neg_sub_org, scores_org, flfb_map, plaac_map)
        summaries.append(summary_like_example(df_perf))

    if len(summaries) == 1:
        return summaries[0]
    blocks: List[pd.DataFrame] = []
    for organism, s in zip(organisms_eval, summaries):
        t = s.copy()
        t.insert(0, "Organism_eval", organism)
        blocks.append(t)
    return pd.concat(blocks, ignore_index=True)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Predictor performance evaluation")
    parser.add_argument(
        "--scores",
        default="predictor_scores_excludeTraining.csv",
        help="Wide scores table (default: predictor_scores_excludeTraining.csv)",
    )
    parser.add_argument(
        "--output",
        default="example_out.csv",
        help="Output CSV path (default: example_out.csv)",
    )
    parser.add_argument(
        "--organism",
        action="append",
        dest="organisms",
        help="Organism to evaluate (repeat for multiple). Default: Plants",
    )
    parser.add_argument("--n-repeats", type=int, default=10)
    parser.add_argument("--sub-ratio", type=float, default=1.0)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--neg-csv", default=None, help="Write/read negative subsample UID CSV path")
    parser.add_argument("--work-dir", default=".", help="Directory for default neg CSV if --neg-csv omitted")
    parser.add_argument(
        "--flfb-csv",
        default=None,
        help="Optional FLFB CSV (columns: uniprot, Class; positive if Class == LLPS)",
    )
    parser.add_argument(
        "--plaac-csv",
        default=None,
        help="Optional PLAAC CSV (columns: uniprot, COREscore; positive if COREscore not NaN)",
    )
    args = parser.parse_args(argv)

    if not os.path.isfile(args.scores):
        print(f"Input not found: {args.scores}", file=sys.stderr)
        return 1

    data1_raw = pd.read_csv(args.scores)
    organisms_eval = args.organisms or ["Plants"]

    neg_path = args.neg_csv
    if neg_path is None:
        neg_path = os.path.join(args.work_dir or ".", "neg_subsample_uids.csv")

    out = run_subsample_and_summary(
        data1_raw,
        organisms_eval,
        n_repeats=args.n_repeats,
        sub_ratio=args.sub_ratio,
        random_seed=args.random_seed,
        neg_csv_path=neg_path,
        flfb_path=args.flfb_csv,
        plaac_path=args.plaac_csv,
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote {args.output} ({len(out)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
