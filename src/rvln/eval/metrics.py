"""
Evaluation metrics for UAV trajectory comparison.

Includes DTW distance, nDTW, and classification-based evaluation.
Vendored from UAV-Flow/UAV-Flow-Eval/metric.py (commit 0114801).
"""

import json
import os

import numpy as np
from scipy.spatial.distance import cdist


def _print_table(headers, rows, align=None):
    """Print a clean ASCII table."""
    headers = [str(h) for h in headers]
    str_rows = [["" if c is None else str(c) for c in r] for r in rows]
    widths = [len(h) for h in headers]
    for r in str_rows:
        for i, c in enumerate(r):
            if i >= len(widths):
                widths.append(len(c))
            else:
                widths[i] = max(widths[i], len(c))
    if align is None:
        align = ["l"] * len(widths)

    def fmt_cell(i, s):
        if align[i] == "r":
            return s.rjust(widths[i])
        return s.ljust(widths[i])

    sep = "+" + "+".join(["-" * (w + 2) for w in widths]) + "+"
    print(sep)
    print("| " + " | ".join(fmt_cell(i, headers[i]) for i in range(len(widths))) + " |")
    print(sep)
    for r in str_rows:
        print("| " + " | ".join(fmt_cell(i, r[i] if i < len(r) else "") for i in range(len(widths))) + " |")
    print(sep)


def get_gt_states_from_rule_log(gt_path):
    """Load preprocessed ground-truth states from a rule-based log JSON file."""
    with open(gt_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["reference_path_preprocessed"]


def get_sampled_state6d_from_model_rule(model_path, step=5, zero_pos=False):
    """Sample 6D states from a model trajectory log.

    Output vector per sample: [x, y, z, cos(roll), cos(yaw), cos(pitch)].
    """
    with open(model_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    result = []
    for idx, item in enumerate(data):
        if idx % step == 0:
            if zero_pos:
                pos = np.zeros(3)
            else:
                pos = np.array(item["state"][0]) / 100
            rot = np.array(item["state"][1])
            rot_cos = np.cos(np.deg2rad(rot))
            vec = np.concatenate([pos, rot_cos])
            result.append(vec)
    return result


def get_sampled_state6d_from_gt_rule(gt_path, step=5, max_points=20, zero_pos=False):
    """Sample 6D states from ground-truth states."""
    states = get_gt_states_from_rule_log(gt_path)
    result = []
    for idx, item in enumerate(states):
        if idx % step == 0:
            if zero_pos:
                pos = np.zeros(3)
            else:
                pos = np.array(item[:3]) / 100
            rot = np.array(item[3:6])
            rot_cos = np.cos(np.deg2rad(rot))
            vec = np.concatenate([pos, rot_cos])
            result.append(vec)
    return result[:max_points]


def dtw_distance(vecs1, vecs2):
    """Compute DTW (Dynamic Time Warping) distance between two sequences."""
    if len(vecs1) == 0 or len(vecs2) == 0:
        return None
    dist_matrix = cdist(vecs1, vecs2, metric="euclidean")
    n, m = dist_matrix.shape
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = dist_matrix[i - 1, j - 1]
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
    return dtw[n, m]


def path_length(points):
    """Compute the polyline length for a sequence of points."""
    if len(points) < 2:
        return 0
    length = 0
    for i in range(1, len(points)):
        length += np.linalg.norm(points[i] - points[i - 1])
    return length


def ndtw(dtw_dist, gt_len, eta=1):
    """Compute normalized DTW (nDTW) score: exp(-DTW / (eta * L_gt))."""
    if dtw_dist is None or gt_len == 0:
        return None
    return np.exp(-dtw_dist / (eta * gt_len))


def evaluate_by_classification(classified_json_path, model_dir, gt_rule_dir, default_step=5):
    """Evaluate trajectories grouped by classification and report nDTW statistics."""
    with open(classified_json_path, "r", encoding="utf-8") as f:
        class_dict = json.load(f)
    all_ndtw_results = []
    table_rows = []

    for class_name, file_list in class_dict.items():
        ndtw_results = []
        zero_pos = class_name in ["Turn", "Rotate"]
        if class_name in ["Turn", "Move"]:
            step = 2
        else:
            step = default_step
        num_valid = 0
        for file_name in file_list:
            gt_path = os.path.join(gt_rule_dir, file_name)
            model_path = os.path.join(model_dir, file_name)
            if not os.path.exists(gt_path) or not os.path.exists(model_path):
                continue
            model_vecs = get_sampled_state6d_from_model_rule(model_path, step, zero_pos=zero_pos)
            gt_vecs = get_sampled_state6d_from_gt_rule(gt_path, step, max_points=20, zero_pos=zero_pos)
            dtw_dist = dtw_distance(gt_vecs, model_vecs)
            gt_len = path_length(gt_vecs)
            ndtw_score = ndtw(dtw_dist, gt_len, eta=1)
            if ndtw_score is not None:
                ndtw_results.append(ndtw_score)
                all_ndtw_results.append(ndtw_score)
            num_valid += 1
        mean_ndtw = np.mean(ndtw_results) if len(ndtw_results) > 0 else None
        table_rows.append([
            class_name,
            str(len(file_list)),
            str(num_valid),
            f"{mean_ndtw:.4f}" if mean_ndtw is not None else "-",
        ])

    print("\nUAV-Flow Evaluation by Class (nDTW)")
    _print_table(
        headers=["Class", "#Tasks", "#Evaluated", "Mean nDTW"],
        rows=table_rows,
        align=["l", "r", "r", "r"],
    )

    print("\nOverall Summary (nDTW)")
    overall_rows = [[
        str(len(all_ndtw_results)),
        f"{np.mean(all_ndtw_results):.4f}" if len(all_ndtw_results) else "-",
    ]]
    _print_table(
        headers=["#nDTW Samples", "Overall Mean nDTW"],
        rows=overall_rows,
        align=["r", "r"],
    )
