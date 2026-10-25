#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
End-to-End Metrics Pipeline
---------------------------
Runs the locked final gait-test detection pipeline on evaluation datasets
(Between/Within) and computes event-level plus battery-level metrics.
"""

import argparse
import json
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from pipeline.gait_detection import detect_gait_test
from pipeline.postprocessing import run_postprocessing
from pipeline.end2end_metrics.utils import (
    aggregate_and_report,
    evaluate_event_detection,
    initialize_metrics_dict,
    load_json,
    process_targets,
    save_json,
)
from pipeline.end2end_metrics.battery_metrics import (
    aggregate_battery_collections,
    evaluate_battery_collection,
)


LOCKED_POSTPROCESSING_CONFIG = {
    "decoder_mode": "competition_aware",
    "smooth_probabilities": True,
    "smooth_sec": 1.5,
    "class_on_thresholds": {1: 0.40, 2: 0.40, 3: 0.40, 4: 0.35},
    "class_off_thresholds": {1: 0.28, 2: 0.28, 3: 0.28, 4: 0.25},
    "gap_fill_sec": {1: 5.0, 2: 5.0, 3: 5.0, 4: 15.0},
    "min_duration_sec": {1: 15.0, 2: 12.0, 3: 20.0, 4: 70.0},
    "island_filter": {
        "enabled": True,
        "neighborhood_sec": 300.0,
        "min_neighbor_count": 1,
        "target_classes": [1, 2, 3, 4],
        "require_different_class_neighbor": False,
    },
}
NON_EVENT_CACHE_STATUSES = {
    "no_valid_windows",
    "no_stage1_positive_windows",
    "no_windows_after_continuity",
    "no_events_after_postprocessing",
    "unknown_failure",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run end-to-end evaluation with the locked final pipeline.")
    parser.add_argument(
        "--dataset-source",
        type=str,
        default="test",
        choices=["test", "val"],
        help="Which dataset-info source to evaluate: test_dataset_info or val_dataset_info.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of session workers to run in parallel. Use 1 for sequential mode.",
    )
    parser.add_argument(
        "--results-dir-name",
        type=str,
        default="",
        help="Optional output directory name under pipeline/end2end_metrics/.",
    )
    parser.add_argument(
        "--postprocessing-cache-dir",
        type=str,
        default="",
        help=(
            "Optional directory used to cache Stage 1/2 outputs before post-processing. "
            "Repeated runs can then reuse the cache."
        ),
    )
    parser.add_argument(
        "--preprocessing-config-json",
        type=str,
        default="",
        help=(
            "Optional JSON file describing preprocessing overrides. "
            "Accepts either a raw config dict or a candidate_config.json from the Ullrich preprocessing grid search."
        ),
    )
    return parser.parse_args()


def resolve_dataset_info_paths(root: Path, dataset_source: str) -> dict:
    info_dir = root / "pipeline" / "end2end_metrics" / f"{dataset_source}_dataset_info"
    return {
        "info_dir": info_dir,
        "between_paths": info_dir / f"{dataset_source}_between_paths.json",
        "within_paths": info_dir / f"{dataset_source}_within_paths.json",
    }


def extract_patient_ids(paths_dict: dict) -> list:
    return sorted({session_id.split("_")[0] for session_id in paths_dict.keys()})


def build_postprocessing_cache_path(
    cache_root: Path,
    dataset_source: str,
    split_name: str,
    session_id: str,
) -> Path:
    return cache_root / dataset_source / split_name / f"{session_id}.npz"


def save_postprocessing_cache(
    cache_path: Path,
    *,
    status: str,
    window_probs: Optional[np.ndarray],
    window_start_times: Optional[np.ndarray],
    signal_path: str,
    preprocessing_signature: str = "",
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": status,
        "signal_path": signal_path,
        "schema_version": 2,
        "preprocessing_signature": preprocessing_signature,
    }
    np.savez_compressed(
        cache_path,
        window_probs=np.asarray(
            window_probs if window_probs is not None else np.zeros((0, 0, 0), dtype=np.float32),
            dtype=np.float32,
        ),
        window_start_times=np.asarray(
            window_start_times if window_start_times is not None else np.zeros((0,), dtype=np.int64),
            dtype=np.int64,
        ),
        meta_json=np.array(json.dumps(payload)),
    )


def load_postprocessing_cache(cache_path: Path) -> dict:
    with np.load(cache_path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta_json"]))
        return {
            "status": meta.get("status", "unknown"),
            "signal_path": meta.get("signal_path", ""),
            "preprocessing_signature": meta.get("preprocessing_signature", ""),
            "window_probs": data["window_probs"],
            "window_start_times": data["window_start_times"],
        }


def resolve_preprocessing_config(preprocessing_config_json: str) -> dict:
    if not preprocessing_config_json:
        return {}

    raw = load_json(Path(preprocessing_config_json))
    if isinstance(raw, dict) and "candidate" in raw:
        raw = raw["candidate"]

    if not isinstance(raw, dict):
        raise ValueError("Preprocessing config JSON must contain a dict-like payload.")

    preprocessing_config = dict(raw.get("preprocessing_config", {}))
    if "ullrich_config" in raw:
        preprocessing_config["ullrich_config"] = raw["ullrich_config"]
    elif not preprocessing_config and any(
        key in raw
        for key in (
            "sensor_scope",
            "sensor_channel_config",
            "peak_prominence",
            "window_size_s",
            "active_signal_threshold",
            "locomotion_band",
            "harmonic_tolerance_hz",
            "merge_gait_sequences_from_sensors",
            "additional_margin_s",
        )
    ):
        preprocessing_config["ullrich_config"] = raw

    return preprocessing_config


def build_preprocessing_signature(preprocessing_config: Optional[dict]) -> str:
    return json.dumps(preprocessing_config or {}, sort_keys=True, separators=(",", ":"))


def sanitize_results_dir_component(text: str) -> str:
    allowed = []
    for char in text.strip().lower():
        if char.isalnum() or char in {"-", "_"}:
            allowed.append(char)
        else:
            allowed.append("_")
    sanitized = "".join(allowed).strip("_")
    while "__" in sanitized:
        sanitized = sanitized.replace("__", "_")
    return sanitized


def resolve_results_dir(base_parent: Path, args: argparse.Namespace) -> Path:
    if args.results_dir_name:
        dir_name = sanitize_results_dir_component(args.results_dir_name)
        if not dir_name:
            raise ValueError("The provided --results-dir-name becomes empty after sanitization.")
        return base_parent / dir_name
    return base_parent / "results"


def build_battery_ground_truth(targets_processed: pd.DataFrame) -> list:
    if targets_processed is None or targets_processed.empty:
        return []

    gt_battery = targets_processed[["class_id", "time_stamp_start", "time_stamp_end"]].copy()
    gt_battery = gt_battery.sort_values("time_stamp_start").reset_index(drop=True)
    gt_battery = gt_battery.rename(columns={"time_stamp_start": "start_ns", "time_stamp_end": "end_ns"})
    return gt_battery.to_dict("records")


def process_single_session(
    patid_date: str,
    paths: dict,
    dataset_source: str = "test",
    split_name: str = "unknown",
    use_author_weights_stage1: bool = True,
    use_author_weights_stage2: bool = True,
    postprocessing_cache_dir: str = "",
    preprocessing_config: Optional[dict] = None,
    postprocessing_config: Optional[dict] = None,
    return_full_session_data: bool = False,
) -> dict:
    patid = patid_date.split("_")[0]
    csv_path = paths["csv_files"][0]
    h5_path = paths["h5_files"][0]

    try:
        effective_postprocessing_config = postprocessing_config or LOCKED_POSTPROCESSING_CONFIG
        targets = pd.read_csv(csv_path)
        targets_processed = process_targets(targets)
        battery_ground_truth = build_battery_ground_truth(targets_processed)
        preprocessing_signature = build_preprocessing_signature(preprocessing_config)

        predictions = []
        cache_root = Path(postprocessing_cache_dir) if postprocessing_cache_dir else None
        cache_path = None if cache_root is None else build_postprocessing_cache_path(
            cache_root=cache_root,
            dataset_source=dataset_source,
            split_name=split_name,
            session_id=patid_date,
        )

        loaded_cache = None
        if cache_path is not None and cache_path.exists():
            loaded_cache = load_postprocessing_cache(cache_path)
            if loaded_cache.get("preprocessing_signature", "") != preprocessing_signature:
                loaded_cache = None

        if loaded_cache is not None:
            if (
                loaded_cache["status"] in NON_EVENT_CACHE_STATUSES
                or loaded_cache["window_probs"].size == 0
            ):
                predictions = []
            else:
                predictions = run_postprocessing(
                    window_probs=loaded_cache["window_probs"],
                    window_start_times=loaded_cache["window_start_times"],
                    postprocessing_config=effective_postprocessing_config,
                )
        else:
            detection_output = detect_gait_test(
                signal_path=h5_path,
                plot_signal=False,
                use_author_weights_stage1=use_author_weights_stage1,
                use_author_weights_stage2=use_author_weights_stage2,
                preprocessing_config=preprocessing_config,
                postprocessing_config=effective_postprocessing_config,
                return_debug=bool(cache_path is not None),
            )

            if cache_path is None:
                predictions = detection_output or []
            else:
                if detection_output is None:
                    predictions = []
                    save_postprocessing_cache(
                        cache_path,
                        status="unknown_failure",
                        window_probs=None,
                        window_start_times=None,
                        signal_path=str(h5_path),
                        preprocessing_signature=preprocessing_signature,
                    )
                else:
                    predictions = detection_output["events"]
                    debug = detection_output["debug"]
                    prepost_inputs = debug.get("postprocessing_inputs", {})
                    save_postprocessing_cache(
                        cache_path,
                        status=debug.get("status", "unknown"),
                        window_probs=prepost_inputs.get("window_probs"),
                        window_start_times=prepost_inputs.get("window_start_times"),
                        signal_path=str(h5_path),
                        preprocessing_signature=preprocessing_signature,
                    )

        event_metrics = evaluate_event_detection(predictions, targets_processed)
        battery_metrics = evaluate_battery_collection(
            predicted_events=predictions,
            ground_truth_subtests=battery_ground_truth,
        )
    except Exception as exc:
        return {
            "patid_date": patid_date,
            "patid": patid,
            "error": str(exc),
        }

    result = {
        "patid_date": patid_date,
        "patid": patid,
        "event_metrics": event_metrics,
        "battery_metrics": battery_metrics,
        "error": None,
    }
    if return_full_session_data:
        result["predictions"] = predictions
        result["ground_truth_subtests"] = battery_ground_truth
    return result


def merge_session_result(
    session_result: dict,
    metrics_storage: dict,
    battery_storage: dict,
) -> None:
    patid = session_result["patid"]
    patid_date = session_result["patid_date"]

    if session_result["error"] is not None:
        print(f"[Error] Skipping {patid_date}: {session_result['error']}")
        return

    battery_storage[patid_date] = session_result["battery_metrics"]
    metrics = session_result["event_metrics"]

    if patid in metrics_storage:
        for class_id in range(1, 5):
            if class_id in metrics:
                metrics_storage[patid][class_id]["TP"] += metrics[class_id]["TP"]
                metrics_storage[patid][class_id]["FP"] += metrics[class_id]["FP"]
                metrics_storage[patid][class_id]["FN"] += metrics[class_id]["FN"]
    else:
        print(f"[Warning] Patient {patid} found in paths but not in initialized metrics keys.")


def process_dataset(
    paths_dict: dict,
    metrics_storage: dict,
    battery_storage: dict,
    description: str,
    dataset_source: str = "test",
    split_name: str = "unknown",
    use_author_weights_stage1: bool = True,
    use_author_weights_stage2: bool = True,
    num_workers: int = 1,
    postprocessing_cache_dir: str = "",
    preprocessing_config: Optional[dict] = None,
) -> dict:
    items = list(paths_dict.items())
    if num_workers <= 1:
        for patid_date, paths in tqdm(items, desc=description):
            session_result = process_single_session(
                patid_date=patid_date,
                paths=paths,
                dataset_source=dataset_source,
                split_name=split_name,
                use_author_weights_stage1=use_author_weights_stage1,
                use_author_weights_stage2=use_author_weights_stage2,
                postprocessing_cache_dir=postprocessing_cache_dir,
                preprocessing_config=preprocessing_config,
            )
            merge_session_result(session_result, metrics_storage, battery_storage)
        return metrics_storage

    max_workers = min(num_workers, max(len(items), 1))
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        futures = [
            executor.submit(
                process_single_session,
                patid_date,
                paths,
                dataset_source,
                split_name,
                use_author_weights_stage1,
                use_author_weights_stage2,
                postprocessing_cache_dir,
                preprocessing_config,
            )
            for patid_date, paths in items
        ]

        for future in tqdm(as_completed(futures), total=len(futures), desc=description):
            try:
                session_result = future.result()
            except Exception as exc:
                session_result = {
                    "patid_date": "unknown_session",
                    "patid": "unknown_patient",
                    "error": f"Worker crashed: {exc}",
                }
            merge_session_result(session_result, metrics_storage, battery_storage)

    return metrics_storage


if __name__ == "__main__":
    args = parse_args()

    dataset_info = resolve_dataset_info_paths(ROOT, args.dataset_source)
    info_dir = dataset_info["info_dir"]
    if not info_dir.exists():
        raise FileNotFoundError(f"Dataset info directory not found: {info_dir}")

    results_parent = ROOT / "pipeline" / "end2end_metrics"
    results_dir = resolve_results_dir(results_parent, args)
    results_dir.mkdir(parents=True, exist_ok=True)

    use_author_w1 = True
    use_author_w2 = True

    print("Loading dataset configurations...")
    if args.num_workers > 1:
        print(
            f"[Info] Parallel mode enabled with {args.num_workers} workers. "
            "If all workers share one GPU, memory contention may reduce speed."
        )
    print("[Info] Using locked final post-processing configuration.")
    print(f"[Info] Using dataset source: {args.dataset_source}")
    print(f"[Info] Writing outputs to: {results_dir}")
    if args.postprocessing_cache_dir:
        print(f"[Info] Using post-processing cache dir: {args.postprocessing_cache_dir}")
    preprocessing_config = resolve_preprocessing_config(args.preprocessing_config_json)
    if preprocessing_config:
        print(f"[Info] Preprocessing override loaded from: {args.preprocessing_config_json}")
        print(f"[Info] Preprocessing config: {preprocessing_config}")

    pat_between_paths = load_json(dataset_info["between_paths"])
    pat_within_paths = load_json(dataset_info["within_paths"])

    pat_between_list = extract_patient_ids(pat_between_paths)
    pat_within_list = extract_patient_ids(pat_within_paths)

    pat_between_metrics = initialize_metrics_dict(pat_between_list)
    pat_within_metrics = initialize_metrics_dict(pat_within_list)
    between_battery_metrics = {}
    within_battery_metrics = {}

    process_dataset(
        paths_dict=pat_between_paths,
        metrics_storage=pat_between_metrics,
        battery_storage=between_battery_metrics,
        description="Between-Subject",
        dataset_source=args.dataset_source,
        split_name="between",
        use_author_weights_stage1=use_author_w1,
        use_author_weights_stage2=use_author_w2,
        num_workers=args.num_workers,
        postprocessing_cache_dir=args.postprocessing_cache_dir,
        preprocessing_config=preprocessing_config,
    )

    process_dataset(
        paths_dict=pat_within_paths,
        metrics_storage=pat_within_metrics,
        battery_storage=within_battery_metrics,
        description="Within-Subject",
        dataset_source=args.dataset_source,
        split_name="within",
        use_author_weights_stage1=use_author_w1,
        use_author_weights_stage2=use_author_w2,
        num_workers=args.num_workers,
        postprocessing_cache_dir=args.postprocessing_cache_dir,
        preprocessing_config=preprocessing_config,
    )

    print("\nAggregating Results...")
    between_results = aggregate_and_report(metrics_data=pat_between_metrics, prefix="Between-Subject")
    within_results = aggregate_and_report(metrics_data=pat_within_metrics, prefix="Within-Subject")
    between_battery_results = aggregate_battery_collections(between_battery_metrics)
    within_battery_results = aggregate_battery_collections(within_battery_metrics)

    print(f"Between-Subject Battery Summary: {between_battery_results}")
    print(f"Within-Subject Battery Summary: {within_battery_results}")

    print(f"\nSaving results to: {results_dir}")
    save_json(pat_between_metrics, results_dir / "between_metrics.json")
    save_json(pat_within_metrics, results_dir / "within_metrics.json")
    save_json(between_results, results_dir / "between_results.json")
    save_json(within_results, results_dir / "within_results.json")
    save_json(between_battery_metrics, results_dir / "between_battery_metrics.json")
    save_json(within_battery_metrics, results_dir / "within_battery_metrics.json")
    save_json(between_battery_results, results_dir / "between_battery_results.json")
    save_json(within_battery_results, results_dir / "within_battery_results.json")
    print("Done.")
