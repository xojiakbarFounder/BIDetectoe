"""YOLOv8 fine-tuning entry point for the surveillance platform."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger
from ultralytics import YOLO

from config import settings


def _version_name(prefix: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{prefix}-{timestamp}"


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _metrics_to_dict(metrics: Any) -> dict[str, Any]:
    results = getattr(metrics, "results_dict", {}) or {}
    box = getattr(metrics, "box", None)
    summary = {key: _jsonable(value) for key, value in dict(results).items()}

    if box is not None:
        summary.update(
            {
                "precision": float(getattr(box, "mp", 0.0)),
                "recall": float(getattr(box, "mr", 0.0)),
                "map50": float(getattr(box, "map50", 0.0)),
                "map50_95": float(getattr(box, "map", 0.0)),
            }
        )
    return summary


def _jsonable(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


def _read_last_training_row(results_csv: Path) -> dict[str, Any]:
    if not results_csv.exists():
        return {}

    with results_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}

    cleaned: dict[str, Any] = {}
    for key, value in rows[-1].items():
        clean_key = key.strip()
        try:
            cleaned[clean_key] = float(value)
        except (TypeError, ValueError):
            cleaned[clean_key] = value
    return cleaned


def _update_registry(registry_path: Path, record: dict[str, Any]) -> None:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    if registry_path.exists():
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    else:
        registry = {"models": []}

    registry["models"].append(record)
    registry["latest"] = record["version"]
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")


def run_training(args: argparse.Namespace) -> Path:
    data_yaml = Path(args.data)
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset config not found: {data_yaml}")

    version = args.version or _version_name(args.name)
    experiments_dir = Path(args.project)
    run_dir = experiments_dir / version
    model_dir = Path(args.models_dir) / version
    visual_dir = model_dir / "visualizations"

    logger.info("Starting YOLOv8 fine-tuning from '{}'.", args.weights)
    logger.info("Dataset: {}", data_yaml)
    logger.info("Device: {}", args.device)

    model = YOLO(args.weights)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=str(experiments_dir),
        name=version,
        exist_ok=False,
        pretrained=True,
        plots=True,
        save=True,
    )

    best_src = run_dir / "weights" / "best.pt"
    last_src = run_dir / "weights" / "last.pt"
    best_dst = model_dir / "best.pt"
    last_dst = model_dir / "last.pt"
    _copy_if_exists(best_src, best_dst)
    _copy_if_exists(last_src, last_dst)
    if not best_dst.exists() and not last_dst.exists():
        raise FileNotFoundError(f"No trained weights were found in {run_dir / 'weights'}")

    trained_model = YOLO(str(best_dst if best_dst.exists() else last_dst))
    metrics = trained_model.val(
        data=str(data_yaml),
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(experiments_dir),
        name=f"{version}-val",
        plots=True,
    )
    metrics_dict = _metrics_to_dict(metrics)

    for filename in (
        "results.csv",
        "results.png",
        "confusion_matrix.png",
        "P_curve.png",
        "R_curve.png",
        "PR_curve.png",
        "F1_curve.png",
    ):
        _copy_if_exists(run_dir / filename, visual_dir / filename)

    summary = {
        "version": version,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": str(data_yaml),
        "pretrained_weights": args.weights,
        "best_model": str(best_dst),
        "last_model": str(last_dst),
        "experiment_dir": str(run_dir),
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": args.device,
        "metrics": metrics_dict,
        "last_epoch_results": _read_last_training_row(run_dir / "results.csv"),
    }
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    _update_registry(Path(args.models_dir) / "registry.json", summary)

    latest_dir = Path(args.models_dir) / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    _copy_if_exists(best_dst, latest_dir / "best.pt")
    _copy_if_exists(last_dst, latest_dir / "last.pt")

    logger.info("Training complete. Best model: {}", best_dst)
    logger.info("Validation metrics: {}", metrics_dict)
    return best_dst


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 for surveillance.")
    parser.add_argument("--data", default=settings.dataset_yaml, help="YOLO dataset.yaml path.")
    parser.add_argument(
        "--weights",
        default=settings.yolo_pretrained_model,
        help="Pretrained YOLOv8 weights to start from.",
    )
    parser.add_argument("--epochs", type=int, default=settings.train_epochs)
    parser.add_argument("--batch", type=int, default=settings.train_batch)
    parser.add_argument("--imgsz", type=int, default=settings.train_imgsz)
    parser.add_argument("--device", default=settings.resolved_yolo_device)
    parser.add_argument("--project", default=settings.training_project)
    parser.add_argument("--name", default=settings.training_name)
    parser.add_argument("--version", default="", help="Optional model version name.")
    parser.add_argument("--models-dir", default=settings.trained_models_dir)
    return parser.parse_args()


def main() -> None:
    run_training(parse_args())


if __name__ == "__main__":
    main()
