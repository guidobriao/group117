"""Simulate submission checks using a few days from the released dataset."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from build_lagged_features import DATE, TARGET, read_table
from model_artifact_policy import (
    validate_explicit_transformation_sources,
    validate_explicit_transformations,
)


def import_module(path: Path):
    module_dir = str(path.parent.resolve())
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    spec = importlib.util.spec_from_file_location("student_my_model", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not import my_model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def is_pdf(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 5:
        return False
    with path.open("rb") as handle:
        return b"%PDF-" in handle.read(1024)


def validate(submission_dir: Path, data_path: Path,
             require_declarations: bool = True) -> dict:
    best_model_dir = submission_dir / "best_model"
    if not best_model_dir.is_dir():
        raise ValueError("missing required directory: best_model")
    required = ["my_model.py", "model.pkl", "utils.py"]
    missing = [name for name in required if not (best_model_dir / name).is_file()]
    if missing:
        raise ValueError(f"missing required files in best_model: {missing}")
    training_dir = submission_dir / "training_code"
    if not training_dir.is_dir():
        raise ValueError("missing required directory: training_code")
    if not (training_dir / "train.py").is_file():
        raise ValueError("missing required file: training_code/train.py")
    training_sources = sorted(path for path in training_dir.rglob("*.py") if path.is_file())
    prediction_sources = sorted(path for path in best_model_dir.rglob("*.py") if path.is_file())
    validate_explicit_transformation_sources(prediction_sources + training_sources)
    ai_declarations = []
    if require_declarations:
        declarations_dir = submission_dir / "declarations"
        if not declarations_dir.is_dir():
            raise ValueError("missing required directory: declarations")
        honor = declarations_dir / "group_declaration_of_honor.pdf"
        if not is_pdf(honor):
            raise ValueError(
                "declarations/group_declaration_of_honor.pdf is missing or is not a valid PDF"
            )
        ai_declarations = [
            declarations_dir / "student_1_ai_declaration.pdf",
            declarations_dir / "student_2_ai_declaration.pdf",
        ]
        invalid = [path.name for path in ai_declarations if not is_pdf(path)]
        if invalid:
            raise ValueError(f"missing or invalid declaration PDF files: {invalid}")
    module = import_module(best_model_dir / "my_model.py")
    for name in ["load_model", "predict"]:
        if not callable(getattr(module, name, None)):
            raise ValueError(f"my_model.py must define callable {name}")
    model = module.load_model(best_model_dir / "model.pkl")
    validate_explicit_transformations(model)
    raw = read_table(data_path)
    checks = []
    for index in range(len(raw) - 3, len(raw)):
        history = raw.iloc[:index].copy()
        day = raw.iloc[[index]].drop(columns=[TARGET, "qc_flag"], errors="ignore").copy()
        history_before = history.copy(deep=True)
        day_before = day.copy(deep=True)
        pred = np.asarray(module.predict(model, history, day), dtype=float)
        if pred.shape != (1,) or not np.isfinite(pred).all() or pred[0] < 0:
            raise ValueError(
                "predict must return one finite, nonnegative value; "
                f"received shape {pred.shape}, value {pred}"
            )
        if not history.equals(history_before):
            raise ValueError("predict modified history_df in place")
        if not day.equals(day_before):
            raise ValueError("predict modified day_features_df in place")
        checks.append({"date": str(day.iloc[0][DATE].date()), "prediction": float(pred[0])})
    return {"status": "PASS", "group_id": submission_dir.name,
            "training_source_files": len(training_sources),
            "ai_declaration_files": len(ai_declarations),
            "checks": checks}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission_dir", type=Path)
    parser.add_argument("released_data", type=Path)
    parser.add_argument("--allow-missing-declarations", action="store_true",
                        help="instructor/reference testing only")
    args = parser.parse_args()
    print(json.dumps(validate(args.submission_dir, args.released_data,
                              require_declarations=not args.allow_missing_declarations), indent=2))


if __name__ == "__main__":
    main()
