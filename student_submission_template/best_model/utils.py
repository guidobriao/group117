"""Auxiliary prediction functions imported by my_model.py."""

from __future__ import annotations

from typing import Any


def constant_prediction(model: Any) -> float:
    """Extract the dummy model's constant prediction."""
    return float(model["constant_prediction"])
