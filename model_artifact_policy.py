"""Allow input preprocessing while rejecting prohibited library wrappers."""

from __future__ import annotations

import ast
from collections.abc import Mapping
from pathlib import Path
from types import FunctionType, ModuleType
from typing import Any

import numpy as np
import pandas as pd


# Predictive estimators and sklearn.preprocessing tools for input features
# are permitted. These modules still contain prohibited pipeline, composition,
# imputation, feature-selection, and output-processing helpers.
FORBIDDEN_MODULE_PREFIXES = (
    "sklearn.pipeline",
    "sklearn.compose",
    "sklearn.impute",
    "sklearn.feature_selection",
    "sklearn.feature_extraction",
    "sklearn.decomposition",
    "sklearn.kernel_approximation",
    "sklearn.random_projection",
    "sklearn.calibration",
)


def _forbidden_module(name: str) -> bool:
    return any(
        name == prefix or name.startswith(prefix + ".")
        for prefix in FORBIDDEN_MODULE_PREFIXES
    )


def _dotted_name(node: ast.AST) -> str | None:
    """Return a dotted source name such as ``sklearn.pipeline.Pipeline``."""
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    parts.append(current.id)
    return ".".join(reversed(parts))


def validate_explicit_transformation_sources(paths: list[Path]) -> None:
    """Reject imports or references to prohibited library helpers.

    Predictive estimators and ``sklearn.preprocessing`` input tools remain
    permitted. Lag construction and output post-processing must be visible in
    the submitted source. Source checks do not prove correct data timing.
    """
    for path in paths:
        source = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            raise ValueError(f"cannot parse submitted Python file {path}: {exc}") from exc

        aliases: dict[str, str] = {}
        for node in ast.walk(tree):
            imported_names: list[str] = []
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_names.append(alias.name)
                    bound_name = alias.asname or alias.name.split(".")[0]
                    aliases[bound_name] = alias.name if alias.asname else bound_name
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                imported_names.append(module)
                for alias in node.names:
                    full_name = f"{module}.{alias.name}" if module else alias.name
                    imported_names.append(full_name)
                    aliases[alias.asname or alias.name] = full_name

            forbidden = next((name for name in imported_names if _forbidden_module(name)), None)
            if forbidden is not None:
                raise ValueError(
                    f"{path} imports prohibited library code '{forbidden}' "
                    f"on line {getattr(node, 'lineno', '?')}. Use only the "
                    "permitted preprocessing and predictive-estimator APIs."
                )

        # Also catch access through an alias, for example
        # ``import sklearn as sk; sk.pipeline.Pipeline()``.
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute):
                continue
            dotted = _dotted_name(node)
            if dotted is None:
                continue
            root, separator, remainder = dotted.partition(".")
            resolved = aliases.get(root, root)
            if separator:
                resolved = f"{resolved}.{remainder}"
            if _forbidden_module(resolved):
                raise ValueError(
                    f"{path} refers to prohibited library code '{resolved}' "
                    f"on line {getattr(node, 'lineno', '?')}. Use only the "
                    "permitted preprocessing and predictive-estimator APIs."
                )


def _is_forbidden_library_object(value: Any) -> bool:
    module = type(value).__module__
    return _forbidden_module(module)


def validate_explicit_transformations(model: Any) -> None:
    """Reject serialized objects from prohibited library modules.

    Plain dictionaries, sequences, numbers, strings, NumPy arrays, Pandas
    objects, predictive estimators, and fitted ``sklearn.preprocessing`` objects
    are allowed. The object graph is walked recursively so a prohibited
    object cannot be hidden inside a wrapper, dictionary, or sequence.
    """
    stack: list[tuple[str, Any]] = [("model", model)]
    seen: set[int] = set()

    while stack:
        location, value = stack.pop()
        identity = id(value)
        if identity in seen:
            continue
        seen.add(identity)

        if _is_forbidden_library_object(value):
            object_type = f"{type(value).__module__}.{type(value).__qualname__}"
            raise ValueError(
                "model.pkl contains a prohibited library object "
                f"at {location}: {object_type}. Fitted sklearn.preprocessing "
                "input objects are allowed, but pipelines and the other "
                "prohibited wrappers are not."
            )

        if isinstance(value, Mapping):
            for key, item in value.items():
                stack.append((f"{location}[{key!r}]", item))
            continue

        if isinstance(value, (list, tuple, set, frozenset)):
            for index, item in enumerate(value):
                stack.append((f"{location}[{index}]", item))
            continue

        if isinstance(value, np.ndarray):
            if value.dtype == object:
                for index, item in enumerate(value.flat):
                    stack.append((f"{location}.flat[{index}]", item))
            continue

        # DataFrames and Series are allowed as stored data. Their internal
        # manager objects are implementation details and are not traversed.
        if isinstance(value, (pd.DataFrame, pd.Series, pd.Index)):
            continue

        if isinstance(value, (str, bytes, bytearray, int, float, complex,
                              bool, type(None), np.generic, FunctionType,
                              ModuleType, type)):
            continue

        attributes = getattr(value, "__dict__", None)
        if isinstance(attributes, dict):
            for name, item in attributes.items():
                stack.append((f"{location}.{name}", item))
