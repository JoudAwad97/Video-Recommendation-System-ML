"""
I/O utilities for saving and loading data and artifacts.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Union

import pandas as pd
import numpy as np


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if it doesn't.

    Args:
        path: Directory path to ensure exists.

    Returns:
        Path object for the directory.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Any, filepath: Union[str, Path]) -> None:
    """Save data to JSON file.

    Args:
        data: Data to save (must be JSON serializable).
        filepath: Path to save the JSON file.
    """
    filepath = Path(filepath)
    ensure_dir(filepath.parent)

    # Convert numpy types to Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj

    serializable_data = convert_to_serializable(data)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(serializable_data, f, indent=2, ensure_ascii=False)


def load_json(filepath: Union[str, Path]) -> Any:
    """Load data from JSON file.

    Args:
        filepath: Path to the JSON file.

    Returns:
        Loaded data.
    """
    filepath = Path(filepath)
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_parquet(
    df: pd.DataFrame,
    filepath: Union[str, Path],
    compression: str = "snappy"
) -> None:
    """Save DataFrame to Parquet file.

    Args:
        df: DataFrame to save.
        filepath: Path to save the Parquet file.
        compression: Compression algorithm ('snappy', 'gzip', 'brotli', None).
    """
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    df.to_parquet(filepath, compression=compression, index=False)


def load_parquet(filepath: Union[str, Path]) -> pd.DataFrame:
    """Load DataFrame from Parquet file.

    Args:
        filepath: Path to the Parquet file.

    Returns:
        Loaded DataFrame.
    """
    filepath = Path(filepath)
    return pd.read_parquet(filepath)


def save_numpy(
    array: np.ndarray,
    filepath: Union[str, Path],
    compressed: bool = True
) -> None:
    """Save numpy array to file.

    Args:
        array: Numpy array to save.
        filepath: Path to save the array.
        compressed: Whether to use compression.
    """
    filepath = Path(filepath)
    ensure_dir(filepath.parent)

    if compressed:
        np.savez_compressed(filepath, array=array)
    else:
        np.save(filepath, array)


def load_numpy(filepath: Union[str, Path]) -> np.ndarray:
    """Load numpy array from file.

    Args:
        filepath: Path to the numpy file.

    Returns:
        Loaded numpy array.
    """
    filepath = Path(filepath)

    if filepath.suffix == ".npz":
        data = np.load(filepath)
        return data["array"]
    else:
        return np.load(filepath)
