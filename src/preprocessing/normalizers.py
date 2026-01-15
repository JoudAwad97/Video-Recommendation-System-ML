"""
Numeric feature normalizers and transformers.

Provides various normalization and transformation strategies for numeric features
including standard normalization, min-max scaling, log transforms, bucketing,
and cyclical encoding for time features.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union, Tuple
from pathlib import Path
import numpy as np
import pandas as pd

from ..utils.io_utils import save_json, load_json


class BaseTransformer(ABC):
    """Abstract base class for numeric transformers."""

    @abstractmethod
    def fit(self, values: Union[List, np.ndarray, pd.Series]) -> "BaseTransformer":
        """Fit the transformer to the data."""
        pass

    @abstractmethod
    def transform(self, values: Union[List, np.ndarray, pd.Series]) -> np.ndarray:
        """Transform values."""
        pass

    def fit_transform(self, values: Union[List, np.ndarray, pd.Series]) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(values).transform(values)

    @abstractmethod
    def save(self, filepath: Union[str, Path]) -> None:
        """Save transformer state to file."""
        pass

    @abstractmethod
    def load(self, filepath: Union[str, Path]) -> "BaseTransformer":
        """Load transformer state from file."""
        pass


class StandardNormalizer(BaseTransformer):
    """Standard (Z-score) normalization.

    Transforms values to have mean=0 and std=1.

    Example:
        >>> normalizer = StandardNormalizer(name="age")
        >>> normalizer.fit([20, 30, 40, 50, 60])
        >>> normalizer.transform([25, 35])  # Returns z-scores
    """

    def __init__(self, name: str, epsilon: float = 1e-8):
        """Initialize the standard normalizer.

        Args:
            name: Name of the feature.
            epsilon: Small value to prevent division by zero.
        """
        self.name = name
        self.epsilon = epsilon
        self.mean: Optional[float] = None
        self.std: Optional[float] = None
        self._is_fitted = False

    def fit(self, values: Union[List, np.ndarray, pd.Series]) -> "StandardNormalizer":
        """Fit the normalizer by computing mean and std.

        Args:
            values: Numeric values to fit on.

        Returns:
            Self for method chaining.
        """
        arr = np.asarray(values, dtype=np.float64)
        arr = arr[~np.isnan(arr)]  # Remove NaN values

        self.mean = float(np.mean(arr))
        self.std = float(np.std(arr))
        self._is_fitted = True

        return self

    def transform(self, values: Union[List, np.ndarray, pd.Series]) -> np.ndarray:
        """Transform values using z-score normalization.

        Args:
            values: Values to transform.

        Returns:
            Normalized values.
        """
        if not self._is_fitted:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")

        arr = np.asarray(values, dtype=np.float64)
        return (arr - self.mean) / (self.std + self.epsilon)

    def inverse_transform(self, values: Union[List, np.ndarray, pd.Series]) -> np.ndarray:
        """Inverse transform to get original scale.

        Args:
            values: Normalized values.

        Returns:
            Original scale values.
        """
        if not self._is_fitted:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")

        arr = np.asarray(values, dtype=np.float64)
        return arr * self.std + self.mean

    def get_stats(self) -> Dict[str, float]:
        """Get computed statistics.

        Returns:
            Dictionary with mean and std.
        """
        return {"mean": self.mean, "std": self.std}

    def save(self, filepath: Union[str, Path]) -> None:
        """Save normalizer state to JSON file."""
        if not self._is_fitted:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")

        data = {
            "name": self.name,
            "type": "standard",
            "mean": self.mean,
            "std": self.std,
            "epsilon": self.epsilon,
        }
        save_json(data, filepath)

    def load(self, filepath: Union[str, Path]) -> "StandardNormalizer":
        """Load normalizer state from JSON file."""
        data = load_json(filepath)

        self.name = data["name"]
        self.mean = data["mean"]
        self.std = data["std"]
        self.epsilon = data.get("epsilon", 1e-8)
        self._is_fitted = True

        return self

    def __repr__(self) -> str:
        if self._is_fitted:
            return f"StandardNormalizer(name='{self.name}', mean={self.mean:.4f}, std={self.std:.4f})"
        return f"StandardNormalizer(name='{self.name}', not fitted)"


class MinMaxNormalizer(BaseTransformer):
    """Min-Max normalization.

    Scales values to [0, 1] range.

    Example:
        >>> normalizer = MinMaxNormalizer(name="duration")
        >>> normalizer.fit([60, 300, 600, 1800])
        >>> normalizer.transform([180])  # Returns scaled value in [0, 1]
    """

    def __init__(self, name: str, feature_range: Tuple[float, float] = (0.0, 1.0)):
        """Initialize the min-max normalizer.

        Args:
            name: Name of the feature.
            feature_range: Target range (min, max).
        """
        self.name = name
        self.feature_range = feature_range
        self.min_val: Optional[float] = None
        self.max_val: Optional[float] = None
        self._is_fitted = False

    def fit(self, values: Union[List, np.ndarray, pd.Series]) -> "MinMaxNormalizer":
        """Fit the normalizer by computing min and max.

        Args:
            values: Numeric values to fit on.

        Returns:
            Self for method chaining.
        """
        arr = np.asarray(values, dtype=np.float64)
        arr = arr[~np.isnan(arr)]

        self.min_val = float(np.min(arr))
        self.max_val = float(np.max(arr))
        self._is_fitted = True

        return self

    def transform(self, values: Union[List, np.ndarray, pd.Series]) -> np.ndarray:
        """Transform values using min-max scaling.

        Args:
            values: Values to transform.

        Returns:
            Scaled values in feature_range.
        """
        if not self._is_fitted:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")

        arr = np.asarray(values, dtype=np.float64)

        # Scale to [0, 1]
        range_val = self.max_val - self.min_val
        if range_val == 0:
            scaled = np.zeros_like(arr)
        else:
            scaled = (arr - self.min_val) / range_val

        # Scale to feature_range
        min_range, max_range = self.feature_range
        return scaled * (max_range - min_range) + min_range

    def inverse_transform(self, values: Union[List, np.ndarray, pd.Series]) -> np.ndarray:
        """Inverse transform to get original scale."""
        if not self._is_fitted:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")

        arr = np.asarray(values, dtype=np.float64)
        min_range, max_range = self.feature_range

        # Unscale from feature_range to [0, 1]
        scaled = (arr - min_range) / (max_range - min_range)

        # Unscale to original range
        return scaled * (self.max_val - self.min_val) + self.min_val

    def get_stats(self) -> Dict[str, float]:
        """Get computed statistics."""
        return {"min": self.min_val, "max": self.max_val}

    def save(self, filepath: Union[str, Path]) -> None:
        """Save normalizer state to JSON file."""
        if not self._is_fitted:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")

        data = {
            "name": self.name,
            "type": "minmax",
            "min": self.min_val,
            "max": self.max_val,
            "feature_range": list(self.feature_range),
        }
        save_json(data, filepath)

    def load(self, filepath: Union[str, Path]) -> "MinMaxNormalizer":
        """Load normalizer state from JSON file."""
        data = load_json(filepath)

        self.name = data["name"]
        self.min_val = data["min"]
        self.max_val = data["max"]
        self.feature_range = tuple(data.get("feature_range", [0.0, 1.0]))
        self._is_fitted = True

        return self

    def __repr__(self) -> str:
        if self._is_fitted:
            return f"MinMaxNormalizer(name='{self.name}', min={self.min_val:.4f}, max={self.max_val:.4f})"
        return f"MinMaxNormalizer(name='{self.name}', not fitted)"


class LogTransformer(BaseTransformer):
    """Log transformation for skewed distributions.

    Applies log1p (log(1 + x)) transformation to handle zero values.

    Example:
        >>> transformer = LogTransformer(name="view_count")
        >>> transformer.transform([0, 100, 10000, 1000000])
    """

    def __init__(self, name: str, base: str = "natural"):
        """Initialize the log transformer.

        Args:
            name: Name of the feature.
            base: Log base ("natural", "10", "2").
        """
        self.name = name
        self.base = base
        self._is_fitted = True  # No fitting required

    def fit(self, values: Union[List, np.ndarray, pd.Series]) -> "LogTransformer":
        """Log transformer doesn't require fitting.

        Args:
            values: Ignored.

        Returns:
            Self for method chaining.
        """
        return self

    def transform(self, values: Union[List, np.ndarray, pd.Series]) -> np.ndarray:
        """Apply log transformation.

        Args:
            values: Values to transform.

        Returns:
            Log-transformed values.
        """
        arr = np.asarray(values, dtype=np.float64)

        # Handle negative values by shifting
        min_val = np.nanmin(arr)
        if min_val < 0:
            arr = arr - min_val

        # Apply log1p
        if self.base == "natural":
            return np.log1p(arr)
        elif self.base == "10":
            return np.log10(arr + 1)
        elif self.base == "2":
            return np.log2(arr + 1)
        else:
            return np.log1p(arr)

    def inverse_transform(self, values: Union[List, np.ndarray, pd.Series]) -> np.ndarray:
        """Inverse log transformation.

        Args:
            values: Log-transformed values.

        Returns:
            Original scale values.
        """
        arr = np.asarray(values, dtype=np.float64)

        if self.base == "natural":
            return np.expm1(arr)
        elif self.base == "10":
            return np.power(10, arr) - 1
        elif self.base == "2":
            return np.power(2, arr) - 1
        else:
            return np.expm1(arr)

    def save(self, filepath: Union[str, Path]) -> None:
        """Save transformer config to JSON file."""
        data = {
            "name": self.name,
            "type": "log",
            "base": self.base,
        }
        save_json(data, filepath)

    def load(self, filepath: Union[str, Path]) -> "LogTransformer":
        """Load transformer config from JSON file."""
        data = load_json(filepath)
        self.name = data["name"]
        self.base = data.get("base", "natural")
        return self

    def __repr__(self) -> str:
        return f"LogTransformer(name='{self.name}', base='{self.base}')"


class BucketTransformer(BaseTransformer):
    """Bucket/bin numeric values into discrete categories.

    Useful for age groups, duration buckets, subscriber tiers, etc.

    Example:
        >>> transformer = BucketTransformer(
        ...     name="age",
        ...     boundaries=[18, 25, 35, 45, 55, 65]
        ... )
        >>> transformer.transform([15, 22, 30, 50, 70])
        # Returns [0, 1, 2, 4, 6] (bucket indices)
    """

    def __init__(
        self,
        name: str,
        boundaries: Optional[List[float]] = None,
        num_buckets: Optional[int] = None
    ):
        """Initialize the bucket transformer.

        Args:
            name: Name of the feature.
            boundaries: Explicit bucket boundaries. If None, will be computed from data.
            num_buckets: Number of buckets to create (used if boundaries is None).
        """
        self.name = name
        self.boundaries = boundaries
        self.num_buckets = num_buckets
        self._is_fitted = boundaries is not None

    def fit(self, values: Union[List, np.ndarray, pd.Series]) -> "BucketTransformer":
        """Fit the transformer by computing bucket boundaries.

        Only needed if boundaries weren't provided at initialization.

        Args:
            values: Numeric values to compute boundaries from.

        Returns:
            Self for method chaining.
        """
        if self.boundaries is not None:
            return self

        arr = np.asarray(values, dtype=np.float64)
        arr = arr[~np.isnan(arr)]

        if self.num_buckets is None:
            self.num_buckets = 10

        # Compute quantile-based boundaries
        percentiles = np.linspace(0, 100, self.num_buckets + 1)[1:-1]
        self.boundaries = np.percentile(arr, percentiles).tolist()
        self._is_fitted = True

        return self

    def transform(self, values: Union[List, np.ndarray, pd.Series]) -> np.ndarray:
        """Transform values to bucket indices.

        Args:
            values: Values to bucket.

        Returns:
            Bucket indices (0 to num_buckets).
        """
        if not self._is_fitted:
            raise RuntimeError("Transformer not fitted. Call fit() first or provide boundaries.")

        arr = np.asarray(values, dtype=np.float64)
        return np.digitize(arr, self.boundaries)

    def get_bucket_labels(self) -> List[str]:
        """Get human-readable bucket labels.

        Returns:
            List of bucket label strings.
        """
        if not self._is_fitted:
            raise RuntimeError("Transformer not fitted.")

        labels = []
        boundaries = [-float('inf')] + self.boundaries + [float('inf')]

        for i in range(len(boundaries) - 1):
            if boundaries[i] == -float('inf'):
                labels.append(f"< {boundaries[i+1]}")
            elif boundaries[i+1] == float('inf'):
                labels.append(f">= {boundaries[i]}")
            else:
                labels.append(f"{boundaries[i]} - {boundaries[i+1]}")

        return labels

    @property
    def num_output_buckets(self) -> int:
        """Return number of output buckets."""
        if not self._is_fitted:
            raise RuntimeError("Transformer not fitted.")
        return len(self.boundaries) + 1

    def save(self, filepath: Union[str, Path]) -> None:
        """Save transformer to JSON file."""
        if not self._is_fitted:
            raise RuntimeError("Transformer not fitted. Call fit() first.")

        data = {
            "name": self.name,
            "type": "bucket",
            "boundaries": self.boundaries,
        }
        save_json(data, filepath)

    def load(self, filepath: Union[str, Path]) -> "BucketTransformer":
        """Load transformer from JSON file."""
        data = load_json(filepath)

        self.name = data["name"]
        self.boundaries = data["boundaries"]
        self._is_fitted = True

        return self

    def __repr__(self) -> str:
        if self._is_fitted:
            return f"BucketTransformer(name='{self.name}', buckets={self.num_output_buckets})"
        return f"BucketTransformer(name='{self.name}', not fitted)"


class CyclicalEncoder(BaseTransformer):
    """Cyclical encoding for periodic features.

    Encodes periodic features (hour, day of week, month) using
    sin/cos transformation to preserve cyclical nature.

    Example:
        >>> encoder = CyclicalEncoder(name="hour", period=24)
        >>> encoder.transform([0, 6, 12, 18])
        # Returns array with sin and cos components
    """

    def __init__(self, name: str, period: float):
        """Initialize the cyclical encoder.

        Args:
            name: Name of the feature.
            period: Period of the cycle (e.g., 24 for hours, 7 for days).
        """
        self.name = name
        self.period = period
        self._is_fitted = True  # No fitting required

    def fit(self, values: Union[List, np.ndarray, pd.Series]) -> "CyclicalEncoder":
        """Cyclical encoder doesn't require fitting.

        Args:
            values: Ignored.

        Returns:
            Self for method chaining.
        """
        return self

    def transform(self, values: Union[List, np.ndarray, pd.Series]) -> np.ndarray:
        """Transform values to sin/cos encoding.

        Args:
            values: Values to encode.

        Returns:
            Array of shape (n, 2) with [sin, cos] components.
        """
        arr = np.asarray(values, dtype=np.float64)
        angle = 2 * np.pi * arr / self.period

        sin_component = np.sin(angle)
        cos_component = np.cos(angle)

        return np.column_stack([sin_component, cos_component])

    def transform_sin(self, values: Union[List, np.ndarray, pd.Series]) -> np.ndarray:
        """Get only the sin component."""
        arr = np.asarray(values, dtype=np.float64)
        angle = 2 * np.pi * arr / self.period
        return np.sin(angle)

    def transform_cos(self, values: Union[List, np.ndarray, pd.Series]) -> np.ndarray:
        """Get only the cos component."""
        arr = np.asarray(values, dtype=np.float64)
        angle = 2 * np.pi * arr / self.period
        return np.cos(angle)

    def save(self, filepath: Union[str, Path]) -> None:
        """Save encoder config to JSON file."""
        data = {
            "name": self.name,
            "type": "cyclical",
            "period": self.period,
        }
        save_json(data, filepath)

    def load(self, filepath: Union[str, Path]) -> "CyclicalEncoder":
        """Load encoder config from JSON file."""
        data = load_json(filepath)
        self.name = data["name"]
        self.period = data["period"]
        return self

    def __repr__(self) -> str:
        return f"CyclicalEncoder(name='{self.name}', period={self.period})"


class CompositeTransformer:
    """Combine multiple transformers in sequence.

    Useful for applying log transform followed by normalization, etc.

    Example:
        >>> transformer = CompositeTransformer(name="view_count", transformers=[
        ...     LogTransformer(name="view_count"),
        ...     StandardNormalizer(name="view_count"),
        ... ])
        >>> transformer.fit_transform([100, 1000, 10000, 100000])
    """

    def __init__(self, name: str, transformers: List[BaseTransformer]):
        """Initialize composite transformer.

        Args:
            name: Name of the feature.
            transformers: List of transformers to apply in sequence.
        """
        self.name = name
        self.transformers = transformers

    def fit(self, values: Union[List, np.ndarray, pd.Series]) -> "CompositeTransformer":
        """Fit all transformers in sequence.

        Args:
            values: Values to fit on.

        Returns:
            Self for method chaining.
        """
        current_values = values
        for transformer in self.transformers:
            transformer.fit(current_values)
            current_values = transformer.transform(current_values)
        return self

    def transform(self, values: Union[List, np.ndarray, pd.Series]) -> np.ndarray:
        """Apply all transformers in sequence.

        Args:
            values: Values to transform.

        Returns:
            Transformed values.
        """
        current_values = values
        for transformer in self.transformers:
            current_values = transformer.transform(current_values)
        return current_values

    def fit_transform(self, values: Union[List, np.ndarray, pd.Series]) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(values).transform(values)

    def __repr__(self) -> str:
        transformer_names = [t.__class__.__name__ for t in self.transformers]
        return f"CompositeTransformer(name='{self.name}', steps={transformer_names})"
