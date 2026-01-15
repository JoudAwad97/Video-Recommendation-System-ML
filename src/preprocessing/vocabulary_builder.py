"""
Vocabulary builders for categorical features.

Provides vocabulary builders for string and integer lookup operations,
supporting both small vocabularies and hash buckets for large cardinality features.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union, Set
from pathlib import Path
import numpy as np
import pandas as pd

from ..config.feature_config import UNK_TOKEN, PAD_TOKEN, START_TOKEN
from ..utils.io_utils import save_json, load_json


class VocabularyBuilder(ABC):
    """Abstract base class for vocabulary builders."""

    @abstractmethod
    def build(self, values: Union[List, np.ndarray, pd.Series]) -> "VocabularyBuilder":
        """Build vocabulary from values."""
        pass

    @abstractmethod
    def lookup(self, value: Union[str, int]) -> int:
        """Look up index for a value."""
        pass

    @abstractmethod
    def inverse_lookup(self, index: int) -> Union[str, int]:
        """Look up value for an index."""
        pass

    @abstractmethod
    def save(self, filepath: Union[str, Path]) -> None:
        """Save vocabulary to file."""
        pass

    @abstractmethod
    def load(self, filepath: Union[str, Path]) -> "VocabularyBuilder":
        """Load vocabulary from file."""
        pass

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        pass


class StringLookupVocabulary(VocabularyBuilder):
    """Vocabulary builder for string categorical features.

    Maps string values to integer indices with support for special tokens
    (UNK, PAD, START) and optional mask value handling.

    Example:
        >>> vocab = StringLookupVocabulary(name="category")
        >>> vocab.build(["Tech", "Gaming", "Music"])
        >>> vocab.lookup("Tech")  # Returns 2 (0=PAD, 1=UNK)
        >>> vocab.lookup("Unknown")  # Returns 1 (UNK token)
    """

    def __init__(
        self,
        name: str,
        include_unk: bool = True,
        include_pad: bool = True,
        include_start: bool = False,
        mask_value: Optional[str] = None,
    ):
        """Initialize the string lookup vocabulary.

        Args:
            name: Name of the feature/vocabulary.
            include_unk: Whether to include UNK token for unknown values.
            include_pad: Whether to include PAD token for padding.
            include_start: Whether to include START token (for sequence features).
            mask_value: Value to treat as special start token (e.g., "-").
        """
        self.name = name
        self.include_unk = include_unk
        self.include_pad = include_pad
        self.include_start = include_start
        self.mask_value = mask_value

        self._vocab: Dict[str, int] = {}
        self._inverse_vocab: Dict[int, str] = {}
        self._is_built = False

    def build(self, values: Union[List[str], np.ndarray, pd.Series]) -> "StringLookupVocabulary":
        """Build vocabulary from string values.

        Args:
            values: Iterable of string values to build vocabulary from.

        Returns:
            Self for method chaining.
        """
        # Convert to set to get unique values
        if isinstance(values, pd.Series):
            unique_values = set(values.dropna().unique())
        else:
            unique_values = set(v for v in values if v is not None and pd.notna(v))

        # Remove mask value from vocabulary (will be mapped to START)
        if self.mask_value and self.mask_value in unique_values:
            unique_values.discard(self.mask_value)

        # Sort for deterministic ordering
        sorted_values = sorted(unique_values)

        # Build vocabulary with special tokens
        self._vocab = {}
        idx = 0

        if self.include_pad:
            self._vocab[PAD_TOKEN] = idx
            idx += 1

        if self.include_unk:
            self._vocab[UNK_TOKEN] = idx
            idx += 1

        if self.include_start:
            self._vocab[START_TOKEN] = idx
            idx += 1

        # Add regular values
        for value in sorted_values:
            self._vocab[str(value)] = idx
            idx += 1

        # Build inverse vocabulary
        self._inverse_vocab = {v: k for k, v in self._vocab.items()}
        self._is_built = True

        return self

    def lookup(self, value: Union[str, None]) -> int:
        """Look up index for a string value.

        Args:
            value: String value to look up.

        Returns:
            Integer index for the value.
        """
        if not self._is_built:
            raise RuntimeError("Vocabulary not built. Call build() first.")

        # Handle None/NaN
        if value is None or (isinstance(value, float) and np.isnan(value)):
            if self.include_pad:
                return self._vocab[PAD_TOKEN]
            elif self.include_unk:
                return self._vocab[UNK_TOKEN]
            return 0

        # Handle mask value (map to START token)
        if self.mask_value and str(value) == self.mask_value:
            if self.include_start:
                return self._vocab[START_TOKEN]
            elif self.include_unk:
                return self._vocab[UNK_TOKEN]
            return 0

        # Regular lookup
        str_value = str(value)
        if str_value in self._vocab:
            return self._vocab[str_value]
        elif self.include_unk:
            return self._vocab[UNK_TOKEN]
        else:
            raise ValueError(f"Unknown value '{value}' and no UNK token configured")

    def lookup_batch(self, values: Union[List[str], np.ndarray, pd.Series]) -> np.ndarray:
        """Look up indices for a batch of values.

        Args:
            values: Iterable of string values.

        Returns:
            Numpy array of indices.
        """
        return np.array([self.lookup(v) for v in values])

    def inverse_lookup(self, index: int) -> str:
        """Look up string value for an index.

        Args:
            index: Integer index to look up.

        Returns:
            String value for the index.
        """
        if not self._is_built:
            raise RuntimeError("Vocabulary not built. Call build() first.")

        if index in self._inverse_vocab:
            return self._inverse_vocab[index]
        else:
            raise ValueError(f"Index {index} not in vocabulary")

    def save(self, filepath: Union[str, Path]) -> None:
        """Save vocabulary to JSON file.

        Args:
            filepath: Path to save the vocabulary.
        """
        if not self._is_built:
            raise RuntimeError("Vocabulary not built. Call build() first.")

        data = {
            "name": self.name,
            "include_unk": self.include_unk,
            "include_pad": self.include_pad,
            "include_start": self.include_start,
            "mask_value": self.mask_value,
            "vocab": self._vocab,
        }
        save_json(data, filepath)

    def load(self, filepath: Union[str, Path]) -> "StringLookupVocabulary":
        """Load vocabulary from JSON file.

        Args:
            filepath: Path to load the vocabulary from.

        Returns:
            Self for method chaining.
        """
        data = load_json(filepath)

        self.name = data["name"]
        self.include_unk = data["include_unk"]
        self.include_pad = data["include_pad"]
        self.include_start = data["include_start"]
        self.mask_value = data["mask_value"]
        self._vocab = data["vocab"]
        self._inverse_vocab = {int(v): k for k, v in self._vocab.items()}
        self._is_built = True

        return self

    @property
    def vocab_size(self) -> int:
        """Return vocabulary size including special tokens."""
        return len(self._vocab)

    @property
    def vocabulary(self) -> Dict[str, int]:
        """Return the vocabulary dictionary."""
        return self._vocab.copy()

    def __repr__(self) -> str:
        return f"StringLookupVocabulary(name='{self.name}', size={self.vocab_size})"


class IntegerLookupVocabulary(VocabularyBuilder):
    """Vocabulary builder for integer ID features.

    Maps integer IDs to sequential indices. Useful for user_id, video_id, etc.

    Example:
        >>> vocab = IntegerLookupVocabulary(name="user_id")
        >>> vocab.build([1001, 1002, 1003, 2001])
        >>> vocab.lookup(1001)  # Returns 2 (0=PAD, 1=UNK)
        >>> vocab.lookup(9999)  # Returns 1 (UNK token)
    """

    def __init__(
        self,
        name: str,
        include_unk: bool = True,
        include_pad: bool = True,
    ):
        """Initialize the integer lookup vocabulary.

        Args:
            name: Name of the feature/vocabulary.
            include_unk: Whether to include UNK token for unknown values.
            include_pad: Whether to include PAD token for padding.
        """
        self.name = name
        self.include_unk = include_unk
        self.include_pad = include_pad

        self._vocab: Dict[int, int] = {}
        self._inverse_vocab: Dict[int, int] = {}
        self._is_built = False

        # Reserve indices for special tokens
        self._pad_idx = 0 if include_pad else None
        self._unk_idx = (1 if include_pad else 0) if include_unk else None

    def build(self, values: Union[List[int], np.ndarray, pd.Series]) -> "IntegerLookupVocabulary":
        """Build vocabulary from integer values.

        Args:
            values: Iterable of integer values to build vocabulary from.

        Returns:
            Self for method chaining.
        """
        # Get unique values
        if isinstance(values, pd.Series):
            unique_values = set(values.dropna().astype(int).unique())
        else:
            unique_values = set(int(v) for v in values if v is not None and pd.notna(v))

        # Sort for deterministic ordering
        sorted_values = sorted(unique_values)

        # Build vocabulary
        self._vocab = {}
        idx = 0

        # Reserve special token indices
        if self.include_pad:
            idx += 1
        if self.include_unk:
            idx += 1

        # Add regular values
        for value in sorted_values:
            self._vocab[int(value)] = idx
            idx += 1

        # Build inverse vocabulary
        self._inverse_vocab = {v: k for k, v in self._vocab.items()}

        # Add special tokens to inverse
        if self.include_pad:
            self._inverse_vocab[self._pad_idx] = -1  # PAD represented as -1
        if self.include_unk:
            self._inverse_vocab[self._unk_idx] = -2  # UNK represented as -2

        self._is_built = True
        return self

    def lookup(self, value: Union[int, None]) -> int:
        """Look up index for an integer value.

        Args:
            value: Integer value to look up.

        Returns:
            Integer index for the value.
        """
        if not self._is_built:
            raise RuntimeError("Vocabulary not built. Call build() first.")

        # Handle None/NaN
        if value is None or (isinstance(value, float) and np.isnan(value)):
            if self._pad_idx is not None:
                return self._pad_idx
            elif self._unk_idx is not None:
                return self._unk_idx
            return 0

        # Regular lookup
        int_value = int(value)
        if int_value in self._vocab:
            return self._vocab[int_value]
        elif self._unk_idx is not None:
            return self._unk_idx
        else:
            raise ValueError(f"Unknown value '{value}' and no UNK token configured")

    def lookup_batch(self, values: Union[List[int], np.ndarray, pd.Series]) -> np.ndarray:
        """Look up indices for a batch of values.

        Args:
            values: Iterable of integer values.

        Returns:
            Numpy array of indices.
        """
        return np.array([self.lookup(v) for v in values])

    def inverse_lookup(self, index: int) -> int:
        """Look up integer value for an index.

        Args:
            index: Integer index to look up.

        Returns:
            Original integer value for the index.
        """
        if not self._is_built:
            raise RuntimeError("Vocabulary not built. Call build() first.")

        if index in self._inverse_vocab:
            return self._inverse_vocab[index]
        else:
            raise ValueError(f"Index {index} not in vocabulary")

    def save(self, filepath: Union[str, Path]) -> None:
        """Save vocabulary to JSON file.

        Args:
            filepath: Path to save the vocabulary.
        """
        if not self._is_built:
            raise RuntimeError("Vocabulary not built. Call build() first.")

        # Convert int keys to strings for JSON
        vocab_str_keys = {str(k): v for k, v in self._vocab.items()}

        data = {
            "name": self.name,
            "include_unk": self.include_unk,
            "include_pad": self.include_pad,
            "vocab": vocab_str_keys,
        }
        save_json(data, filepath)

    def load(self, filepath: Union[str, Path]) -> "IntegerLookupVocabulary":
        """Load vocabulary from JSON file.

        Args:
            filepath: Path to load the vocabulary from.

        Returns:
            Self for method chaining.
        """
        data = load_json(filepath)

        self.name = data["name"]
        self.include_unk = data["include_unk"]
        self.include_pad = data["include_pad"]

        # Convert string keys back to integers
        self._vocab = {int(k): v for k, v in data["vocab"].items()}
        self._inverse_vocab = {v: k for k, v in self._vocab.items()}

        # Reconstruct special token indices
        self._pad_idx = 0 if self.include_pad else None
        self._unk_idx = (1 if self.include_pad else 0) if self.include_unk else None

        if self.include_pad:
            self._inverse_vocab[self._pad_idx] = -1
        if self.include_unk:
            self._inverse_vocab[self._unk_idx] = -2

        self._is_built = True
        return self

    @property
    def vocab_size(self) -> int:
        """Return vocabulary size including special tokens."""
        size = len(self._vocab)
        if self.include_pad:
            size += 1
        if self.include_unk:
            size += 1
        return size

    @property
    def vocabulary(self) -> Dict[int, int]:
        """Return the vocabulary dictionary."""
        return self._vocab.copy()

    def __repr__(self) -> str:
        return f"IntegerLookupVocabulary(name='{self.name}', size={self.vocab_size})"


class HashBucketVocabulary:
    """Hash bucket vocabulary for very large cardinality features.

    Uses hashing to map values to a fixed number of buckets.
    Useful when the vocabulary is too large to enumerate.

    Example:
        >>> vocab = HashBucketVocabulary(name="user_id", num_buckets=100000)
        >>> vocab.lookup(123456789)  # Returns hash bucket index
    """

    def __init__(self, name: str, num_buckets: int):
        """Initialize the hash bucket vocabulary.

        Args:
            name: Name of the feature.
            num_buckets: Number of hash buckets.
        """
        self.name = name
        self.num_buckets = num_buckets

    def lookup(self, value: Union[str, int]) -> int:
        """Look up bucket index for a value.

        Args:
            value: Value to hash.

        Returns:
            Hash bucket index.
        """
        return hash(str(value)) % self.num_buckets

    def lookup_batch(self, values: Union[List, np.ndarray, pd.Series]) -> np.ndarray:
        """Look up bucket indices for a batch of values.

        Args:
            values: Iterable of values.

        Returns:
            Numpy array of bucket indices.
        """
        return np.array([self.lookup(v) for v in values])

    @property
    def vocab_size(self) -> int:
        """Return number of buckets."""
        return self.num_buckets

    def save(self, filepath: Union[str, Path]) -> None:
        """Save vocabulary config to JSON file."""
        data = {
            "name": self.name,
            "num_buckets": self.num_buckets,
            "type": "hash_bucket",
        }
        save_json(data, filepath)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "HashBucketVocabulary":
        """Load vocabulary config from JSON file."""
        data = load_json(filepath)
        return cls(name=data["name"], num_buckets=data["num_buckets"])

    def __repr__(self) -> str:
        return f"HashBucketVocabulary(name='{self.name}', buckets={self.num_buckets})"
