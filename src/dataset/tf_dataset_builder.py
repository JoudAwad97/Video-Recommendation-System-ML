"""
TensorFlow Dataset builder for training pipelines.

Converts processed DataFrames to tf.data.Dataset objects with proper
batching, shuffling, and prefetching for efficient training.
"""

from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import pandas as pd

from ..config.feature_config import FeatureConfig, DEFAULT_CONFIG
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class TFDatasetBuilder:
    """Build TensorFlow datasets for model training.

    Converts processed DataFrames to tf.data.Dataset objects with:
    - Proper feature specs for Two-Tower and Ranker models
    - Batching, shuffling, and prefetching
    - TFRecord serialization/deserialization

    Example:
        >>> builder = TFDatasetBuilder()
        >>> train_ds = builder.build_two_tower_dataset(train_df, batch_size=256)
        >>> for batch in train_ds:
        ...     user_features, video_features = batch
    """

    def __init__(self, config: FeatureConfig = DEFAULT_CONFIG):
        """Initialize the dataset builder.

        Args:
            config: Feature configuration.
        """
        self.config = config
        self._tf_available = self._check_tensorflow()

    def _check_tensorflow(self) -> bool:
        """Check if TensorFlow is available."""
        try:
            import tensorflow as tf
            return True
        except ImportError:
            logger.warning("TensorFlow not available. Some features will be disabled.")
            return False

    def build_two_tower_dataset(
        self,
        df: pd.DataFrame,
        batch_size: int = 256,
        shuffle: bool = True,
        shuffle_buffer: int = 10000,
        prefetch: bool = True
    ) -> "tf.data.Dataset":
        """Build a TensorFlow dataset for Two-Tower model training.

        Args:
            df: DataFrame with processed Two-Tower features.
            batch_size: Batch size for training.
            shuffle: Whether to shuffle the data.
            shuffle_buffer: Buffer size for shuffling.
            prefetch: Whether to prefetch batches.

        Returns:
            tf.data.Dataset yielding (user_features, video_features) tuples.
        """
        if not self._tf_available:
            raise RuntimeError("TensorFlow is required for this method.")

        import tensorflow as tf

        # Identify user and video feature columns
        user_cols = [c for c in df.columns if any(
            c.startswith(prefix) for prefix in
            ["user_id", "country", "user_language", "age", "prev_category"]
        )]
        video_cols = [c for c in df.columns if any(
            c.startswith(prefix) for prefix in
            ["video_id", "category", "video_language", "popularity", "duration", "title_emb", "tag_emb"]
        )]

        # Create feature dictionaries
        user_features = {col: df[col].values for col in user_cols}
        video_features = {col: df[col].values for col in video_cols}

        # Create dataset from tensors
        dataset = tf.data.Dataset.from_tensor_slices((user_features, video_features))

        # Apply transformations
        if shuffle:
            dataset = dataset.shuffle(buffer_size=shuffle_buffer)

        dataset = dataset.batch(batch_size)

        if prefetch:
            dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    def build_ranker_dataset(
        self,
        df: pd.DataFrame,
        label_col: str = "label",
        batch_size: int = 256,
        shuffle: bool = True,
        shuffle_buffer: int = 10000,
        prefetch: bool = True
    ) -> "tf.data.Dataset":
        """Build a TensorFlow dataset for Ranker model training.

        Args:
            df: DataFrame with processed ranker features.
            label_col: Name of the label column.
            batch_size: Batch size for training.
            shuffle: Whether to shuffle the data.
            shuffle_buffer: Buffer size for shuffling.
            prefetch: Whether to prefetch batches.

        Returns:
            tf.data.Dataset yielding (features, labels) tuples.
        """
        if not self._tf_available:
            raise RuntimeError("TensorFlow is required for this method.")

        import tensorflow as tf

        # Separate features and labels
        feature_cols = [c for c in df.columns if c != label_col]
        features = {col: df[col].values for col in feature_cols}
        labels = df[label_col].values

        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))

        # Apply transformations
        if shuffle:
            dataset = dataset.shuffle(buffer_size=shuffle_buffer)

        dataset = dataset.batch(batch_size)

        if prefetch:
            dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    def get_feature_specs(
        self,
        df: pd.DataFrame,
        model_type: str = "two_tower"
    ) -> Dict[str, "tf.io.FixedLenFeature"]:
        """Get TensorFlow feature specs for parsing TFRecords.

        Args:
            df: Sample DataFrame to infer feature types.
            model_type: Either "two_tower" or "ranker".

        Returns:
            Dictionary of feature specs.
        """
        if not self._tf_available:
            raise RuntimeError("TensorFlow is required for this method.")

        import tensorflow as tf

        feature_specs = {}
        for col in df.columns:
            dtype = df[col].dtype

            if np.issubdtype(dtype, np.integer):
                feature_specs[col] = tf.io.FixedLenFeature([], tf.int64)
            elif np.issubdtype(dtype, np.floating):
                feature_specs[col] = tf.io.FixedLenFeature([], tf.float32)
            else:
                feature_specs[col] = tf.io.FixedLenFeature([], tf.string)

        return feature_specs

    def write_tfrecords(
        self,
        df: pd.DataFrame,
        output_path: Union[str, Path],
        records_per_file: int = 10000
    ) -> List[Path]:
        """Write DataFrame to TFRecord files.

        Args:
            df: DataFrame to write.
            output_path: Base path for output files.
            records_per_file: Number of records per TFRecord file.

        Returns:
            List of written file paths.
        """
        if not self._tf_available:
            raise RuntimeError("TensorFlow is required for this method.")

        import tensorflow as tf

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        written_files = []
        num_files = (len(df) + records_per_file - 1) // records_per_file

        for i in range(num_files):
            start_idx = i * records_per_file
            end_idx = min((i + 1) * records_per_file, len(df))
            chunk = df.iloc[start_idx:end_idx]

            filename = output_path.parent / f"{output_path.stem}_{i:05d}.tfrecord"

            with tf.io.TFRecordWriter(str(filename)) as writer:
                for _, row in chunk.iterrows():
                    example = self._create_tf_example(row)
                    writer.write(example.SerializeToString())

            written_files.append(filename)
            logger.info(f"Written {filename} ({end_idx - start_idx} records)")

        return written_files

    def _create_tf_example(self, row: pd.Series) -> "tf.train.Example":
        """Create a TF Example from a DataFrame row.

        Args:
            row: Single row from DataFrame.

        Returns:
            tf.train.Example proto.
        """
        import tensorflow as tf

        feature_dict = {}
        for col, value in row.items():
            if isinstance(value, (int, np.integer)):
                feature_dict[col] = tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[int(value)])
                )
            elif isinstance(value, (float, np.floating)):
                feature_dict[col] = tf.train.Feature(
                    float_list=tf.train.FloatList(value=[float(value)])
                )
            else:
                feature_dict[col] = tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[str(value).encode()])
                )

        return tf.train.Example(features=tf.train.Features(feature=feature_dict))

    def read_tfrecords(
        self,
        file_pattern: str,
        feature_specs: Dict[str, "tf.io.FixedLenFeature"],
        batch_size: int = 256,
        shuffle: bool = True,
        num_parallel_reads: int = 4
    ) -> "tf.data.Dataset":
        """Read TFRecord files into a dataset.

        Args:
            file_pattern: Glob pattern for TFRecord files.
            feature_specs: Feature specifications for parsing.
            batch_size: Batch size.
            shuffle: Whether to shuffle.
            num_parallel_reads: Number of parallel file reads.

        Returns:
            Parsed tf.data.Dataset.
        """
        if not self._tf_available:
            raise RuntimeError("TensorFlow is required for this method.")

        import tensorflow as tf

        # Get list of files
        files = tf.io.gfile.glob(file_pattern)
        dataset = tf.data.TFRecordDataset(
            files,
            num_parallel_reads=num_parallel_reads
        )

        # Parse function
        def parse_fn(example_proto):
            return tf.io.parse_single_example(example_proto, feature_specs)

        dataset = dataset.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)

        if shuffle:
            dataset = dataset.shuffle(buffer_size=10000)

        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    def create_input_layers(
        self,
        vocab_sizes: Dict[str, int],
        embedding_dims: Dict[str, int]
    ) -> Tuple[Dict, Dict]:
        """Create Keras input layers and embedding layers.

        Args:
            vocab_sizes: Dictionary mapping feature names to vocab sizes.
            embedding_dims: Dictionary mapping feature names to embedding dimensions.

        Returns:
            Tuple of (input_layers, embedding_layers) dictionaries.
        """
        if not self._tf_available:
            raise RuntimeError("TensorFlow is required for this method.")

        import tensorflow as tf

        input_layers = {}
        embedding_layers = {}

        for feature, vocab_size in vocab_sizes.items():
            # Input layer
            input_layers[feature] = tf.keras.layers.Input(
                shape=(1,),
                name=f"{feature}_input",
                dtype=tf.int32
            )

            # Embedding layer
            emb_dim = embedding_dims.get(feature, 32)
            embedding_layers[feature] = tf.keras.layers.Embedding(
                input_dim=vocab_size,
                output_dim=emb_dim,
                name=f"{feature}_embedding"
            )

        return input_layers, embedding_layers

    def __repr__(self) -> str:
        return f"TFDatasetBuilder(tf_available={self._tf_available})"
