"""
Parquet file handling utilities with robust fallback support.
"""

import os
from typing import Any, Dict, Optional

import pandas as pd


class ParquetHandler:
    """
    Handles parquet file operations with robust fallback support.

    This class provides methods for reading and writing parquet files
    with automatic engine detection and fallback mechanisms.
    """

    def __init__(self, preferred_engine: Optional[str] = None):
        """
        Initialize the parquet handler.

        Args:
            preferred_engine: Preferred parquet engine ('pyarrow' or 'fastparquet')
        """
        self.engine = preferred_engine or self._configure_parquet_engine()

    def _configure_parquet_engine(self) -> str:
        """
        Configure the best available parquet engine.

        Returns:
            str: The configured engine name
        """
        try:
            pass

            print("Using PyArrow parquet engine")
            return "pyarrow"
        except (ImportError, ValueError) as e:
            print(f"PyArrow not available or has issues: {e}")
            try:
                pass

                print("Falling back to fastparquet engine")
                return "fastparquet"
            except ImportError:
                print(
                    "Neither PyArrow nor fastparquet available. Installing fastparquet..."
                )
                os.system("conda install fastparquet -y")
                return "fastparquet"

    def read_parquet(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Read parquet file with fallback engine support.

        Args:
            file_path: Path to the parquet file
            **kwargs: Additional arguments to pass to pd.read_parquet

        Returns:
            pd.DataFrame: The loaded dataframe

        Raises:
            FileNotFoundError: If the file doesn't exist
            Exception: If both engines fail to read the file
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Parquet file not found: {file_path}")

        try:
            return pd.read_parquet(file_path, engine=self.engine, **kwargs)
        except Exception as e:
            print(f"Error reading {file_path} with {self.engine}: {e}")
            # Try alternative engine
            alt_engine = "fastparquet" if self.engine == "pyarrow" else "pyarrow"
            print(f"Trying alternative engine: {alt_engine}")
            try:
                return pd.read_parquet(file_path, engine=alt_engine, **kwargs)
            except Exception as e2:
                raise Exception(
                    f"Failed to read {file_path} with both engines. "
                    f"PyArrow error: {e}, {alt_engine} error: {e2}"
                )

    def write_parquet(self, df: pd.DataFrame, file_path: str, **kwargs) -> None:
        """
        Write parquet file with fallback engine support.

        Args:
            df: DataFrame to write
            file_path: Path where to save the parquet file
            **kwargs: Additional arguments to pass to df.to_parquet

        Raises:
            Exception: If both engines fail to write the file
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        try:
            df.to_parquet(file_path, engine=self.engine, **kwargs)
        except Exception as e:
            print(f"Error writing {file_path} with {self.engine}: {e}")
            # Try alternative engine
            alt_engine = "fastparquet" if self.engine == "pyarrow" else "pyarrow"
            print(f"Trying alternative engine: {alt_engine}")
            try:
                df.to_parquet(file_path, engine=alt_engine, **kwargs)
            except Exception as e2:
                raise Exception(
                    f"Failed to write {file_path} with both engines. "
                    f"PyArrow error: {e}, {alt_engine} error: {e2}"
                )

    def consolidate_partitioned_data(self, input_path: str, output_path: str) -> str:
        """
        Consolidate partitioned parquet data into a single file.

        Args:
            input_path: Path to partitioned parquet directory or single file
            output_path: Path for consolidated output file

        Returns:
            str: Path to consolidated file
        """
        if os.path.isdir(input_path):
            print(f"Consolidating partitioned parquet data from: {input_path}")
            # Read all partitions into a single DataFrame
            df = self.read_parquet(input_path)
            partition_count = len(
                [f for f in os.listdir(input_path) if f.endswith(".parquet")]
            )
            print(f"Loaded {len(df)} records from {partition_count} partition files")

            # Save as single consolidated file
            self.write_parquet(df, output_path, index=False)
            print(f"Consolidated data saved to: {output_path}")
            return output_path
        else:
            print(f"Input is already a single file: {input_path}")
            return input_path

    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get information about a parquet file.

        Args:
            file_path: Path to the parquet file

        Returns:
            dict: Information about the file including size, row count, columns
        """
        if not os.path.exists(file_path):
            return {"exists": False}

        try:
            df = self.read_parquet(file_path)
            return {
                "exists": True,
                "size_bytes": os.path.getsize(file_path),
                "rows": len(df),
                "columns": list(df.columns),
                "memory_usage": df.memory_usage(deep=True).sum(),
            }
        except Exception as e:
            return {"exists": True, "error": str(e)}
