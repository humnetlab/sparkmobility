"""
File and directory utility functions.
"""

import glob
import os
from typing import List, Optional


class FileHandler:
    """
    Handles file and directory operations for the TimeGeo workflow.
    """

    @staticmethod
    def ensure_dir(directory: str) -> None:
        """
        Ensure that the given directory exists.

        Args:
            directory: Path to the directory to create
        """
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"Directory created: {directory}")

    @staticmethod
    def ensure_directories(directories: List[str]) -> None:
        """
        Ensure that multiple directories exist.

        Args:
            directories: List of directory paths to create
        """
        for directory in directories:
            FileHandler.ensure_dir(directory)

    @staticmethod
    def get_file_size(file_path: str) -> int:
        """
        Get the size of a file in bytes.

        Args:
            file_path: Path to the file

        Returns:
            int: File size in bytes, 0 if file doesn't exist
        """
        return os.path.getsize(file_path) if os.path.exists(file_path) else 0

    @staticmethod
    def file_exists_and_not_empty(file_path: str) -> bool:
        """
        Check if a file exists and is not empty.

        Args:
            file_path: Path to the file

        Returns:
            bool: True if file exists and has content
        """
        return os.path.exists(file_path) and os.path.getsize(file_path) > 0

    @staticmethod
    def list_files_with_pattern(directory: str, pattern: str) -> List[str]:
        """
        List files in a directory matching a pattern.

        Args:
            directory: Directory to search in
            pattern: Glob pattern to match (e.g., "*.txt", "*.parquet")

        Returns:
            List[str]: List of matching file paths
        """
        pattern_path = os.path.join(directory, pattern)
        return glob.glob(pattern_path)

    @staticmethod
    def count_lines_in_file(file_path: str) -> int:
        """
        Count the number of lines in a text file.

        Args:
            file_path: Path to the text file

        Returns:
            int: Number of lines in the file
        """
        if not os.path.exists(file_path):
            return 0

        try:
            with open(file_path, "r") as f:
                return sum(1 for _ in f)
        except Exception as e:
            print(f"Error counting lines in {file_path}: {e}")
            return 0

    @staticmethod
    def backup_file(file_path: str, backup_suffix: str = ".backup") -> Optional[str]:
        """
        Create a backup of a file.

        Args:
            file_path: Path to the file to backup
            backup_suffix: Suffix for the backup file

        Returns:
            Optional[str]: Path to the backup file, None if backup failed
        """
        if not os.path.exists(file_path):
            return None

        backup_path = file_path + backup_suffix
        try:
            import shutil

            shutil.copy2(file_path, backup_path)
            print(f"Backup created: {backup_path}")
            return backup_path
        except Exception as e:
            print(f"Failed to create backup of {file_path}: {e}")
            return None

    @staticmethod
    def clean_directory(directory: str, pattern: str = "*") -> int:
        """
        Remove all files matching a pattern in a directory.

        Args:
            directory: Directory to clean
            pattern: Glob pattern for files to remove

        Returns:
            int: Number of files removed
        """
        if not os.path.exists(directory):
            return 0

        files_to_remove = FileHandler.list_files_with_pattern(directory, pattern)
        removed_count = 0

        for file_path in files_to_remove:
            try:
                os.remove(file_path)
                removed_count += 1
            except Exception as e:
                print(f"Failed to remove {file_path}: {e}")

        print(f"Removed {removed_count} files from {directory}")
        return removed_count

    @staticmethod
    def get_directory_size(directory: str) -> int:
        """
        Calculate the total size of a directory in bytes.

        Args:
            directory: Path to the directory

        Returns:
            int: Total size in bytes
        """
        if not os.path.exists(directory):
            return 0

        total_size = 0
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                try:
                    total_size += os.path.getsize(file_path)
                except OSError:
                    pass

        return total_size

    @staticmethod
    def format_file_size(size_bytes: int) -> str:
        """
        Format file size in human-readable format.

        Args:
            size_bytes: Size in bytes

        Returns:
            str: Formatted size string (e.g., "1.5 MB")
        """
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} PB"
