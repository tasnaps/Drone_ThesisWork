#!/usr/bin/env python3
"""
File overwrite protection utilities for transformer model evaluation.
This module provides safe file operations with automatic backup and versioning.
"""

import os
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, Union, Tuple
import re


class FileProtectionManager:
    """Manages file overwrite protection with automatic versioning."""

    def __init__(self, backup_enabled: bool = True, timestamp_format: str = "%Y%m%d_%H%M%S"):
        self.backup_enabled = backup_enabled
        self.timestamp_format = timestamp_format

    def get_safe_path(self, filepath: Union[str, Path], overwrite: bool = False) -> Tuple[Path, bool]:
        """
        Get a safe path for saving files, avoiding overwrites unless explicitly allowed.

        Args:
            filepath: The desired file path
            overwrite: Whether to allow overwriting existing files

        Returns:
            Tuple of (safe_path, is_new_name) where is_new_name indicates if the path was modified
        """
        filepath = Path(filepath)

        if not filepath.exists():
            return filepath, False

        if overwrite:
            # Create backup if enabled
            if self.backup_enabled:
                self._create_backup(filepath)
            return filepath, False

        # Find a safe alternative name
        return self._get_versioned_path(filepath), True

    def _get_versioned_path(self, filepath: Path) -> Path:
        """Generate a versioned path that doesn't exist."""
        base_dir = filepath.parent
        stem = filepath.stem
        suffix = filepath.suffix

        # Try timestamp-based naming first
        timestamp = datetime.now().strftime(self.timestamp_format)
        timestamped_path = base_dir / f"{stem}_{timestamp}{suffix}"

        if not timestamped_path.exists():
            return timestamped_path

        # Fall back to numerical versioning
        counter = 1
        while True:
            versioned_path = base_dir / f"{stem}_v{counter:03d}{suffix}"
            if not versioned_path.exists():
                return versioned_path
            counter += 1

            # Safety check to prevent infinite loop
            if counter > 999:
                raise RuntimeError(f"Cannot create safe version of {filepath} - too many versions exist")

    def _create_backup(self, filepath: Path) -> Path:
        """Create a backup of the existing file."""
        timestamp = datetime.now().strftime(self.timestamp_format)
        backup_path = filepath.parent / f"{filepath.stem}_backup_{timestamp}{filepath.suffix}"

        shutil.copy2(filepath, backup_path)
        print(f"📦 Backup created: {backup_path}")
        return backup_path

    def safe_save_plot(self, fig, filepath: Union[str, Path], overwrite: bool = False, **kwargs):
        """Safely save a matplotlib figure."""
        import matplotlib.pyplot as plt

        safe_path, is_renamed = self.get_safe_path(filepath, overwrite)

        if is_renamed:
            print(f"📊 Plot will be saved as: {safe_path} (original exists)")
        else:
            print(f"📊 Saving plot: {safe_path}")

        # Ensure directory exists
        safe_path.parent.mkdir(parents=True, exist_ok=True)

        # Save with default settings if not provided
        save_kwargs = {
            'dpi': 300,
            'bbox_inches': 'tight',
            'facecolor': 'white',
            'edgecolor': 'none'
        }
        save_kwargs.update(kwargs)

        fig.savefig(safe_path, **save_kwargs)
        return safe_path

    def safe_save_csv(self, df, filepath: Union[str, Path], overwrite: bool = False, **kwargs):
        """Safely save a pandas DataFrame to CSV."""
        safe_path, is_renamed = self.get_safe_path(filepath, overwrite)

        if is_renamed:
            print(f"📄 CSV will be saved as: {safe_path} (original exists)")
        else:
            print(f"📄 Saving CSV: {safe_path}")

        # Ensure directory exists
        safe_path.parent.mkdir(parents=True, exist_ok=True)

        # Save with default settings if not provided
        save_kwargs = {
            'index': False,
            'encoding': 'utf-8'
        }
        save_kwargs.update(kwargs)

        df.to_csv(safe_path, **save_kwargs)
        return safe_path

    def safe_save_text(self, content: str, filepath: Union[str, Path], overwrite: bool = False, encoding: str = 'utf-8'):
        """Safely save text content to a file."""
        safe_path, is_renamed = self.get_safe_path(filepath, overwrite)

        if is_renamed:
            print(f"📝 Text file will be saved as: {safe_path} (original exists)")
        else:
            print(f"📝 Saving text file: {safe_path}")

        # Ensure directory exists
        safe_path.parent.mkdir(parents=True, exist_ok=True)

        with open(safe_path, 'w', encoding=encoding) as f:
            f.write(content)

        return safe_path

    def safe_save_json(self, data, filepath: Union[str, Path], overwrite: bool = False, **kwargs):
        """Safely save data to JSON file."""
        import json

        safe_path, is_renamed = self.get_safe_path(filepath, overwrite)

        if is_renamed:
            print(f"🔧 JSON file will be saved as: {safe_path} (original exists)")
        else:
            print(f"🔧 Saving JSON file: {safe_path}")

        # Ensure directory exists
        safe_path.parent.mkdir(parents=True, exist_ok=True)

        # Save with default settings if not provided
        save_kwargs = {
            'indent': 2,
            'ensure_ascii': False
        }
        save_kwargs.update(kwargs)

        with open(safe_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, **save_kwargs)

        return safe_path

    def check_existing_files(self, output_dir: Union[str, Path], patterns: list = None) -> dict:
        """
        Check for existing files that might be overwritten.

        Args:
            output_dir: Directory to check
            patterns: List of file patterns to check (e.g., ['*.csv', '*.png'])

        Returns:
            Dictionary with pattern -> list of existing files
        """
        if patterns is None:
            patterns = ['*.csv', '*.png', '*.txt', '*.json']

        output_dir = Path(output_dir)
        existing_files = {}

        for pattern in patterns:
            files = list(output_dir.glob(pattern))
            if files:
                existing_files[pattern] = files

        return existing_files

    def prompt_user_for_overwrite(self, existing_files: dict) -> str:
        """
        Prompt user about how to handle existing files.

        Returns:
            'overwrite', 'backup', 'version', or 'cancel'
        """
        if not existing_files:
            return 'proceed'

        print("\n⚠️  Existing files detected:")
        for pattern, files in existing_files.items():
            print(f"  {pattern}: {len(files)} files")

        print("\nHow would you like to proceed?")
        print("  [o] Overwrite existing files (creates backups)")
        print("  [v] Create versioned files (recommended)")
        print("  [c] Cancel operation")

        while True:
            choice = input("Your choice [v]: ").lower().strip()
            if choice == '' or choice == 'v':
                return 'version'
            elif choice == 'o':
                return 'overwrite'
            elif choice == 'c':
                return 'cancel'
            else:
                print("Invalid choice. Please enter 'o', 'v', or 'c'.")


# Global instance for easy access
file_protector = FileProtectionManager()


def safe_plt_savefig(filepath: Union[str, Path], overwrite: bool = False, **kwargs):
    """
    Wrapper for matplotlib's savefig with file protection.

    Usage:
        import matplotlib.pyplot as plt
        from file_protection import safe_plt_savefig

        plt.plot([1, 2, 3], [1, 4, 9])
        safe_plt_savefig('my_plot.png')
    """
    import matplotlib.pyplot as plt

    safe_path, is_renamed = file_protector.get_safe_path(filepath, overwrite)

    if is_renamed:
        print(f"📊 Plot will be saved as: {safe_path} (original exists)")

    # Ensure directory exists
    safe_path.parent.mkdir(parents=True, exist_ok=True)

    # Save with default settings if not provided
    save_kwargs = {
        'dpi': 300,
        'bbox_inches': 'tight',
        'facecolor': 'white',
        'edgecolor': 'none'
    }
    save_kwargs.update(kwargs)

    plt.savefig(safe_path, **save_kwargs)
    return safe_path


def safe_df_to_csv(df, filepath: Union[str, Path], overwrite: bool = False, **kwargs):
    """
    Wrapper for pandas DataFrame.to_csv with file protection.

    Usage:
        import pandas as pd
        from file_protection import safe_df_to_csv

        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        safe_df_to_csv(df, 'results.csv')
    """
    return file_protector.safe_save_csv(df, filepath, overwrite, **kwargs)


def safe_write_text(content: str, filepath: Union[str, Path], overwrite: bool = False, encoding: str = 'utf-8'):
    """
    Safely write text content to a file with protection.

    Usage:
        from file_protection import safe_write_text

        safe_write_text("My analysis results...", "report.txt")
    """
    return file_protector.safe_save_text(content, filepath, overwrite, encoding)


def check_and_handle_existing_files(output_dir: Union[str, Path], auto_version: bool = False) -> bool:
    """
    Check for existing files and handle them based on user preference or auto-versioning.

    Args:
        output_dir: Directory to check
        auto_version: If True, automatically use versioning without prompting

    Returns:
        True if operation should continue, False if cancelled
    """
    existing_files = file_protector.check_existing_files(output_dir)

    if not existing_files:
        print("✅ No existing files detected - proceeding with evaluation")
        return True

    if auto_version:
        print("🔄 Auto-versioning enabled - will create new versions of existing files")
        return True

    action = file_protector.prompt_user_for_overwrite(existing_files)

    if action == 'cancel':
        print("❌ Operation cancelled by user")
        return False
    elif action == 'overwrite':
        print("⚠️  Will overwrite existing files (backups will be created)")
        file_protector.backup_enabled = True
        return True
    else:  # version
        print("🔄 Will create versioned files to avoid overwrites")
        return True
