#!/usr/bin/env python3
"""
Output management utilities for organizing evaluation results.
This module provides centralized management of all evaluation outputs.
"""

import shutil
import json
from datetime import datetime
from pathlib import Path
from typing import Dict

class EvaluationOutputManager:
    """Manages all evaluation outputs in a structured format."""

    def __init__(self, base_output_dir: str = "./eval_results"):
        """
        Initialize the output manager.

        Args:
            base_output_dir: Base directory for all evaluation outputs
        """
        self.base_output_dir = Path(base_output_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Define the standard output structure
        self.structure = {
            'runs': self.base_output_dir / 'runs',
            'summaries': self.base_output_dir / 'summaries',
            'plots': self.base_output_dir / 'plots',
            'reports': self.base_output_dir / 'reports',
            'datasets': self.base_output_dir / 'datasets',
            'models': self.base_output_dir / 'models',
            'archive': self.base_output_dir / 'archive'
        }

        # Create the directory structure
        self._create_structure()

    def _create_structure(self):
        """Create the standard directory structure."""
        for directory in self.structure.values():
            directory.mkdir(parents=True, exist_ok=True)

    def get_run_directory(self, run_name: str = None) -> Path:
        """
        Get a directory for a specific evaluation run.

        Args:
            run_name: Optional name for the run, defaults to timestamp

        Returns:
            Path to the run directory
        """
        if run_name is None:
            run_name = f"run_{self.timestamp}"

        run_dir = self.structure['runs'] / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for this run
        subdirs = ['csv', 'plots', 'logs', 'configs']
        for subdir in subdirs:
            (run_dir / subdir).mkdir(exist_ok=True)

        return run_dir

    def save_csv_results(self, data, filename: str, run_name: str = None,
                        dataset_specific: bool = False):
        """
        Save CSV results in an organized manner.

        Args:
            data: DataFrame or data to save
            filename: Name of the CSV file
            run_name: Run identifier
            dataset_specific: If True, save in dataset-specific location
        """
        if dataset_specific:
            output_dir = self.structure['datasets']
        else:
            run_dir = self.get_run_directory(run_name)
            output_dir = run_dir / 'csv'

        output_path = output_dir / filename

        # Save the data (assuming pandas DataFrame)
        if hasattr(data, 'to_csv'):
            data.to_csv(output_path, index=False)
        else:
            # Handle other data types as needed
            import pandas as pd
            if isinstance(data, dict):
                pd.DataFrame(data).to_csv(output_path, index=False)
            else:
                raise ValueError(f"Unsupported data type for CSV export: {type(data)}")

        print(f"✅ CSV saved: {output_path}")
        return output_path

    def save_plot(self, plot_path: str, plot_name: str, run_name: str = None,
                  plot_type: str = 'general'):
        """
        Save a plot in an organized manner.

        Args:
            plot_path: Current path of the plot
            plot_name: Desired name for the plot
            run_name: Run identifier
            plot_type: Type of plot (general, analysis, comparison, etc.)
        """
        if run_name:
            run_dir = self.get_run_directory(run_name)
            output_dir = run_dir / 'plots' / plot_type
        else:
            output_dir = self.structure['plots'] / plot_type

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / plot_name

        # Move or copy the plot
        if Path(plot_path).exists():
            shutil.move(str(plot_path), str(output_path))
            print(f"📊 Plot saved: {output_path}")
            return output_path
        else:
            print(f"⚠️  Plot not found: {plot_path}")
            return None

    def save_summary(self, summary_data: Dict, run_name: str = None):
        """
        Save a summary of the evaluation run.

        Args:
            summary_data: Dictionary containing summary information
            run_name: Run identifier
        """
        if run_name:
            run_dir = self.get_run_directory(run_name)
            summary_path = run_dir / 'run_summary.json'
        else:
            summary_path = self.structure['summaries'] / f'summary_{self.timestamp}.json'

        # Add metadata
        summary_data.update({
            'timestamp': self.timestamp,
            'run_name': run_name or f"run_{self.timestamp}",
            'generated_by': 'Evaluation Pipeline'
        })

        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)

        print(f"📋 Summary saved: {summary_path}")
        return summary_path

    def cleanup_scattered_files(self, dry_run: bool = False):
        """
        Clean up scattered evaluation files in the project root and other locations.

        Args:
            dry_run: If True, only show what would be moved without actually moving
        """
        print(f"🧹 {'DRY RUN: ' if dry_run else ''}Cleaning up scattered evaluation files...")

        # Define patterns for files that should be organized
        project_root = Path(".")
        patterns_to_move = {
            '*.csv': 'datasets',
            '*_results.csv': 'datasets',
            'dataset_summary*.csv': 'summaries',
            '*prediction*.png': 'plots/predictions',
            '*threshold*.png': 'plots/analysis',
            '*analysis*.png': 'plots/analysis',
            'logits*.png': 'plots/analysis',
            '*.md': 'reports'
        }

        moved_files = []

        for pattern, destination in patterns_to_move.items():
            matching_files = list(project_root.glob(pattern))

            for file_path in matching_files:
                if file_path.is_file():
                    dest_dir = self.structure[destination.split('/')[0]]
                    if '/' in destination:
                        dest_dir = dest_dir / destination.split('/')[1]
                        dest_dir.mkdir(parents=True, exist_ok=True)

                    dest_path = dest_dir / file_path.name

                    if dry_run:
                        print(f"  Would move: {file_path} → {dest_path}")
                    else:
                        try:
                            shutil.move(str(file_path), str(dest_path))
                            print(f"  ✅ Moved: {file_path} → {dest_path}")
                            moved_files.append(str(dest_path))
                        except Exception as e:
                            print(f"  ❌ Failed to move {file_path}: {e}")

        # Also clean up results/outputs directory if it exists
        results_outputs = Path("results/outputs")
        if results_outputs.exists():
            print(f"\n🧹 Organizing results/outputs directory...")
            self._organize_results_outputs(results_outputs, dry_run)

        return moved_files

    def _organize_results_outputs(self, results_dir: Path, dry_run: bool = False):
        """Organize files in the results/outputs directory."""
        csv_files = list(results_dir.glob("*.csv"))

        for csv_file in csv_files:
            # Archive old CSV files to prevent clutter
            archive_dir = self.structure['archive'] / 'csv'
            archive_dir.mkdir(parents=True, exist_ok=True)

            dest_path = archive_dir / csv_file.name

            if dry_run:
                print(f"  Would archive: {csv_file} → {dest_path}")
            else:
                try:
                    shutil.move(str(csv_file), str(dest_path))
                    print(f"  📦 Archived: {csv_file} → {dest_path}")
                except Exception as e:
                    print(f"  ❌ Failed to archive {csv_file}: {e}")

    def generate_index(self):
        """Generate an index of all evaluation results."""
        index_data = {
            'generated': self.timestamp,
            'structure': {},
            'recent_runs': [],
            'summary_statistics': {}
        }

        # Scan the directory structure
        for name, path in self.structure.items():
            if path.exists():
                items = []
                for item in path.iterdir():
                    if item.is_file():
                        items.append({
                            'name': item.name,
                            'type': 'file',
                            'size': item.stat().st_size,
                            'modified': datetime.fromtimestamp(item.stat().st_mtime).isoformat()
                        })
                    elif item.is_dir():
                        items.append({
                            'name': item.name,
                            'type': 'directory',
                            'contents': len(list(item.iterdir()))
                        })

                index_data['structure'][name] = {
                    'path': str(path),
                    'items': items,
                    'total_items': len(items)
                }

        # Get recent runs
        runs_dir = self.structure['runs']
        if runs_dir.exists():
            run_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
            run_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)

            for run_dir in run_dirs[:10]:  # Get 10 most recent
                summary_file = run_dir / 'run_summary.json'
                run_info = {'name': run_dir.name, 'path': str(run_dir)}

                if summary_file.exists():
                    try:
                        with open(summary_file, 'r') as f:
                            summary = json.load(f)
                            run_info.update(summary)
                    except Exception:
                        pass

                index_data['recent_runs'].append(run_info)

        # Save the index
        index_path = self.base_output_dir / 'index.json'
        with open(index_path, 'w') as f:
            json.dump(index_data, f, indent=2)

        print(f"📑 Index generated: {index_path}")
        return index_path

    def create_readme(self):
        """Create a README file explaining the output structure."""
        readme_content = """# Evaluation Results Directory Structure

This directory contains organized evaluation results from the TRANSFORMER drone pipeline.

## Directory Structure

- **`runs/`**: Individual evaluation runs, each in its own timestamped directory
  - `run_YYYYMMDD_HHMMSS/`: Individual run results
    - `csv/`: CSV data files for this run
    - `plots/`: Visualization plots for this run  
    - `logs/`: Log files and debug information
    - `configs/`: Configuration files used for this run
    - `run_summary.json`: Summary of this run's results

- **`summaries/`**: Cross-run summary files and aggregated results

- **`plots/`**: Organized plots by category
  - `analysis/`: Analysis plots (thresholds, correlations, etc.)
  - `predictions/`: Prediction visualization plots
  - `comparisons/`: Comparison plots between models/datasets

- **`reports/`**: Generated reports and documentation
  - Markdown reports, executive summaries, etc.

- **`datasets/`**: Dataset-specific results and analysis

- **`models/`**: Model-specific results and comparisons

- **`archive/`**: Archived old results to prevent clutter

- **`index.json`**: Programmatic index of all results for easy navigation

## Usage

The output manager automatically organizes all evaluation results into this structure.
Use the `EvaluationOutputManager` class to save results in the correct locations.

## Recent Runs

Check `index.json` for a list of the most recent evaluation runs and their summaries.
"""

        readme_path = self.base_output_dir / 'README.md'
        with open(readme_path, 'w') as f:
            f.write(readme_content)

        print(f"📖 README created: {readme_path}")
        return readme_path


def cleanup_project_outputs(dry_run: bool = True):
    """
    Convenience function to clean up scattered outputs in the project.

    Args:
        dry_run: If True, only show what would be done without making changes
    """
    manager = EvaluationOutputManager()

    print("🚀 Starting project output cleanup...")
    print(f"📁 Target directory: {manager.base_output_dir}")

    # Clean up scattered files
    moved_files = manager.cleanup_scattered_files(dry_run=dry_run)

    if not dry_run:
        # Generate index and README
        manager.generate_index()
        manager.create_readme()

        print(f"\n✅ Cleanup complete!")
        print(f"📊 Moved {len(moved_files)} files")
        print(f"📁 Results organized in: {manager.base_output_dir}")
    else:
        print(f"\n👀 DRY RUN COMPLETE - No files were actually moved")
        print(f"🔄 Run with dry_run=False to perform the actual cleanup")

    return manager


if __name__ == "__main__":
    # Run cleanup when script is executed directly
    import argparse

    parser = argparse.ArgumentParser(description="Clean up evaluation outputs")
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without making changes')
    parser.add_argument('--output-dir', default='./eval_results',
                       help='Base directory for organized outputs')

    args = parser.parse_args()

    manager = EvaluationOutputManager(args.output_dir)
    cleanup_project_outputs(dry_run=args.dry_run)
