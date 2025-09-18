#!/usr/bin/env python3
"""
Clip-based evaluation using the evaluation strategy factory with centralized configuration.
"""

import warnings
from evaluation_strategy_factory import run_evaluation
from TRANSFORMER.config.config import get_config, print_config

# Try to import advanced analysis
try:
    from advanced_analysis import generate_comprehensive_analysis_report
    ADVANCED_ANALYSIS_AVAILABLE = True
except ImportError:
    ADVANCED_ANALYSIS_AVAILABLE = False
    print("Note: Advanced analysis not available. Install dependencies for enhanced insights.")

# Suppress sklearn warnings about undefined metrics
warnings.filterwarnings("ignore", message=".*(Precision|F-score|Recall) is ill-defined.*")

def main():
    """Run clip-based evaluation using the strategy factory with centralized config."""
    # Load configuration from environment or defaults
    config = get_config()

    print("Starting clip-based evaluation...")
    print_config(config)

    # Run evaluation with clip-based strategy using configuration
    results = run_evaluation(
        strategy_type="clip",
        model_path=config.model.model_path,
        clip_duration=config.clip.clip_duration,
        batch_size=config.clip.batch_size,
        max_clips_per_dataset=config.clip.max_clips_per_dataset,
        output_dir=config.output.base_output_dir
    )

    # Add comprehensive analysis
    if ADVANCED_ANALYSIS_AVAILABLE and results:
        print("\n" + "="*60)
        print("RUNNING COMPREHENSIVE ANALYSIS")
        print("="*60)

        analysis_results = generate_comprehensive_analysis_report(
            results,
            f"{config.output.base_output_dir}/comprehensive_analysis"
        )

        print("\nComprehensive analysis completed!")
        print("Key insights generated:")
        if 'optimal_thresholds' in analysis_results:
            opt_f1 = analysis_results['optimal_thresholds'].get('f1', 0.5)
            print(f"  - Optimal F1 threshold: {opt_f1:.3f} (current: 0.5)")
        print("  - Dataset difficulty ranking")
        print("  - Error pattern analysis")
        print("  - Performance consistency assessment")

    print(f"\nClip-based evaluation completed successfully!")
    print(f"Processed {len(results)} datasets")
    print("\nResults summary:")
    for dataset_name, metrics in results.items():
        print(f"  {dataset_name}: {metrics.get('num_files', 0)} files, {metrics.get('num_clips', 0)} clips")

if __name__ == "__main__":
    main()
