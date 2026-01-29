#!/usr/bin/env python3
"""
Command-line interface for transformer model evaluation.
This provides a unified entry point for all evaluation strategies with argument parsing.
"""
import argparse
import sys
import warnings
import os
import winsound
from pathlib import Path
import time
# Add the current src directory to the Python path

sys.path.append(os.path.dirname(__file__))

from evaluation.evaluation_strategy_factory import run_evaluation, EvaluationStrategyFactory
from utils.aggregation_utils import analyze_logits_statistics
from config.config import EvaluationConfig, ModelConfigForEval, ClipEvaluationConfig, FileEvaluationConfig, OutputConfig, print_config

# Try to import advanced analysis and unified pipeline (optional)
try:
    from evaluation.advanced_analysis import generate_comprehensive_analysis_report
    ADVANCED_ANALYSIS_AVAILABLE = True
except ImportError:
    ADVANCED_ANALYSIS_AVAILABLE = False

try:
    from evaluation.evaluation_utils import create_unified_pipeline_runner
    UNIFIED_PIPELINE_AVAILABLE = True
except ImportError:
    UNIFIED_PIPELINE_AVAILABLE = False

try:
    from evaluation.multi_model_evaluation import MultiModelEvaluator, ModelConfig as MultiModelConfig, create_multi_model_config_from_checkpoints
    MULTI_MODEL_AVAILABLE = True
except ImportError:
    MULTI_MODEL_AVAILABLE = False

try:
    from evaluation.resilient_evaluation import create_resilient_evaluation_strategy
    RESILIENT_EVALUATION_AVAILABLE = True
except ImportError:
    RESILIENT_EVALUATION_AVAILABLE = False


def create_parser():
    """Create argument parser for CLI."""
    parser = argparse.ArgumentParser(
        description="Transformer Model Evaluation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run clip-based evaluation with defaults
  python cli.py clip
  
  # Run whole-file evaluation with custom model
  python cli.py file --model-path /path/to/model
  
  # Run unified pipeline with comprehensive analysis
  python cli.py unified --comprehensive-analysis
  
  # Run multi-model evaluation
  python cli.py multi-model --checkpoint-dir ./checkpoints
  
  # Run resilient evaluation with error recovery
  python cli.py file --resilient --max-retries 5
  
  # List available strategies and features
  python cli.py --list-strategies
        """
    )

    # Global options
    parser.add_argument(
        '--list-strategies',
        action='store_true',
        help='List available evaluation strategies and exit'
    )

    parser.add_argument(
        '--config-only',
        action='store_true',
        help='Print configuration and exit without running evaluation'
    )

    # Strategy selection - expanded to include new pipeline options
    parser.add_argument(
        'strategy',
        nargs='?',
        choices=['clip', 'file', 'unified', 'multi-model'],
        help='Evaluation strategy to use'
    )

    # Model configuration
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument(
        '--model-path',
        type=str,
        help='Path to the transformer model'
    )
    model_group.add_argument(
        '--sample-rate',
        type=int,
        default=16000,
        help='Audio sample rate (default: 16000)'
    )

    # Output configuration
    output_group = parser.add_argument_group('Output Configuration')
    output_group.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for results'
    )
    output_group.add_argument(
        '--plot-dpi',
        type=int,
        default=300,
        help='DPI for saved plots (default: 300)'
    )
    output_group.add_argument(
        '--no-plots',
        action='store_true',
        help='Disable plot generation'
    )
    output_group.add_argument(
        '--no-csvs',
        action='store_true',
        help='Disable CSV output'
    )
    output_group.add_argument(
        '--overwrite',
        action='store_true',
        help='Allow overwriting existing files (creates backups)'
    )
    output_group.add_argument(
        '--auto-version',
        action='store_true',
        help='Automatically create versioned files instead of prompting'
    )
    output_group.add_argument(
        '--no-backup',
        action='store_true',
        help='Disable backup creation when overwriting files'
    )
    output_group.add_argument(
        '--force-output-dir',
        action='store_true',
        help='Force output to specified directory, bypassing file protection'
    )

    # Clip-based evaluation options
    clip_group = parser.add_argument_group('Clip-Based Evaluation Options')
    clip_group.add_argument(
        '--clip-duration',
        type=float,
        help='Duration of audio clips in seconds'
    )
    clip_group.add_argument(
        '--clip-batch-size',
        type=int,
        help='Batch size for clip processing'
    )
    clip_group.add_argument(
        '--max-clips',
        type=int,
        help='Maximum clips per dataset'
    )
    clip_group.add_argument(
        '--clip-threshold',
        type=float,
        help='Classification threshold for clips'
    )
    clip_group.add_argument(
        '--aggregation-threshold',
        type=float,
        help='Threshold for clip aggregation'
    )

    # File-based evaluation options
    file_group = parser.add_argument_group('File-Based Evaluation Options')
    file_group.add_argument(
        '--file-batch-size',
        type=int,
        help='Batch size for file processing'
    )
    file_group.add_argument(
        '--file-threshold',
        type=float,
        help='Classification threshold for files'
    )
    file_group.add_argument(
        '--large-file-threshold',
        type=float,
        help='Threshold for large file detection (seconds)'
    )
    file_group.add_argument(
        '--very-large-threshold',
        type=float,
        help='Threshold for very large file detection (seconds)'
    )
    file_group.add_argument(
        '--max-file-length',
        type=float,
        help='Maximum file length before splitting (seconds)'
    )
    file_group.add_argument(
        '--max-file-size',
        type=int,
        help='Maximum file size before splitting (MB)'
    )
    file_group.add_argument(
        '--overlap-seconds',
        type=float,
        help='Overlap between chunks when splitting files'
    )

    # Enhanced analysis options
    analysis_group = parser.add_argument_group('Enhanced Analysis Options')
    if ADVANCED_ANALYSIS_AVAILABLE:
        analysis_group.add_argument(
            '--comprehensive-analysis',
            action='store_true',
            help='Run comprehensive analysis including difficulty assessment, error patterns, etc.'
        )

    if UNIFIED_PIPELINE_AVAILABLE:
        analysis_group.add_argument(
            '--unified-pipeline',
            action='store_true',
            help='Use unified pipeline runner that combines all analysis approaches'
        )

    analysis_group.add_argument(
        '--logits-analysis',
        action='store_true',
        help='Perform logits analysis (for file-based evaluation)'
    )
    analysis_group.add_argument(
        '--enhanced-plots',
        action='store_true',
        help='Generate enhanced plots including log-scale probability distributions'
    )
    analysis_group.add_argument(
        '--plot-suite',
        action='store_true',
        help='Generate comprehensive plot suite with all visualization types'
    )
    analysis_group.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    #Calibration options
    calibration_group = parser.add_argument_group('Calibration Options')
    calibration_group.add_argument(
        '--calibrate',
        action='store_true',
        help='Run calibration on the calibration set first to find the optimal threshold'
    )
    calibration_group.add_argument(
        '--calibration-key',
        type=str,
        default='Calibration',
        help='Dataset key in ENHANCED_DATASETS used for calibration (default: Calibration)'
    )
    calibration_group.add_argument(
        '--threshold-file',
        type=str,
        help='Path to threshold JSON file for load/save (default: <model_path>/threshold.json)'
    )
    calibration_group.add_argument(
        '--force-recalibrate',
        action='store_true',
        help='Recompute threshold even if a saved threshold file exists'
    )
    calibration_group.add_argument(
        '--calibration-fraction',
        type=float,
        default=1.0,
        help='Fraction of the calibration dataset to use for calibration (0.0-1.0). Use <1.0 to sample a small subset.'
    )
    # Multi-model evaluation options
    if MULTI_MODEL_AVAILABLE:
        multi_group = parser.add_argument_group('Multi-Model Evaluation Options')
        multi_group.add_argument(
            '--checkpoint-dir',
            type=str,
            help='Directory containing model checkpoints for multi-model evaluation'
        )
        multi_group.add_argument(
            '--model-configs',
            type=str,
            nargs='+',
            help='Paths to individual model configurations (name:path:weight format)'
        )
        multi_group.add_argument(
            '--base-strategy',
            type=str,
            choices=['file', 'clip'],
            default='file',
            help='Base evaluation strategy for each model in multi-model mode (default: file)'
        )
        multi_group.add_argument(
            '--parallel-models',
            action='store_true',
            help='Run models in parallel for multi-model evaluation'
        )
        multi_group.add_argument(
            '--max-workers',
            type=int,
            default=2,
            help='Maximum number of parallel workers for multi-model evaluation'
        )

    # Resilient evaluation options
    if RESILIENT_EVALUATION_AVAILABLE:
        resilient_group = parser.add_argument_group('Resilient Evaluation Options')
        resilient_group.add_argument(
            '--resilient',
            action='store_true',
            help='Enable resilient evaluation with automatic error recovery'
        )
        resilient_group.add_argument(
            '--max-retries',
            type=int,
            default=3,
            help='Maximum number of retries for failed operations'
        )
        resilient_group.add_argument(
            '--retry-delay',
            type=float,
            default=2.0,
            help='Delay between retries in seconds'
        )

    return parser


def run_unified_pipeline(args, config):
    """Run the unified analysis pipeline."""
    if not UNIFIED_PIPELINE_AVAILABLE:
        print("Unified pipeline not available. Check imports.")
        return False

    print("Starting Unified Analysis Pipeline...")

    # Get the unified pipeline runner
    pipeline_runner = create_unified_pipeline_runner()

    # Prepare pipeline arguments
    pipeline_kwargs = {}

    # Add strategy-specific parameters based on strategy type
    strategy_type = getattr(args, 'unified_strategy', 'file')
    if strategy_type == "clip":
        pipeline_kwargs.update({
            'clip_duration': config.clip.clip_duration,
            'batch_size': config.clip.batch_size,
            'max_clips_per_dataset': config.clip.max_clips_per_dataset,
        })
    else:  # file
        pipeline_kwargs.update({
            'batch_size': config.file.batch_size,
            'large_file_threshold': config.file.large_file_threshold,
            'very_large_threshold': config.file.very_large_threshold,
            'max_file_length': config.file.max_file_length,
            'max_file_size_mb': config.file.max_file_size_mb,
        })

    # Run the unified pipeline
    results = pipeline_runner(
        strategy_type=strategy_type,
        model_path=config.model.model_path,
        output_dir=config.output.base_output_dir,
        **pipeline_kwargs
    )

    return results is not None


def run_multi_model_evaluation(args, config):
    """Run multi-model evaluation."""
    if not MULTI_MODEL_AVAILABLE:
        print("❌ Multi-model evaluation not available. Check imports.")
        return False

    print("🚀 Starting Multi-Model Evaluation...")

    # Get base strategy type (file or clip)
    base_strategy = getattr(args, 'base_strategy', 'file')
    print(f"📋 Using base strategy: {base_strategy}")

    # Create model configurations
    models = []

    if args.checkpoint_dir:
        # Auto-discover from checkpoint directory
        models = create_multi_model_config_from_checkpoints(
            args.checkpoint_dir,
            strategy_type=base_strategy
        )
        print(f"📁 Found {len(models)} models in checkpoint directory")

    elif args.model_configs:
        # Parse manual model configurations
        for config_str in args.model_configs:
            parts = config_str.split(':')
            if len(parts) >= 2:
                name = parts[0]
                path = parts[1]
                weight = float(parts[2]) if len(parts) > 2 else 1.0

                models.append(MultiModelConfig(
                    name=name,
                    model_path=path,
                    weight=weight,
                    strategy_type=base_strategy
                ))

    if not models:
        print("❌ No models specified for multi-model evaluation")
        print("Use --checkpoint-dir or --model-configs to specify models")
        return False

    # Prepare calibration options
    calibration_options = {
        'calibrate': getattr(args, 'calibrate', False),
        'calibration_key': getattr(args, 'calibration_key', 'Calibration'),
        'force_recalibrate': getattr(args, 'force_recalibrate', False),
        'calibration_fraction': getattr(args, 'calibration_fraction', 1.0),
    }

    if calibration_options['calibrate']:
        print(f"🎯 Calibration enabled (key: {calibration_options['calibration_key']})")

    # Prepare strategy parameters based on base strategy
    if base_strategy == "clip":
        strategy_params = {
            'clip_duration': config.clip.clip_duration,
            'batch_size': config.clip.batch_size,
            'max_clips_per_dataset': config.clip.max_clips_per_dataset,
        }
    else:  # file
        strategy_params = {
            'batch_size': config.file.batch_size,
            'large_file_threshold': config.file.large_file_threshold,
            'very_large_threshold': config.file.very_large_threshold,
            'max_file_length': config.file.max_file_length,
            'max_file_size_mb': config.file.max_file_size_mb,
        }

    # Create and run multi-model evaluator
    evaluator = MultiModelEvaluator(
        models,
        output_base_dir=config.output.base_output_dir,
        strategy_params=strategy_params,
        calibration_options=calibration_options
    )

    # Run individual model evaluations
    individual_results = evaluator.evaluate_individual_models(
        parallel=args.parallel_models,
        max_workers=args.max_workers
    )

    # Run ensemble evaluations
    ensemble_results = evaluator.evaluate_ensembles()

    # Generate comparison report
    report = evaluator.generate_comparison_report(
        f"{config.output.base_output_dir}/multi_model_comparison.json"
    )

    print(f"✅ Multi-model evaluation complete!")
    print(f"📊 Results saved to: {config.output.base_output_dir}")

    return True


def create_resilient_strategy(args, config, strategy_type):
    """Create a resilient evaluation strategy."""
    if not RESILIENT_EVALUATION_AVAILABLE:
        print("⚠️  Resilient evaluation not available - using standard strategy")
        return None

    resilience_config = {
        'max_retries': args.max_retries,
        'retry_delay': args.retry_delay,
        'memory_cleanup_threshold': 0.8
    }

    # Import the appropriate base strategy
    from evaluation.evaluation_strategy_factory import ClipBasedEvaluationStrategy, WholeFileEvaluationStrategy

    if strategy_type == "clip":
        base_strategy_class = ClipBasedEvaluationStrategy
    else:
        base_strategy_class = WholeFileEvaluationStrategy

    # Create resilient version
    ResilientStrategy = create_resilient_evaluation_strategy(
        base_strategy_class,
        resilience_config
    )

    print(f"🛡️  Using resilient {strategy_type}-based evaluation")
    return ResilientStrategy


def main():
    """Main CLI entry point."""
    start = time.process_time()
    parser = create_parser()
    args = parser.parse_args()

    # Handle special cases
    if args.list_strategies:
        print("Available evaluation strategies:")
        for strategy in EvaluationStrategyFactory.get_available_strategies():
            print(f"  - {strategy}")

        if UNIFIED_PIPELINE_AVAILABLE:
            print("  - unified (comprehensive pipeline)")
        if MULTI_MODEL_AVAILABLE:
            print("  - multi-model (ensemble evaluation)")

        print("\nAvailable features:")
        if ADVANCED_ANALYSIS_AVAILABLE:
            print("Advanced analysis")
        if RESILIENT_EVALUATION_AVAILABLE:
            print("Resilient evaluation")
        if UNIFIED_PIPELINE_AVAILABLE:
            print("Unified pipeline")
        if MULTI_MODEL_AVAILABLE:
            print("Multi-model evaluation")
        return

    if not args.strategy:
        parser.error("Strategy is required unless using --list-strategies")

    # Create configuration from arguments
    config = create_config_from_args(args)

    if args.config_only:
        print_config(config)
        return

    # Suppress warnings unless verbose
    if not args.verbose:
        warnings.filterwarnings("ignore", message=".*(Precision|F-score|Recall) is ill-defined.*")

    # Validate model path exists (except for multi-model which handles this internally)
    if args.strategy != 'multi-model' and not Path(config.model.model_path).exists():
        print(f"Error: Model path does not exist: {config.model.model_path}")
        sys.exit(1)

    # Check for existing files and handle file protection
    if args.force_output_dir:
        # Bypass all file protection - use the directory as-is
        print(f"Force output mode: Using directory as-is: {config.output.base_output_dir}")
    elif not args.overwrite and not args.auto_version:
        # Simple auto-versioning: if directory exists, create numbered version
        original_dir = config.output.base_output_dir
        counter = 1
        while Path(config.output.base_output_dir).exists():
            config.output.base_output_dir = f"{original_dir}_{counter}"
            counter += 1

        if config.output.base_output_dir != original_dir:
            print(f"Directory exists, using: {config.output.base_output_dir}")
        else:
            print(f"Using directory: {config.output.base_output_dir}")
    elif args.auto_version:
        # Same simple auto-versioning
        original_dir = config.output.base_output_dir
        counter = 1
        while Path(config.output.base_output_dir).exists():
            config.output.base_output_dir = f"{original_dir}_{counter}"
            counter += 1

        if config.output.base_output_dir != original_dir:
            print(f"Auto-versioning: Using {config.output.base_output_dir}")
    else:
        # Overwrite mode - just warn
        if Path(config.output.base_output_dir).exists():
            print(f"Overwrite mode: Will overwrite existing files in {config.output.base_output_dir}")
        else:
            print(f"Using directory: {config.output.base_output_dir}")

    print(f"Starting {args.strategy} evaluation...")
    if args.verbose:
        print_config(config)

    try:
        success = False

        if args.strategy == "unified":
            success = run_unified_pipeline(args, config)

        elif args.strategy == "multi-model":
            success = run_multi_model_evaluation(args, config)

        else:
            # Standard single-strategy evaluation
            strategy_type = args.strategy

            # Check if resilient evaluation is requested
            if getattr(args, 'resilient', False):
                ResilientStrategy = create_resilient_strategy(args, config, strategy_type)
                if ResilientStrategy:
                    # Use resilient strategy
                    if strategy_type == "clip":
                        strategy = ResilientStrategy(
                            model_path=config.model.model_path,
                            clip_duration=config.clip.clip_duration,
                            batch_size=config.clip.batch_size,
                            max_clips_per_dataset=config.clip.max_clips_per_dataset,
                            output_dir=config.output.base_output_dir
                        )
                    else:  # file
                        strategy = ResilientStrategy(
                            model_path=config.model.model_path,
                            batch_size=config.file.batch_size,
                            large_file_threshold=config.file.large_file_threshold,
                            very_large_threshold=config.file.very_large_threshold,
                            max_file_length=config.file.max_file_length,
                            max_file_size_mb=config.file.max_file_size_mb,
                            output_dir=config.output.base_output_dir
                        )

                    # Setup and run
                    strategy.setup()
                    results = strategy.evaluate_all_datasets()
                    strategy.generate_outputs(results)

                    # Show resilience report
                    resilience_report = strategy.get_resilience_report()
                    print(f"\n Resilience Report:")
                    print(f"   Total errors handled: {resilience_report['total_errors']}")
                    print(f"   Successful recoveries: {resilience_report['recovered_errors']}")
                    print(f"   Recovery rate: {resilience_report['recovery_rate_percent']:.1f}%")

                    success = results is not None
                else:
                    # Fall back to standard evaluation
                    results = run_standard_evaluation(args, config, strategy_type)
                    success = results is not None
            else:
                # Standard evaluation
                results = run_standard_evaluation(args, config, strategy_type)
                success = results is not None
        winsound.MessageBeep()
        end = time.process_time()

        print("Time in hours: ", (end - start)/3600)
        if success:
            print(f"\n{args.strategy.title()} evaluation completed successfully!")
        else:
            print(f"\n{args.strategy.title()} evaluation failed!")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error during evaluation: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def run_standard_evaluation(args, config, strategy_type):
    """Run standard evaluation using the strategy factory."""
    # Prepare strategy-specific parameters
    if strategy_type == "clip":
        strategy_params = {
            'clip_duration': config.clip.clip_duration,
            'batch_size': config.clip.batch_size,
            'max_clips_per_dataset': config.clip.max_clips_per_dataset,
        }
    else:  # file
        strategy_params = {
            'batch_size': config.file.batch_size,
            'large_file_threshold': config.file.large_file_threshold,
            'very_large_threshold': config.file.very_large_threshold,
            'max_file_length': config.file.max_file_length,
            'max_file_size_mb': config.file.max_file_size_mb,
        }

    # Create and set up strategy explicitly so we can calibrate before evaluation
    strategy = EvaluationStrategyFactory.create_strategy(
        strategy_type,
        config.model.model_path,
        output_dir=config.output.base_output_dir,
        **strategy_params
    )
    strategy.setup()
    # Derive threshold file path
    default_threshold_path = os.path.join(config.model.model_path, 'threshold.json')
    threshold_file = args.threshold_file if args.threshold_file else default_threshold_path

    # Calibration flow
    if getattr(args, 'calibrate', False):
        # If not forcing recalibration, try loading an existing threshold
        if not getattr(args, 'force_recalibrate', False):
            loaded = strategy.load_saved_threshold(threshold_file)
            if loaded:
                print(f"✓ Using saved calibrated threshold: {strategy.threshold:.6f}")
            else:
                # No saved threshold; perform calibration
                strategy.calibrate_threshold(
                    calibration_key=getattr(args, 'calibration_key', 'Calibration'),
                    threshold_file=threshold_file,
                    calibration_fraction=getattr(args, 'calibration_fraction', 1.0)
                )
        else:
            # Force recalibration
            strategy.calibrate_threshold(
                calibration_key=getattr(args, 'calibration_key', 'Calibration'),
                threshold_file=threshold_file,
                calibration_fraction=getattr(args, 'calibration_fraction', 1.0)
            )
    else:
        # Not calibrating explicitly; try to load a saved threshold if present
        _ = strategy.load_saved_threshold(threshold_file)

    results = strategy.evaluate_all_datasets()
    strategy.generate_outputs(results)

    ## Run evaluation
    #results = run_evaluation(
    #    strategy_type=strategy_type,
    #    model_path=config.model.model_path,
    #    output_dir=config.output.base_output_dir,
    #    **strategy_params
    #)

    # Enhanced analysis options
    if results:
        if ADVANCED_ANALYSIS_AVAILABLE and getattr(args, 'comprehensive_analysis', False):
            print("\n" + "="*60)
            print("RUNNING COMPREHENSIVE ANALYSIS")
            print("="*60)

            analysis_results = generate_comprehensive_analysis_report(
                results,
                f"{config.output.base_output_dir}/comprehensive_analysis"
            )

            print("\nComprehensive analysis completed!")
            print("Additional insights generated:")
            print("  - Optimal threshold recommendations")
            print("  - Class imbalance impact analysis")
            print("  - Performance consistency assessment")
            print("  - Dataset difficulty ranking")
            print("  - Error pattern analysis")
            print("  - Executive summary report")

        # Perform additional analysis if requested
        if args.logits_analysis and strategy_type == "file":
            print("\nPerforming logits analysis...")
            analyze_logits_statistics(results, "logits_analysis.png")

        print(f"\n{strategy_type.title()}-based evaluation completed successfully!")
        print(f"Processed {len(results)} datasets")

        if args.verbose:
            print("\nResults summary:")
            for dataset_name, metrics in results.items():
                if strategy_type == "clip":
                    print(f"  {dataset_name}: {metrics.get('num_files', 0)} files, {metrics.get('num_clips', 0)} clips")
                else:
                    print(f"  {dataset_name}: {metrics.get('num_files', 0)} files across {metrics.get('num_splits', 0)} splits")

        print(f"\nResults saved to: {config.output.base_output_dir}")

    return results

def create_config_from_args(args):
    """Create configuration from command line arguments."""
    # Start with default/environment config
    config = EvaluationConfig.from_env()

    # Override with command line arguments
    if args.model_path:
        config.model.model_path = args.model_path
    if args.sample_rate != 16000:  # Only override if not default
        config.model.sample_rate = args.sample_rate

    if args.output_dir:
        config.output.base_output_dir = args.output_dir
    if args.plot_dpi != 300:
        config.output.plot_dpi = args.plot_dpi
    if args.no_plots:
        config.output.save_plots = False
    if args.no_csvs:
        config.output.save_individual_csvs = False
        config.output.save_combined_csvs = False

    # Clip-specific overrides
    if args.clip_duration:
        config.clip.clip_duration = args.clip_duration
    if args.clip_batch_size:
        config.clip.batch_size = args.clip_batch_size
    if args.max_clips:
        config.clip.max_clips_per_dataset = args.max_clips
    if args.clip_threshold:
        config.clip.threshold = args.clip_threshold
    if args.aggregation_threshold:
        config.clip.aggregation_threshold = args.aggregation_threshold

    # File-specific overrides
    if args.file_batch_size:
        config.file.batch_size = args.file_batch_size
    if args.file_threshold:
        config.file.threshold = args.file_threshold
    if args.large_file_threshold:
        config.file.large_file_threshold = args.large_file_threshold
    if args.very_large_threshold:
        config.file.very_large_threshold = args.very_large_threshold
    if args.max_file_length:
        config.file.max_file_length = args.max_file_length
    if args.max_file_size:
        config.file.max_file_size_mb = args.max_file_size
    if args.overlap_seconds:
        config.file.overlap_seconds = args.overlap_seconds

    return config


if __name__ == "__main__":
    main()
