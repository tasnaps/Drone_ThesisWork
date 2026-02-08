import os
import time
from pathlib import Path
from typing import Dict, Any
from TRANSFORMER.evaluation.evaluation_strategy_factory import EvaluationStrategyFactory
from TRANSFORMER.evaluation.evaluation_common import safe_cuda_cleanup
from TRANSFORMER.evaluation.evaluation_utils import create_unified_pipeline_runner

# Disable interactive prompts
os.environ['DRONE_EVALUATION_NON_INTERACTIVE'] = '1'

DEFAULT_CALIBRATION = {
    'calibrate': True,
    'calibration_key': 'CalibrationDataset',
    'force_recalibrate': True,
    'calibration_fraction': 1.0,
    'threshold_file': None,  # if None, defaults to `<model_path>/threshold.json`
}

# Define all evaluation runs
EVALUATION_RUNS = [
    # All-Augments
    {
        'name': 'All_Augments_Epoch-10',
        'model_path': r'C:\Users\XXX\Desktop\Experiment\Models\Wav2Vec2Models\Augmented\wav2vec2-Final-20Epoch-Alemadi-ALL_Augments-Default\checkpoint-2930',
        'output_dir': r'C:\Users\XXX\Desktop\Experiment\Evaluation\wav2vec2Augments\All_Augments\Epoch-10',
        'strategy_type': 'file',  # Explicitly set strategy type
    },
    {
        'name': 'All_Augments_Epoch-15',
        'model_path': r'C:\Users\XXX\Desktop\Experiment\Models\Wav2Vec2Models\Augmented\wav2vec2-Final-20Epoch-Alemadi-ALL_Augments-Default\checkpoint-4395',
        'output_dir': r'C:\Users\XXX\Desktop\Experiment\Evaluation\wav2vec2Augments\All_Augments\Epoch-15',
        'strategy_type': 'file',
    },
    {
        'name': 'All_Augments_Epoch-20',
        'model_path': r'C:\Users\XXX\Desktop\Experiment\Models\Wav2Vec2Models\Augmented\wav2vec2-Final-20Epoch-Alemadi-ALL_Augments-Default\checkpoint-5860',
        'output_dir': r'C:\Users\XXX\Desktop\Experiment\Evaluation\wav2vec2Augments\All_Augments\Epoch-20',
        'strategy_type': 'file',
    },

    # BandPass only
    {
        'name': 'BandPass_Epoch-1',
        'model_path': r'C:\Users\XXX\Desktop\Experiment\Models\Wav2Vec2Models\Augmented\wav2vec2-Final-20Epoch-Alemadi-BandPass-Default\1 Epoch\wav2vec2-Final-20Epoch-Alemadi-BandPass-Default',
        'output_dir': r'C:\Users\XXX\Desktop\Experiment\Evaluation\wav2vec2Augments\BandPass\Epoch-1',
        'strategy_type': 'file',
    },
    {
        'name': 'BandPass_Epoch-5',
        'model_path': r'C:\Users\XXX\Desktop\Experiment\Models\Wav2Vec2Models\Augmented\wav2vec2-Final-20Epoch-Alemadi-BandPass-Default\5 Epoch',
        'output_dir': r'C:\Users\XXX\Desktop\Experiment\Evaluation\wav2vec2Augments\BandPass\Epoch-5',
        'strategy_type': 'file',
    },
    {
        'name': 'BandPass_Epoch-10',
        'model_path': r'C:\Users\XXX\Desktop\Experiment\Models\Wav2Vec2Models\Augmented\wav2vec2-Final-20Epoch-Alemadi-BandPass-Default\10Epoch',
        'output_dir': r'C:\Users\XXX\Desktop\Experiment\Evaluation\wav2vec2Augments\BandPass\Epoch-10',
        'strategy_type': 'file',
    },
    {
        'name': 'BandPass_Epoch-15',
        'model_path': r'C:\Users\XXX\Desktop\Experiment\Models\Wav2Vec2Models\Augmented\wav2vec2-Final-20Epoch-Alemadi-BandPass-Default\15Epoch',
        'output_dir': r'C:\Users\XXX\Desktop\Experiment\Evaluation\wav2vec2Augments\BandPass\Epoch-15',
        'strategy_type': 'file',
    },
    {
        'name': 'BandPass_Epoch-20',
        'model_path': r'C:\Users\XXX\Desktop\Experiment\Models\Wav2Vec2Models\Augmented\wav2vec2-Final-20Epoch-Alemadi-BandPass-Default\20Epoch',
        'output_dir': r'C:\Users\XXX\Desktop\Experiment\Evaluation\wav2vec2Augments\BandPass\Epoch-20',
        'strategy_type': 'file',
    },

    # Clipped Only
    {
        'name': 'Clipped_Epoch-1',
        'model_path': r'C:\Users\XXX\Desktop\Experiment\Models\Wav2Vec2Models\Augmented\wav2vec2-Final-20Epoch-Alemadi-Clipped-Default\1Epoch',
        'output_dir': r'C:\Users\XXX\Desktop\Experiment\Evaluation\wav2vec2Augments\Clipped\Epoch-1',
        'strategy_type': 'file',
    },
    {
        'name': 'Clipped_Epoch-5',
        'model_path': r'C:\Users\XXX\Desktop\Experiment\Models\Wav2Vec2Models\Augmented\wav2vec2-Final-20Epoch-Alemadi-Clipped-Default\checkpoint-1465',
        'output_dir': r'C:\Users\XXX\Desktop\Experiment\Evaluation\wav2vec2Augments\Clipped\Epoch-5',
        'strategy_type': 'file',
    },
    {
        'name': 'Clipped_Epoch-10',
        'model_path': r'C:\Users\XXX\Desktop\Experiment\Models\Wav2Vec2Models\Augmented\wav2vec2-Final-20Epoch-Alemadi-Clipped-Default\checkpoint-2930',
        'output_dir': r'C:\Users\XXX\Desktop\Experiment\Evaluation\wav2vec2Augments\Clipped\Epoch-10',
        'strategy_type': 'file',
    },
    {
        'name': 'Clipped_Epoch-15',
        'model_path': r'C:\Users\XXX\Desktop\Experiment\Models\Wav2Vec2Models\Augmented\wav2vec2-Final-20Epoch-Alemadi-Clipped-Default\checkpoint-4395',
        'output_dir': r'C:\Users\XXX\Desktop\Experiment\Evaluation\wav2vec2Augments\Clipped\Epoch-15',
        'strategy_type': 'file',
    },
    {
        'name': 'Clipped_Epoch-20',
        'model_path': r'C:\Users\XXX\Desktop\Experiment\Models\Wav2Vec2Models\Augmented\wav2vec2-Final-20Epoch-Alemadi-Clipped-Default\checkpoint-5860',
        'output_dir': r'C:\Users\XXX\Desktop\Experiment\Evaluation\wav2vec2Augments\Clipped\Epoch-20',
        'strategy_type': 'file',
    },

    # Gaussian Noise Only
    {
        'name': 'GaussianNoise_Epoch-1',
        'model_path': r'C:\Users\XXX\Desktop\Experiment\Models\Wav2Vec2Models\Augmented\wav2vec2-Final-20Epoch-Alemadi-GaussianNoise-Default\wav2vec2-Final-1Epoch-Alemadi-GaussianNoise-Default',
        'output_dir': r'C:\Users\XXX\Desktop\Experiment\Evaluation\wav2vec2Augments\GaussianNoise\Epoch-1',
        'strategy_type': 'file',
    },
    {
        'name': 'GaussianNoise_Epoch-5',
        'model_path': r'C:\Users\XXX\Desktop\Experiment\Models\Wav2Vec2Models\Augmented\wav2vec2-Final-20Epoch-Alemadi-GaussianNoise-Default\checkpoint-1465',
        'output_dir': r'C:\Users\XXX\Desktop\Experiment\Evaluation\wav2vec2Augments\GaussianNoise\Epoch-5',
        'strategy_type': 'file',
    },
    {
        'name': 'GaussianNoise_Epoch-10',
        'model_path': r'C:\Users\XXX\Desktop\Experiment\Models\Wav2Vec2Models\Augmented\wav2vec2-Final-20Epoch-Alemadi-GaussianNoise-Default\checkpoint-2930',
        'output_dir': r'C:\Users\XXX\Desktop\Experiment\Evaluation\wav2vec2Augments\GaussianNoise\Epoch-10',
        'strategy_type': 'file',
    },
    {
        'name': 'GaussianNoise_Epoch-15',
        'model_path': r'C:\Users\XXX\Desktop\Experiment\Models\Wav2Vec2Models\Augmented\wav2vec2-Final-20Epoch-Alemadi-GaussianNoise-Default\checkpoint-4395',
        'output_dir': r'C:\Users\XXX\Desktop\Experiment\Evaluation\wav2vec2Augments\GaussianNoise\Epoch-15',
        'strategy_type': 'file',
    },
    {
        'name': 'GaussianNoise_Epoch-20',
        'model_path': r'C:\Users\XXX\Desktop\Experiment\Models\Wav2Vec2Models\Augmented\wav2vec2-Final-20Epoch-Alemadi-GaussianNoise-Default\checkpoint-5860',
        'output_dir': r'C:\Users\XXX\Desktop\Experiment\Evaluation\wav2vec2Augments\GaussianNoise\Epoch-20',
        'strategy_type': 'file',
    },
]


def _validate_model_path(model_path: str) -> bool:
    """Validate model path exists like CLI does."""
    if not Path(model_path).exists():
        print(f"❌ Model path does not exist: {model_path}")
        return False
    return True


def _prepare_output_dir(output_dir: str, overwrite: bool = False, auto_version: bool = True,
                        force_output_dir: bool = False) -> str:
    """Mirror CLI output directory handling."""
    if force_output_dir:
        print(f"Force output mode: Using directory as-is: {output_dir}")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        return output_dir

    if not overwrite and not auto_version:
        # Simple auto-versioning: if directory exists, create numbered version
        original_dir = output_dir
        counter = 1
        while Path(output_dir).exists():
            output_dir = f"{original_dir}_{counter}"
            counter += 1

        if output_dir != original_dir:
            print(f"Directory exists, using: {output_dir}")
        else:
            print(f"Using directory: {output_dir}")
    elif auto_version:
        # Same simple auto-versioning
        original_dir = output_dir
        counter = 1
        while Path(output_dir).exists():
            output_dir = f"{original_dir}_{counter}"
            counter += 1

        if output_dir != original_dir:
            print(f"Auto-versioning: Using {output_dir}")
    else:
        # Overwrite mode - just warn
        if Path(output_dir).exists():
            print(f"Overwrite mode: Will overwrite existing files in {output_dir}")
        else:
            print(f"Using directory: {output_dir}")

    # Create the directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return output_dir


def _build_strategy_params(run: Dict[str, Any]) -> Dict[str, Any]:
    """Map run dict to strategy kwargs exactly like CLI."""
    strategy_type = run.get('strategy_type', 'file').lower()

    if strategy_type == 'clip':
        print("❌ Clip-based strategy not supported in batch evaluation")
        return {}
    else:
        # File-based strategy parameters
        params = {
            'batch_size': run.get('file_batch_size'),
            'large_file_threshold': run.get('large_file_threshold'),
            'very_large_threshold': run.get('very_large_threshold'),
            'max_file_length': run.get('max_file_length'),
            'max_file_size_mb': run.get('max_file_size_mb'),
            'overlap_seconds': run.get('overlap_seconds'),
        }

    # Drop None values
    return {k: v for k, v in params.items() if v is not None}


def _calibrate_if_needed(strategy, run: Dict[str, Any]) -> None:
    """Mirror CLI calibration flow exactly."""
    # Merge defaults with any per-run overrides
    cfg = {**DEFAULT_CALIBRATION, **{k: v for k, v in run.items() if k in DEFAULT_CALIBRATION}}

    model_path = run.get('model_path')
    default_threshold_path = os.path.join(model_path, 'threshold.json') if model_path else None
    threshold_file = cfg.get('threshold_file') or default_threshold_path

    if cfg.get('calibrate', False):
        if not cfg.get('force_recalibrate', False):
            loaded = strategy.load_saved_threshold(threshold_file)
            if loaded:
                print(f"✓ Using saved calibrated threshold: {strategy.threshold:.6f}")
                return
        # Calibrate now
        print("🎯 Calibrating threshold...")
        strategy.calibrate_threshold(
            calibration_key=cfg.get('calibration_key', 'CalibrationDataset'),
            threshold_file=threshold_file,
            calibration_fraction=cfg.get('calibration_fraction', 1.0)
        )
    else:
        # Try to load existing threshold even if not calibrating
        _ = strategy.load_saved_threshold(threshold_file)


def run_single_evaluation(run: Dict[str, Any]) -> Dict[str, Any] | None:
    """Create strategy, optionally calibrate, evaluate all datasets, and generate outputs.

    If calibration is enabled (per DEFAULT_CALIBRATION or run overrides), this uses the direct
    strategy path (exactly like CLI calibration flow). Otherwise, it delegates to the same
    unified pipeline runner used by the CLI for compactness and identical saving behavior.
    """
    strategy_type = run.get('strategy_type', 'file')
    model_path = run.get('model_path')
    output_dir = run.get('output_dir')

    if not model_path or not output_dir:
        print("❌ Missing required keys: 'model_path' and 'output_dir'")
        return None

    # Validate model path exists (like CLI does)
    if not _validate_model_path(model_path):
        return None

    # Merge calibration settings to decide execution path
    calib_cfg = {**DEFAULT_CALIBRATION, **{k: v for k, v in run.items() if k in DEFAULT_CALIBRATION}}
    do_calibrate = calib_cfg.get('calibrate', False)

    # Prepare output directory (like CLI does)
    output_dir = _prepare_output_dir(output_dir, auto_version=True)
    run['output_dir'] = output_dir  # Update run dict with final output dir

    # Build strategy parameters (shared between paths)
    params = _build_strategy_params(run)

    print(f"🚀 Starting {strategy_type} evaluation...")
    print(f"   Model: {model_path}")
    print(f"   Output: {output_dir}")

    # Path A: Direct strategy path (supports calibration exactly like CLI)
    if do_calibrate:
        print("   Mode: Direct evaluation with calibration")
        try:
            strategy = EvaluationStrategyFactory.create_strategy(
                strategy_type=strategy_type,
                model_path=model_path,
                output_dir=output_dir,
                **params
            )
            strategy.setup()

            # Calibration flow (mirrors CLI)
            _calibrate_if_needed(strategy, run)

            print("📊 Running evaluation on all datasets...")
            results = strategy.evaluate_all_datasets()

            if results:
                print("📈 Generating outputs...")
                strategy.generate_outputs(results)
                print(f"✅ Processed {len(results)} datasets")
                return results
            else:
                print("❌ No results generated")
                return None

        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    # Path B: Unified pipeline runner (same module import as CLI)
    print("   Mode: Unified pipeline (no calibration)")
    try:
        runner = create_unified_pipeline_runner()
        # The unified runner accepts kwargs that are forwarded as strategy params
        results = runner(strategy_type=strategy_type, model_path=model_path, output_dir=output_dir, **params)
        if results:
            print(f"✅ Unified pipeline processed {len(results)} datasets")
            return results
        else:
            print("❌ Unified pipeline returned no results")
            return None
    except Exception as e:
        print(f"❌ Unified pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_all_evaluations() -> None:
    """Run all entries in EVALUATION_RUNS sequentially with CLI-equivalent behavior."""
    print("\n" + "=" * 80)
    print("BATCH EVALUATION - CLI EQUIVALENT")
    print("=" * 80)
    print(f"Total runs: {len(EVALUATION_RUNS)}")
    print("Settings:")
    print(f"  - Strategy: file")
    print(f"  - Calibration: {DEFAULT_CALIBRATION['calibrate']}")
    print(f"  - Calibration key: {DEFAULT_CALIBRATION['calibration_key']}")
    print(f"  - Force recalibrate: {DEFAULT_CALIBRATION['force_recalibrate']}")
    print("=" * 80)

    successes = 0
    failures = 0
    total_time = 0

    for idx, run in enumerate(EVALUATION_RUNS, start=1):
        run_name = run.get('name', f"run_{idx}")
        print(f"\n▶ [{idx}/{len(EVALUATION_RUNS)}] {run_name}")
        print("-" * 60)

        start = time.time()
        try:
            results = run_single_evaluation(run)
            elapsed = time.time() - start
            total_time += elapsed

            if results:
                print(f"✅ {run_name} completed successfully in {elapsed:.1f}s")
                successes += 1
            else:
                print(f"❌ {run_name} returned no results")
                failures += 1

        except KeyboardInterrupt:
            print(f"\n⚠️ Interrupted by user during {run_name}")
            break
        except Exception as e:
            elapsed = time.time() - start
            total_time += elapsed
            print(f"❌ {run_name} failed after {elapsed:.1f}s: {e}")
            failures += 1
        finally:
            # Clean up CUDA memory after each run
            safe_cuda_cleanup()

    print("\n" + "=" * 80)
    print("BATCH EVALUATION SUMMARY")
    print("=" * 80)
    print(f"✅ Successful runs: {successes}")
    print(f"❌ Failed runs: {failures}")
    print(f"⏱️  Total time: {total_time:.1f}s ({total_time / 60:.1f} minutes)")
    if successes + failures > 0:
        print(f"📊 Success rate: {successes / (successes + failures) * 100:.1f}%")
    print("=" * 80)


if __name__ == '__main__':
    # Ensure non-interactive dataset choice
    os.environ['DRONE_EVALUATION_NON_INTERACTIVE'] = '1'

    try:
        run_all_evaluations()
    except KeyboardInterrupt:
        print("\n⚠️ Batch evaluation interrupted by user")
    except Exception as e:
        print(f"\n❌ Batch evaluation failed: {e}")
        import traceback

        traceback.print_exc()