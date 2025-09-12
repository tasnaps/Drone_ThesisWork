#!/usr/bin/env python3
"""
Sequential Training and Evaluation Script
Trains models with different augmentation combinations and evaluates each one.
"""

import os
import sys
import subprocess
import time
import json
import logging
import threading
import tempfile
from datetime import datetime
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from data.data_transformers import update_augmentation_mode, print_augmentation_info

def create_automated_cli_script(fusion_only=False):
    """
    Create a temporary CLI script that automatically answers the dataset selection prompt.

    Args:
        fusion_only: If True, automatically select Fusion dataset only. If False, select all datasets.

    Returns:
        Path to the temporary script file
    """

    # Read the original CLI script
    original_cli = Path(__file__).parent / "cli.py"

    if not original_cli.exists():
        raise FileNotFoundError(f"Original CLI script not found: {original_cli}")

    with open(original_cli, 'r', encoding='utf-8') as f:
        cli_content = f.read()

    # Add non-interactive mode handling at the top of the script
    non_interactive_header = '''
import os
import sys

# Check for non-interactive mode
NON_INTERACTIVE_MODE = os.environ.get('DRONE_EVALUATION_NON_INTERACTIVE', '0') == '1'

# Override input function for non-interactive mode
original_input = input

def safe_input(prompt=""):
    if NON_INTERACTIVE_MODE:
        # Return default values based on the prompt content
        prompt_lower = prompt.lower()
        if "fusion" in prompt_lower and ("y/n" in prompt_lower or "y" in prompt_lower):
            return "Y" if ''' + str(fusion_only).lower() + ''' else "N"
        elif "y/n" in prompt_lower:
            return "N"  # Default to no for other yes/no questions
        else:
            return ""  # Default empty response
    else:
        return original_input(prompt)

# Replace the built-in input function
input = safe_input

'''

    # Insert the non-interactive handling at the beginning after imports
    import_end = cli_content.find('\n\n')
    if import_end != -1:
        modified_content = cli_content[:import_end] + '\n' + non_interactive_header + cli_content[import_end:]
    else:
        modified_content = non_interactive_header + cli_content

    # Create temporary file with UTF-8 encoding
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='_automated_cli.py', delete=False, encoding='utf-8')
    temp_file.write(modified_content)
    temp_file.close()

    return temp_file.name

def cleanup_temp_script(temp_script_path):
    """Clean up the temporary script file."""
    try:
        os.unlink(temp_script_path)
    except Exception as e:
        print(f"Warning: Could not clean up temporary script {temp_script_path}: {e}")

class SequentialTrainEvalRunner:
    """Runner for sequential training and evaluation with different augmentation modes."""

    def __init__(self, fusion_only=False, enable_audio_shortening=False, min_length_ratio=0.5, max_length_ratio=0.75, random_crop=True):
        self.base_dir = Path(__file__).parent
        self.src_dir = self.base_dir.parent.parent
        self.training_script = self.base_dir / "pipelines" / "transformers" / "main_transformers.py"
        self.model_output_dir = "./models"
        self.base_eval_dir = "./eval_results"
        self.fusion_only = fusion_only  # Add fusion_only parameter

        # Audio shortening configuration
        self.enable_audio_shortening = enable_audio_shortening
        self.min_length_ratio = min_length_ratio
        self.max_length_ratio = max_length_ratio
        self.random_crop = random_crop

        # Experiment configurations to test
        self.experiment_configs = [
            # 1. Baseline experiment (no augmentation, full audio)
            {
                "augmentation_mode": "none",
                "name": "baseline_no_aug_full_audio",
                "description": "Baseline: No augmentation, full audio length",
                "enable_audio_shortening": False,
                "experiment_type": "baseline"
            },

            # 2. Augmentation experiments (full audio with different augmentations)
            {
                "augmentation_mode": "gaussian_only",
                "name": "gaussian_only_full_audio",
                "description": "Gaussian noise augmentation, full audio length",
                "enable_audio_shortening": False,
                "experiment_type": "augmentation"
            },
            {
                "augmentation_mode": "bandpass_only",
                "name": "bandpass_only_full_audio",
                "description": "Bandpass filter augmentation, full audio length",
                "enable_audio_shortening": False,
                "experiment_type": "augmentation"
            },
            {
                "augmentation_mode": "all",
                "name": "gaussian_bandpass_full_audio",
                "description": "Both Gaussian noise and bandpass filter, full audio length",
                "enable_audio_shortening": False,
                "experiment_type": "augmentation"
            },

            # 3. Shortened audio experiment (with augmentation)
            {
                "augmentation_mode": "all",  # Use best augmentation setup
                "name": "gaussian_bandpass_short_audio",
                "description": "Both augmentations with shortened audio (50-75% length)",
                "enable_audio_shortening": True,
                "experiment_type": "shortened_audio"
            }
        ]

        # Training configuration
        self.training_config = {
            "data_dir": "C:/Gradu Juttui/Datasets/DroneAudioDataset_Saraalemadi/Binary_Drone_Audio",
            "epochs": 20,
            "batch_size": 16,
            "learning_rate": 1e-5,
            "seed": 42,
            "disable_early_stopping": True  # As requested
        }

        # Training timeout configuration (in seconds)
        self.training_timeout = 8 * 3600  # 8 hours (increased from 1 hour)
        self.evaluation_timeout = 2 * 3600  # 2 hours (increased from 30 minutes)

        # Results tracking
        self.results = {}
        self.failed_runs = []

    def create_training_config_file(self, augmentation_config, run_name):
        """Create a temporary config file for training with specific augmentation settings."""
        config_data = {
            "augmentation": augmentation_config,
            "training": self.training_config,
            "run_name": run_name,
            "timestamp": datetime.now().isoformat()
        }

        config_file = self.base_dir / f"temp_config_{run_name}.json"
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)

        return config_file

    def write_run_info(self, eval_dir, experiment_config, training_config, run_name):
        """Write training and experiment settings to a txt file in the evaluation directory."""
        info_file = Path(eval_dir) / "run_info.txt"

        with open(info_file, 'w') as f:
            f.write(f"Training and Evaluation Run Information\n")
            f.write(f"=" * 50 + "\n\n")
            f.write(f"Run Name: {run_name}\n")
            f.write(f"Experiment Type: {experiment_config['experiment_type']}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write(f"Experiment Configuration:\n")
            f.write(f"  Description: {experiment_config['description']}\n")
            f.write(f"  Augmentation Mode: {experiment_config['augmentation_mode']}\n")
            f.write(f"  Audio Shortening Enabled: {experiment_config.get('enable_audio_shortening', False)}\n")
            
            if experiment_config.get('enable_audio_shortening', False):
                f.write(f"  Audio Length Range: {self.min_length_ratio*100:.0f}% - {self.max_length_ratio*100:.0f}% of original\n")
                f.write(f"  Audio Cropping Mode: {'Random' if self.random_crop else 'From start'}\n")
            f.write(f"\n")

            f.write(f"Training Configuration:\n")
            for key, value in training_config.items():
                f.write(f"  {key}: {value}\n")
            f.write(f"\n")

            f.write(f"Model Output Directory: {self.model_output_dir}\n")
            f.write(f"Evaluation Output Directory: {eval_dir}\n")

        print(f"📝 Run info saved to: {info_file}")

    def run_training(self, experiment_config, run_name):
        """Run training with specific experiment configuration."""
        print(f"\n{'='*60}")
        print(f"🏋️ TRAINING: {run_name}")
        print(f"📋 Experiment: {experiment_config['description']}")
        print(f"🧪 Type: {experiment_config['experiment_type']}")
        print(f"{'='*60}")

        # Update augmentation mode in the data transformer
        update_augmentation_mode(experiment_config["augmentation_mode"])
        print_augmentation_info()

        # Prepare training command
        training_cmd = [
            sys.executable, str(self.training_script),
            "--data-dir", self.training_config["data_dir"],
            "--output-dir", self.model_output_dir,
            "--epochs", str(self.training_config["epochs"]),
            "--batch-size", str(self.training_config["batch_size"]),
            "--learning-rate", str(self.training_config["learning_rate"]),
            "--seed", str(self.training_config["seed"]),
            "--disable-early-stopping"  # Add the early stopping disable flag
        ]

        # Add audio shortening arguments if enabled for this experiment
        if experiment_config.get("enable_audio_shortening", False):
            training_cmd.extend([
                "--shorten-audio",
                "--min-length-ratio", str(self.min_length_ratio),
                "--max-length-ratio", str(self.max_length_ratio)
            ])
            if not self.random_crop:
                training_cmd.append("--no-random-crop")

            print(f"🎵 Audio shortening enabled: {self.min_length_ratio*100:.0f}%-{self.max_length_ratio*100:.0f}% of original length")
            print(f"🎵 Cropping mode: {'Random' if self.random_crop else 'From start'}")
        else:
            print(f"🎵 Audio shortening disabled - using full audio files")

        print(f"🚀 Starting training with command:")
        print(f"   {' '.join(training_cmd)}")

        start_time = time.time()

        try:
            # Start subprocess with real-time output streaming
            process = subprocess.Popen(
                training_cmd,
                cwd=self.base_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                errors='replace',
                bufsize=1,  # Line buffered
                universal_newlines=True
            )

            # Stream output in real-time
            stdout_thread, stderr_thread = self.stream_subprocess_output(process, "TRAIN")

            # Wait for process to complete with timeout
            try:
                return_code = process.wait(timeout=self.training_timeout)  # 8 hours timeout

                # Wait for output threads to finish
                stdout_thread.join(timeout=5)
                stderr_thread.join(timeout=5)

                elapsed_time = time.time() - start_time

                if return_code == 0:
                    print(f"✅ Training completed successfully in {elapsed_time:.1f}s")
                    return True, elapsed_time
                else:
                    print(f"❌ Training failed with return code {return_code}")
                    return False, elapsed_time

            except subprocess.TimeoutExpired:
                print(f"❌ Training timed out after {self.training_timeout/3600:.1f} hours")
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                return False, self.training_timeout

        except Exception as e:
            elapsed_time = time.time() - start_time
            print(f"❌ Training failed with exception: {e}")
            return False, elapsed_time

    def run_evaluation(self, run_name):
        """Run CLI evaluation on the trained model."""
        print(f"\n🔬 EVALUATION: {run_name}")
        print(f"📊 Running CLI evaluation script...")
        print(f"📊 Dataset mode: {'Fusion only' if self.fusion_only else 'All datasets'}")

        # Create evaluation directory name
        eval_dir = f"{self.base_eval_dir}_{run_name}"

        start_time = time.time()
        temp_script_path = None

        try:
            # Create temporary automated CLI script
            temp_script_path = create_automated_cli_script(fusion_only=self.fusion_only)
            print(f"🤖 Created automated CLI script: {temp_script_path}")

            # Prepare evaluation command using the temporary script
            eval_cmd = [
                sys.executable, temp_script_path,
                "file",  # Use file-based evaluation
                "--model-path", self.model_output_dir,
                "--output-dir", eval_dir,
                "--overwrite"  # Allow overwriting
            ]

            print(f"🚀 Starting evaluation with command:")
            print(f"   {' '.join(eval_cmd)}")

            # Start subprocess with real-time output streaming and better error handling
            try:
                # Set environment variables to disable interactive prompts
                env = os.environ.copy()
                env['PYTHONUNBUFFERED'] = '1'  # Ensure unbuffered output
                env['PYTHONIOENCODING'] = 'utf-8'  # Force UTF-8 encoding
                env['DRONE_EVALUATION_NON_INTERACTIVE'] = '1'  # Custom flag for our scripts

                process = subprocess.Popen(
                    eval_cmd,
                    cwd=self.base_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    stdin=subprocess.DEVNULL,  # Disable stdin to prevent input() calls
                    text=True,
                    encoding='utf-8',
                    errors='replace',  # Replace problematic characters instead of crashing
                    bufsize=1,  # Line buffered
                    universal_newlines=True,
                    env=env  # Use modified environment
                )

                # Stream output in real-time
                stdout_thread, stderr_thread = self.stream_subprocess_output(process, "EVAL")

                # Wait for process to complete with timeout
                return_code = process.wait(timeout=self.evaluation_timeout)  # 2 hours timeout

                # Wait for output threads to finish
                stdout_thread.join(timeout=10)  # Increased timeout for thread cleanup
                stderr_thread.join(timeout=10)

                elapsed_time = time.time() - start_time

                if return_code == 0:
                    print(f"✅ Evaluation completed successfully in {elapsed_time:.1f}s")
                    print(f"📁 Results saved to: {eval_dir}")
                    return True, elapsed_time, eval_dir
                else:
                    print(f"❌ Evaluation failed with return code {return_code}")
                    return False, elapsed_time, None

            except subprocess.TimeoutExpired:
                print(f"❌ Evaluation timed out after {self.evaluation_timeout/3600:.1f} hours")
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                return False, self.evaluation_timeout, None
            except UnicodeDecodeError as e:
                print(f"❌ Evaluation failed due to Unicode error: {e}")
                print("💡 This might be due to special characters in the output.")
                elapsed_time = time.time() - start_time
                return False, elapsed_time, None

        except Exception as e:
            elapsed_time = time.time() - start_time
            print(f"❌ Evaluation failed with exception: {e}")
            print(f"   Exception type: {type(e).__name__}")
            return False, elapsed_time, None
        finally:
            # Clean up temporary script
            if temp_script_path:
                cleanup_temp_script(temp_script_path)
                print(f"🧹 Cleaned up temporary script")

    def stream_subprocess_output(self, process, prefix=""):
        """Stream subprocess output in real-time."""
        def stream_output(pipe, prefix):
            try:
                for line in iter(pipe.readline, ''):
                    if line:
                        # Additional Unicode safety for console output
                        try:
                            print(f"{prefix}{line.rstrip()}")
                        except UnicodeEncodeError:
                            # Fallback: encode with error replacement for console output
                            safe_line = line.rstrip().encode('utf-8', errors='replace').decode('utf-8')
                            print(f"{prefix}{safe_line}")
                        sys.stdout.flush()
            except UnicodeDecodeError as e:
                print(f"Unicode decode error in streaming: {e}")
            except Exception as e:
                print(f"Error streaming output: {e}")
            finally:
                pipe.close()

        # Start threads to stream stdout and stderr
        stdout_thread = threading.Thread(
            target=stream_output,
            args=(process.stdout, f"   📝 {prefix}")
        )
        stderr_thread = threading.Thread(
            target=stream_output,
            args=(process.stderr, f"   ❌ {prefix}")
        )

        stdout_thread.daemon = True
        stderr_thread.daemon = True
        stdout_thread.start()
        stderr_thread.start()

        return stdout_thread, stderr_thread

    def run_sequential_experiments(self):
        """Run all training and evaluation experiments sequentially."""
        print(f"\n🚀 Starting Sequential Training and Evaluation")
        print(f"📋 {len(self.experiment_configs)} augmentation configurations to process")
        print(f"🏋️ Training configuration: {self.training_config}")
        print("="*80)

        overall_start_time = time.time()

        for i, experiment_config in enumerate(self.experiment_configs, 1):
            run_name = experiment_config["name"]

            print(f"\n🎯 Processing experiment {i}/{len(self.experiment_configs)}: {run_name}")
            print(f"🧪 Type: {experiment_config['experiment_type']}")

            experiment_start_time = time.time()

            # Step 1: Training
            training_success, training_time = self.run_training(experiment_config, run_name)

            if not training_success:
                self.failed_runs.append({
                    "run_name": run_name,
                    "experiment_config": experiment_config,
                    "training_config": self.training_config,
                    "error": "Training failed",
                    "stage": "training"
                })
                print(f"⏭️  Skipping evaluation for {run_name} due to training failure")
                continue

            # Step 2: Evaluation (immediately after training)
            eval_success, eval_time, eval_dir = self.run_evaluation(run_name)

            if not eval_success:
                self.failed_runs.append({
                    "run_name": run_name,
                    "experiment_config": experiment_config,
                    "training_config": self.training_config,
                    "error": "Evaluation failed",
                    "stage": "evaluation"
                })
                print(f"⚠️  Training succeeded but evaluation failed for {run_name}")
            else:
                # Step 3: Write run information to evaluation directory
                self.write_run_info(eval_dir, experiment_config, self.training_config, run_name)

                # Record successful run
                experiment_time = time.time() - experiment_start_time
                self.results[run_name] = {
                    "experiment_config": experiment_config,
                    "training_config": self.training_config,
                    "training_time": training_time,
                    "evaluation_time": eval_time,
                    "total_time": experiment_time,
                    "evaluation_dir": eval_dir,
                    "status": "success"
                }

                print(f"✅ {run_name} completed successfully in {experiment_time:.1f}s")
                print(f"   Training: {training_time:.1f}s, Evaluation: {eval_time:.1f}s")

        # Generate final summary
        total_time = time.time() - overall_start_time
        self.generate_final_summary(total_time)

    def generate_final_summary(self, total_time):
        """Generate and save final summary of all experiments."""
        print(f"\n📊 SEQUENTIAL EXPERIMENTS SUMMARY")
        print("="*80)
        print(f"⏱️  Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print(f"✅ Successful experiments: {len(self.results)}")
        print(f"❌ Failed experiments: {len(self.failed_runs)}")

        if self.results:
            print(f"\n🏆 Successful Experiments:")
            for run_name, data in self.results.items():
                print(f"  📈 {run_name}:")
                print(f"     Description: {data['experiment_config']['description']}")
                print(f"     Type: {data['experiment_config']['experiment_type']}")
                print(f"     Times: Training {data['training_time']:.1f}s, Eval {data['evaluation_time']:.1f}s")
                print(f"     Results: {data['evaluation_dir']}")

        if self.failed_runs:
            print(f"\n💥 Failed Experiments:")
            for failed in self.failed_runs:
                print(f"  ❌ {failed['run_name']} (failed at {failed['stage']}): {failed['error']}")

        # Save detailed summary
        summary_data = {
            "experiment_summary": {
                "total_time_seconds": total_time,
                "successful_count": len(self.results),
                "failed_count": len(self.failed_runs),
                "timestamp": datetime.now().isoformat()
            },
            "successful_experiments": self.results,
            "failed_experiments": self.failed_runs,
            "training_configuration": self.training_config,
            "experiment_configurations": self.experiment_configs  # Updated key name
        }

        summary_file = self.base_dir / f"sequential_experiments_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        try:
            with open(summary_file, 'w') as f:
                json.dump(summary_data, f, indent=2, default=str)
            print(f"\n📄 Detailed summary saved to: {summary_file}")
        except Exception as e:
            print(f"⚠️  Could not save summary: {e}")

        print(f"\n🎉 Sequential experiments completed!")


def main():
    """Main entry point for sequential training and evaluation."""
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Sequential Training and Evaluation Script")
    parser.add_argument(
        "--fusion-only",
        action="store_true",
        help="Evaluate only on Fusion dataset instead of all datasets"
    )
    parser.add_argument(
        "--eval-all-datasets",
        action="store_true",
        help="Evaluate on all datasets (default behavior)"
    )

    # Audio shortening arguments
    parser.add_argument(
        "--shorten-audio",
        action="store_true",
        help="Enable audio shortening during training (50-75% of original length by default)"
    )
    parser.add_argument(
        "--min-length-ratio",
        type=float,
        default=0.5,
        help="Minimum length ratio for audio shortening (default: 0.5 = 50%%)"
    )
    parser.add_argument(
        "--max-length-ratio",
        type=float,
        default=0.75,
        help="Maximum length ratio for audio shortening (default: 0.75 = 75%%)"
    )
    parser.add_argument(
        "--no-random-crop",
        action="store_true",
        help="Use start cropping instead of random cropping for audio shortening"
    )

    args = parser.parse_args()

    # Determine dataset evaluation mode
    fusion_only = args.fusion_only
    if args.eval_all_datasets and args.fusion_only:
        print("⚠️  Warning: Both --fusion-only and --eval-all-datasets specified. Using --fusion-only.")

    print(f"📊 Dataset evaluation mode: {'Fusion only' if fusion_only else 'All datasets'}")

    # Display audio shortening configuration
    if args.shorten_audio:
        print(f"🎵 Audio shortening: ENABLED ({args.min_length_ratio*100:.0f}%-{args.max_length_ratio*100:.0f}% of original)")
        print(f"🎵 Cropping mode: {'Random' if not args.no_random_crop else 'From start'}")
    else:
        print(f"🎵 Audio shortening: DISABLED (using full audio files)")

    try:
        runner = SequentialTrainEvalRunner(
            fusion_only=fusion_only,
            enable_audio_shortening=args.shorten_audio,
            min_length_ratio=args.min_length_ratio,
            max_length_ratio=args.max_length_ratio,
            random_crop=not args.no_random_crop
        )
        runner.run_sequential_experiments()

    except KeyboardInterrupt:
        print("\n⏹️  Sequential experiments interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error during sequential experiments: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
