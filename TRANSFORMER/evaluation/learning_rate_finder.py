#!/usr/bin/env python3
"""
Learning Rate Finder for Transformer Models.
This module implements a learning rate finder that helps determine the optimal learning rate
by training with exponentially increasing learning rates and plotting the loss curve.
"""
import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm.auto import tqdm
from typing import Dict



class LearningRateFinder:
    """
    Learning Rate Finder for finding optimal learning rates.
    Based on the method described in "Cyclical Learning Rates for Training Neural Networks"
    and popularized by fast.ai.

    Not tested with proper data yet.
    """

    def __init__(self, model, train_dataloader, data_collator, compute_metrics_fn=None, device=None):
        """
        Initialize the Learning Rate Finder.

        Args:
            model: The model to train
            train_dataloader: Training data loader (or dataset that will be converted to dataloader)
            data_collator: Data collator for batching
            compute_metrics_fn: Optional metrics computation function
            device: Device to run on (auto-detected if None)
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.data_collator = data_collator
        self.compute_metrics_fn = compute_metrics_fn
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Results storage
        self.learning_rates = []
        self.losses = []
        self.metrics = []

        # State tracking
        self.best_loss = float('inf')
        self.best_lr = None

    def find(self,
             start_lr: float = 1e-7,
             end_lr: float = 1.0,
             num_iter: int = 100,
             beta: float = 0.98,
             stop_div_threshold: float = 4.0,
             accumulation_steps: int = 1,
             save_plot: bool = True,
             output_dir: str = "./lr_finder_results") -> Dict:
        """
        Find the optimal learning rate.

        Args:
            start_lr: Starting learning rate
            end_lr: Ending learning rate
            num_iter: Number of iterations to test
            beta: Smoothing factor for loss (exponential moving average)
            stop_div_threshold: Stop if loss increases by this factor
            accumulation_steps: Gradient accumulation steps
            save_plot: Whether to save the plot
            output_dir: Directory to save results

        Returns:
            Dict with results including suggested learning rates
        """
        print("🔍 Starting Learning Rate Finder...")
        print(f"📊 Testing {num_iter} iterations from {start_lr:.2e} to {end_lr:.2e}")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Reset results
        self.learning_rates = []
        self.losses = []
        self.metrics = []

        # Save original model state
        original_state = copy.deepcopy(self.model.state_dict())

        # Setup optimizer with starting learning rate
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=start_lr)

        # Calculate learning rate multiplier
        lr_multiplier = (end_lr / start_lr) ** (1.0 / num_iter)

        # Initialize tracking variables
        smoothed_loss = 0
        best_loss = float('inf')
        iteration = 0

        # Move model to device
        self.model.to(self.device)
        self.model.train()

        # Create data iterator
        if hasattr(self.train_dataloader, '__iter__'):
            data_iter = iter(self.train_dataloader)
        else:
            # If it's a dataset, create a simple iterator
            from torch.utils.data import DataLoader
            dataloader = DataLoader(
                self.train_dataloader,
                batch_size=16,
                collate_fn=self.data_collator,
                shuffle=True
            )
            data_iter = iter(dataloader)

        # Progress bar
        pbar = tqdm(range(num_iter), desc="Finding optimal LR")

        try:
            for i in pbar:
                # Get current learning rate
                current_lr = start_lr * (lr_multiplier ** i)

                # Update learning rate
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr

                # Get batch
                try:
                    batch = next(data_iter)
                except StopIteration:
                    # Restart iterator if we run out of data
                    if hasattr(self.train_dataloader, '__iter__'):
                        data_iter = iter(self.train_dataloader)
                    else:
                        dataloader = DataLoader(
                            self.train_dataloader,
                            batch_size=16,
                            collate_fn=self.data_collator,
                            shuffle=True
                        )
                        data_iter = iter(dataloader)
                    batch = next(data_iter)

                # Move batch to device - handle different batch formats
                if isinstance(batch, dict):
                    # Standard dictionary format - ensure all values are tensors
                    processed_batch = {}
                    for k, v in batch.items():
                        if isinstance(v, torch.Tensor):
                            processed_batch[k] = v.to(self.device)
                        elif isinstance(v, (list, np.ndarray)):
                            # Convert lists/arrays to tensors
                            if isinstance(v, list):
                                # Handle nested lists or simple lists
                                if len(v) > 0 and isinstance(v[0], (list, np.ndarray)):
                                    # Nested list - convert to tensor
                                    processed_batch[k] = torch.tensor(v, dtype=torch.float32).to(self.device)
                                else:
                                    # Simple list of numbers
                                    processed_batch[k] = torch.tensor(v, dtype=torch.float32).to(self.device)
                            else:
                                # NumPy array
                                processed_batch[k] = torch.from_numpy(v).to(self.device)
                        elif isinstance(v, (int, float)):
                            # Single number - convert to tensor
                            processed_batch[k] = torch.tensor([v]).to(self.device)
                        else:
                            # Keep as-is for other types
                            processed_batch[k] = v
                    batch = processed_batch
                elif isinstance(batch, (list, tuple)):
                    # Handle tuple/list format - convert to dict if needed
                    if len(batch) >= 2:
                        # Assume (input_values, labels) format or similar
                        if isinstance(batch[0], torch.Tensor) and isinstance(batch[1], torch.Tensor):
                            batch = {
                                'input_values': batch[0].to(self.device),
                                'labels': batch[1].to(self.device)
                            }
                        else:
                            # Try to convert tuple elements to tensors
                            batch_dict = {}
                            if hasattr(batch[0], 'to'):  # First element is tensor-like
                                batch_dict['input_values'] = batch[0].to(self.device)
                            if len(batch) > 1 and hasattr(batch[1], 'to'):  # Second element is tensor-like
                                batch_dict['labels'] = batch[1].to(self.device)
                            batch = batch_dict
                    else:
                        # Single element tuple/list
                        if isinstance(batch[0], torch.Tensor):
                            batch = {'input_values': batch[0].to(self.device)}
                        else:
                            batch = batch[0]
                            if isinstance(batch, dict):
                                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                                        for k, v in batch.items()}
                            else:
                                batch = batch.to(self.device)
                else:
                    # Direct tensor or other format
                    if hasattr(batch, 'to'):
                        batch = batch.to(self.device)

                # Forward pass with error handling
                try:
                    if isinstance(batch, dict):
                        # Ensure we have the required keys
                        if 'input_values' not in batch:
                            # Try to find the input key
                            input_keys = ['input_values', 'input_arrays', 'inputs', 'audio']
                            for key in input_keys:
                                if key in batch:
                                    batch['input_values'] = batch[key]
                                    break

                        # Debug: Print tensor shapes before forward pass
                        if i == 0:  # Only print for first iteration
                            for k, v in batch.items():
                                if isinstance(v, torch.Tensor):
                                    print(f"  🔍 {k}: shape {v.shape}, dtype {v.dtype}")

                        # Clean forward pass with only necessary arguments
                        forward_inputs = {}
                        if 'input_values' in batch:
                            input_values = batch['input_values']
                            # Ensure input_values has the correct shape [batch_size, sequence_length]
                            if input_values.dim() == 1:
                                # If 1D, add batch dimension
                                input_values = input_values.unsqueeze(0)
                            elif input_values.dim() == 3 and input_values.size(0) == 1:
                                # If 3D with batch size 1, squeeze the first dimension
                                input_values = input_values.squeeze(0)
                            forward_inputs['input_values'] = input_values

                        if 'attention_mask' in batch:
                            attention_mask = batch['attention_mask']
                            # Ensure attention_mask has the correct shape to match input_values
                            if attention_mask.dim() == 1 and 'input_values' in forward_inputs:
                                attention_mask = attention_mask.unsqueeze(0)
                            elif attention_mask.dim() == 3 and attention_mask.size(0) == 1:
                                attention_mask = attention_mask.squeeze(0)
                            forward_inputs['attention_mask'] = attention_mask

                        if 'labels' in batch:
                            labels = batch['labels']
                            # Ensure labels have the correct shape
                            if labels.dim() == 0:
                                # If scalar, add batch dimension
                                labels = labels.unsqueeze(0)
                            forward_inputs['labels'] = labels

                        # Debug: Print corrected tensor shapes
                        if i == 0:
                            print("🔧 Corrected tensor shapes:")
                            for k, v in forward_inputs.items():
                                if isinstance(v, torch.Tensor):
                                    print(f"  {k}: shape {v.shape}, dtype {v.dtype}")

                        outputs = self.model(**forward_inputs)
                    else:
                        outputs = self.model(batch)

                except Exception as forward_error:
                    import traceback
                    print(f"🚨 Forward pass error: {forward_error}")
                    print(f"🔍 Error traceback:")
                    traceback.print_exc()
                    print(f"🔍 Batch type: {type(batch)}")
                    if isinstance(batch, dict):
                        print(f"🔍 Batch keys: {list(batch.keys())}")
                        for k, v in batch.items():
                            if isinstance(v, torch.Tensor):
                                print(f"  {k}: shape {v.shape}, dtype {v.dtype}, device {v.device}")
                            else:
                                print(f"  {k}: type {type(v)}, value {v}")

                    # More detailed debugging for the first few errors
                    if i < 5:
                        print(f"🔍 Model type: {type(self.model)}")
                        print(f"🔍 Model device: {next(self.model.parameters()).device}")

                        # Try a minimal forward pass
                        try:
                            if isinstance(batch, dict) and 'input_values' in batch:
                                minimal_inputs = {'input_values': batch['input_values']}
                                print(f"🔍 Trying minimal forward pass with only input_values...")
                                test_outputs = self.model(**minimal_inputs)
                                print(f"✅ Minimal forward pass succeeded")
                        except Exception as minimal_error:
                            print(f"❌ Minimal forward pass also failed: {minimal_error}")

                    # Skip this iteration
                    continue

                # Check if outputs contain loss
                if hasattr(outputs, 'loss') and outputs.loss is not None:
                    loss = outputs.loss / accumulation_steps
                else:
                    # Calculate loss manually if not provided
                    if isinstance(batch, dict) and 'labels' in batch:
                        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                        loss_fn = torch.nn.CrossEntropyLoss()
                        loss = loss_fn(logits, batch['labels']) / accumulation_steps
                    else:
                        print("⚠️ Cannot compute loss: no labels found")
                        continue

                # Backward pass
                loss.backward()

                # Update weights every accumulation_steps
                if (i + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                # Record learning rate and loss
                current_loss = loss.item() * accumulation_steps
                self.learning_rates.append(current_lr)
                self.losses.append(current_loss)

                # Compute smoothed loss
                if i == 0:
                    smoothed_loss = current_loss
                else:
                    smoothed_loss = beta * smoothed_loss + (1 - beta) * current_loss

                # Check for best loss
                if smoothed_loss < best_loss:
                    best_loss = smoothed_loss
                    self.best_lr = current_lr

                # Stop if loss is diverging
                if i > 10 and smoothed_loss > stop_div_threshold * best_loss:
                    print(f"⚠️ Stopping early: loss diverged at LR {current_lr:.2e}")
                    break

                # Update progress bar
                pbar.set_postfix({
                    'LR': f'{current_lr:.2e}',
                    'Loss': f'{current_loss:.4f}',
                    'Smoothed': f'{smoothed_loss:.4f}'
                })

                iteration += 1

        except Exception as e:
            print(f"❌ Error during LR finding: {e}")

        finally:
            # Restore original model state
            self.model.load_state_dict(original_state)
            print("✅ Model state restored")

        # Analyze results
        results = self._analyze_results()

        # Plot results
        if save_plot:
            self._plot_results(output_dir)

        # Save numerical results
        self._save_results(results, output_dir)

        print(f"🎯 Learning Rate Finder completed!")
        print(f"📈 Suggested LR: {results['suggested_lr']:.2e}")
        print(f"📊 Results saved to: {output_dir}")

        return results

    def _analyze_results(self) -> Dict:
        """Analyze the results and suggest optimal learning rates."""
        if not self.learning_rates or not self.losses:
            return {'error': 'No results to analyze'}

        lrs = np.array(self.learning_rates)
        losses = np.array(self.losses)

        # Remove any infinite or NaN values
        valid_mask = np.isfinite(losses) & np.isfinite(lrs)
        lrs = lrs[valid_mask]
        losses = losses[valid_mask]

        if len(lrs) < 10:
            return {'error': 'Insufficient valid data points'}

        # Method 1: Minimum loss
        min_loss_idx = np.argmin(losses)
        min_loss_lr = lrs[min_loss_idx]

        # Method 2: Steepest gradient (fastest decrease)
        # Calculate the gradient of the loss curve
        gradients = np.gradient(losses)
        steepest_idx = np.argmin(gradients)
        steepest_lr = lrs[steepest_idx]

        # Method 3: Loss decrease threshold (first significant drop)
        # Find where loss drops to 90% of initial loss
        initial_loss = np.mean(losses[:5])  # Average of first 5 points
        target_loss = 0.9 * initial_loss

        threshold_idx = np.where(losses < target_loss)[0]
        threshold_lr = lrs[threshold_idx[0]] if len(threshold_idx) > 0 else min_loss_lr

        # Method 4: One order of magnitude before minimum
        one_order_before_lr = min_loss_lr / 10

        # Choose the most conservative (safer) suggestion
        # Usually between steepest gradient and one order before minimum
        suggested_lr = min(steepest_lr, one_order_before_lr)

        # If suggested LR is too close to starting point, use minimum loss LR
        if suggested_lr < lrs[0] * 10:
            suggested_lr = min_loss_lr / 3  # Conservative choice

        return {
            'suggested_lr': suggested_lr,
            'min_loss_lr': min_loss_lr,
            'steepest_gradient_lr': steepest_lr,
            'threshold_lr': threshold_lr,
            'one_order_before_min_lr': one_order_before_lr,
            'learning_rates': lrs.tolist(),
            'losses': losses.tolist(),
            'num_iterations': len(lrs),
            'min_loss': float(np.min(losses)),
            'initial_loss': float(initial_loss) if 'initial_loss' in locals() else None
        }

    def _plot_results(self, output_dir: str):
        """Plot the learning rate finder results."""
        if not self.learning_rates or not self.losses:
            return

        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        lrs = np.array(self.learning_rates)
        losses = np.array(self.losses)

        # Remove invalid values
        valid_mask = np.isfinite(losses) & np.isfinite(lrs)
        lrs = lrs[valid_mask]
        losses = losses[valid_mask]

        # Plot 1: Learning Rate vs Loss (log scale)
        ax1.plot(lrs, losses, 'b-', linewidth=2, alpha=0.8)
        ax1.set_xscale('log')
        ax1.set_xlabel('Learning Rate')
        ax1.set_ylabel('Loss')
        ax1.set_title('Learning Rate Finder - Loss vs Learning Rate')
        ax1.grid(True, alpha=0.3)

        # Mark important points
        if len(losses) > 0:
            min_loss_idx = np.argmin(losses)
            ax1.axvline(x=lrs[min_loss_idx], color='red', linestyle='--',
                       alpha=0.7, label=f'Min Loss LR: {lrs[min_loss_idx]:.2e}')

            # Mark suggested LR
            results = self._analyze_results()
            if 'suggested_lr' in results:
                ax1.axvline(x=results['suggested_lr'], color='green', linestyle='--',
                           alpha=0.7, label=f'Suggested LR: {results["suggested_lr"]:.2e}')

        ax1.legend()

        # Plot 2: Smoothed version
        # Apply smoothing to better see the trend
        from scipy.ndimage import uniform_filter1d
        if len(losses) > 10:
            smoothed_losses = uniform_filter1d(losses, size=max(3, len(losses)//20))
            ax2.plot(lrs, smoothed_losses, 'g-', linewidth=2, alpha=0.8, label='Smoothed Loss')
            ax2.plot(lrs, losses, 'b-', linewidth=1, alpha=0.3, label='Raw Loss')
        else:
            ax2.plot(lrs, losses, 'b-', linewidth=2, alpha=0.8, label='Loss')

        ax2.set_xscale('log')
        ax2.set_xlabel('Learning Rate')
        ax2.set_ylabel('Loss')
        ax2.set_title('Learning Rate Finder - Smoothed View')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()

        # Save plot
        plot_path = os.path.join(output_dir, 'learning_rate_finder.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"📊 Plot saved to: {plot_path}")

    def _save_results(self, results: Dict, output_dir: str):
        """Save numerical results to file."""
        import json

        # Save detailed results
        results_path = os.path.join(output_dir, 'lr_finder_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        # Save summary
        summary_path = os.path.join(output_dir, 'lr_finder_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("Learning Rate Finder Results\n")
            f.write("=" * 40 + "\n\n")

            if 'error' not in results:
                f.write(f"Suggested Learning Rate: {results['suggested_lr']:.2e}\n")
                f.write(f"Minimum Loss LR: {results['min_loss_lr']:.2e}\n")
                f.write(f"Steepest Gradient LR: {results['steepest_gradient_lr']:.2e}\n")
                f.write(f"Threshold LR: {results['threshold_lr']:.2e}\n")
                f.write(f"One Order Before Min LR: {results['one_order_before_min_lr']:.2e}\n\n")

                f.write(f"Number of iterations: {results['num_iterations']}\n")
                f.write(f"Minimum loss achieved: {results['min_loss']:.6f}\n")

                if results['initial_loss']:
                    f.write(f"Initial loss: {results['initial_loss']:.6f}\n")
                    improvement = (results['initial_loss'] - results['min_loss']) / results['initial_loss'] * 100
                    f.write(f"Loss improvement: {improvement:.1f}%\n")
            else:
                f.write(f"Error: {results['error']}\n")

        print(f"📄 Results saved to: {results_path}")
        print(f"📄 Summary saved to: {summary_path}")


def run_lr_finder_for_transformer(model, train_dataset, data_collator,
                                 compute_metrics_fn=None, config=None) -> Dict:
    """
    Convenience function to run learning rate finder for transformer models.

    Args:
        model: Transformer model
        train_dataset: Training dataset
        data_collator: Data collator
        compute_metrics_fn: Optional metrics function
        config: Optional configuration object with parameters

    Returns:
        Dictionary with LR finder results
    """
    # Default parameters
    start_lr = 1e-7
    end_lr = 1.0
    num_iter = 100
    output_dir = "./lr_finder_results"

    # Override with config if provided
    if config:
        start_lr = getattr(config, 'lr_finder_start_lr', start_lr)
        end_lr = getattr(config, 'lr_finder_end_lr', end_lr)
        num_iter = getattr(config, 'lr_finder_num_iter', num_iter)
        output_dir = getattr(config, 'lr_finder_output_dir', output_dir)

    # Initialize LR finder
    lr_finder = LearningRateFinder(
        model=model,
        train_dataloader=train_dataset,
        data_collator=data_collator,
        compute_metrics_fn=compute_metrics_fn
    )

    # Run the finder
    results = lr_finder.find(
        start_lr=start_lr,
        end_lr=end_lr,
        num_iter=num_iter,
        save_plot=True,
        output_dir=output_dir
    )

    return results
