#!/usr/bin/env python3
"""
Enhanced error recovery and resilience utilities for transformer model evaluation.
This module provides automatic retry logic, graceful degradation, and robust error handling.
"""

import time
import torch
import functools
import traceback
from typing import Callable, Any, Optional, Dict, List
from dataclasses import dataclass
from enum import Enum
import warnings


class ErrorSeverity(Enum):
    """Classification of error severity levels"""
    LOW = "low"          # Recoverable, can retry
    MEDIUM = "medium"    # Partially recoverable, may need fallback
    HIGH = "high"        # Critical, requires intervention
    FATAL = "fatal"      # Unrecoverable, must abort


@dataclass
class ErrorContext:
    """Context information for error handling"""
    operation: str
    dataset_name: str
    file_count: int
    attempt_number: int
    error_type: type
    error_message: str
    severity: ErrorSeverity
    recovery_strategy: Optional[str] = None


class ResilientEvaluator:
    """Wrapper that adds resilience to evaluation operations"""

    def __init__(self, max_retries=3, retry_delay=1.0, memory_cleanup_threshold=0.8):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.memory_cleanup_threshold = memory_cleanup_threshold
        self.error_history: List[ErrorContext] = []
        self.recovery_stats = {
            'total_errors': 0,
            'recovered_errors': 0,
            'failed_recoveries': 0,
            'memory_cleanups': 0
        }

    def classify_error(self, error: Exception, context: Dict) -> ErrorSeverity:
        """Classify error severity based on type and context"""
        error_type = type(error)
        error_msg = str(error).lower()

        # Memory-related errors - usually recoverable
        if (error_type in [torch.cuda.OutOfMemoryError, RuntimeError] and
            "out of memory" in error_msg):
            return ErrorSeverity.MEDIUM

        # File/path errors - usually fatal for that dataset
        if "no such file" in error_msg or "permission denied" in error_msg:
            return ErrorSeverity.HIGH

        # Model loading errors - usually fatal
        if "checkpoint" in error_msg or "state_dict" in error_msg:
            return ErrorSeverity.FATAL

        # Network/timeout errors - usually recoverable
        if "timeout" in error_msg or "connection" in error_msg:
            return ErrorSeverity.LOW

        # Data processing errors - usually recoverable
        if error_type in [ValueError, IndexError] and context.get('file_count', 0) > 0:
            return ErrorSeverity.LOW

        # Default to medium severity
        return ErrorSeverity.MEDIUM

    def get_recovery_strategy(self, error: Exception, severity: ErrorSeverity, context: Dict) -> Optional[str]:
        """Determine appropriate recovery strategy"""
        error_msg = str(error).lower()

        if severity == ErrorSeverity.FATAL:
            return None

        if "out of memory" in error_msg:
            return "memory_cleanup"

        if severity == ErrorSeverity.LOW:
            return "simple_retry"

        if severity == ErrorSeverity.MEDIUM:
            if context.get('file_count', 0) > 1:
                return "reduce_batch_size"
            else:
                return "skip_problematic_file"

        return "simple_retry"

    def execute_recovery_strategy(self, strategy: str, context: Dict) -> bool:
        """Execute the specified recovery strategy"""
        try:
            if strategy == "memory_cleanup":
                return self._cleanup_memory()
            elif strategy == "reduce_batch_size":
                return self._reduce_batch_size(context)
            elif strategy == "skip_problematic_file":
                return self._skip_problematic_data(context)
            elif strategy == "simple_retry":
                time.sleep(self.retry_delay)
                return True

        except Exception as e:
            print(f"Recovery strategy '{strategy}' failed: {e}")
            return False

        return False

    def _cleanup_memory(self) -> bool:
        """Aggressive memory cleanup"""
        try:
            import gc

            # Python garbage collection
            gc.collect()

            # CUDA cleanup if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            self.recovery_stats['memory_cleanups'] += 1
            time.sleep(2.0)  # Give system time to stabilize

            print("    🔧 Memory cleanup completed")
            return True

        except Exception as e:
            print(f"    ❌ Memory cleanup failed: {e}")
            return False

    def _reduce_batch_size(self, context: Dict) -> bool:
        """Reduce batch size for memory-constrained operations"""
        trainer = context.get('trainer')
        if trainer and hasattr(trainer.args, 'per_device_eval_batch_size'):
            original_batch_size = trainer.args.per_device_eval_batch_size
            new_batch_size = max(1, original_batch_size // 2)

            if new_batch_size < original_batch_size:
                trainer.args.per_device_eval_batch_size = new_batch_size
                print(f"    🔧 Reduced batch size: {original_batch_size} → {new_batch_size}")
                return True

        return False

    def _skip_problematic_data(self, context: Dict) -> bool:
        """Mark data as problematic and skip it"""
        dataset_name = context.get('dataset_name', 'unknown')
        print(f"    ⚠️  Marking {dataset_name} as problematic - will skip")
        context['skip_dataset'] = True
        return True

    def resilient_operation(self, operation: Callable, context: Dict, *args, **kwargs) -> Any:
        """Execute operation with automatic error recovery"""
        operation_name = getattr(operation, '__name__', str(operation))
        dataset_name = context.get('dataset_name', 'unknown')

        for attempt in range(self.max_retries + 1):
            try:
                # Check if we should skip this dataset
                if context.get('skip_dataset', False):
                    print(f"    ⏭️  Skipping {dataset_name} (marked as problematic)")
                    return None

                # Execute the operation
                result = operation(*args, **kwargs)

                # Success - reset any failure flags
                if attempt > 0:
                    print(f"    ✅ Operation succeeded on attempt {attempt + 1}")

                return result

            except Exception as e:
                self.recovery_stats['total_errors'] += 1

                # Classify error severity
                severity = self.classify_error(e, context)
                recovery_strategy = self.get_recovery_strategy(e, severity, context)

                # Create error context
                error_ctx = ErrorContext(
                    operation=operation_name,
                    dataset_name=dataset_name,
                    file_count=context.get('file_count', 0),
                    attempt_number=attempt + 1,
                    error_type=type(e),
                    error_message=str(e),
                    severity=severity,
                    recovery_strategy=recovery_strategy
                )

                self.error_history.append(error_ctx)

                print(f"    ❌ Attempt {attempt + 1} failed: {e}")
                print(f"    📊 Error severity: {severity.value}")

                # Handle fatal errors immediately
                if severity == ErrorSeverity.FATAL:
                    print(f"    💀 Fatal error - cannot recover")
                    raise e

                # If this is the last attempt, give up
                if attempt >= self.max_retries:
                    print(f"    🔄 Max retries ({self.max_retries}) exceeded")
                    self.recovery_stats['failed_recoveries'] += 1
                    raise e

                # Try recovery strategy
                if recovery_strategy:
                    print(f"    🔧 Attempting recovery: {recovery_strategy}")
                    recovery_success = self.execute_recovery_strategy(recovery_strategy, context)

                    if recovery_success:
                        self.recovery_stats['recovered_errors'] += 1
                        print(f"    ♻️  Recovery successful, retrying...")
                        continue
                    else:
                        print(f"    ❌ Recovery failed")

                # Default retry with delay
                if attempt < self.max_retries:
                    print(f"    ⏳ Waiting {self.retry_delay}s before retry...")
                    time.sleep(self.retry_delay)

        # If we get here, all attempts failed
        self.recovery_stats['failed_recoveries'] += 1
        raise Exception(f"Operation {operation_name} failed after {self.max_retries + 1} attempts")

    def get_recovery_report(self) -> Dict:
        """Generate report on error recovery performance"""
        total_errors = self.recovery_stats['total_errors']
        recovered = self.recovery_stats['recovered_errors']
        failed = self.recovery_stats['failed_recoveries']

        recovery_rate = (recovered / total_errors * 100) if total_errors > 0 else 0

        # Analyze error patterns
        error_patterns = {}
        for error_ctx in self.error_history:
            error_type = error_ctx.error_type.__name__
            if error_type not in error_patterns:
                error_patterns[error_type] = {'count': 0, 'severities': {}}

            error_patterns[error_type]['count'] += 1
            severity = error_ctx.severity.value
            error_patterns[error_type]['severities'][severity] = error_patterns[error_type]['severities'].get(severity, 0) + 1

        return {
            'total_errors': total_errors,
            'recovered_errors': recovered,
            'failed_recoveries': failed,
            'recovery_rate_percent': recovery_rate,
            'memory_cleanups': self.recovery_stats['memory_cleanups'],
            'error_patterns': error_patterns,
            'most_common_errors': sorted(error_patterns.items(), key=lambda x: x[1]['count'], reverse=True)[:5]
        }


def resilient_evaluation_wrapper(evaluator: ResilientEvaluator):
    """Decorator to make evaluation functions resilient"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Extract context from arguments
            context = {}
            if args and hasattr(args[0], '__class__'):
                context['instance'] = args[0]

            # Try to extract dataset name from arguments
            for arg in args:
                if isinstance(arg, str) and ('dataset' in str(arg).lower() or 'data' in str(arg).lower()):
                    context['dataset_name'] = arg
                    break

            return evaluator.resilient_operation(func, context, *args, **kwargs)

        return wrapper
    return decorator


class ProgressiveEvaluation:
    """Progressive evaluation that adapts to system capabilities"""

    def __init__(self, initial_batch_size=4, min_batch_size=1):
        self.initial_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.current_batch_size = initial_batch_size
        self.adaptive_settings = {
            'batch_size': initial_batch_size,
            'memory_threshold': 0.8,
            'timeout_seconds': 300
        }

    def adapt_to_system_load(self):
        """Adapt evaluation parameters based on current system load"""
        import psutil

        # Check system memory
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)

        # Adapt batch size based on memory pressure
        if memory.percent > 85:
            self.adaptive_settings['batch_size'] = max(self.min_batch_size, self.current_batch_size // 2)
            print(f"🔧 High memory usage ({memory.percent:.1f}%) - reducing batch size to {self.adaptive_settings['batch_size']}")
        elif memory.percent < 50 and self.current_batch_size < self.initial_batch_size:
            self.adaptive_settings['batch_size'] = min(self.initial_batch_size, self.current_batch_size * 2)
            print(f"🔧 Low memory usage ({memory.percent:.1f}%) - increasing batch size to {self.adaptive_settings['batch_size']}")

        # Adapt timeout based on CPU load
        if cpu_percent > 90:
            self.adaptive_settings['timeout_seconds'] = 600  # Increase timeout under high load
        else:
            self.adaptive_settings['timeout_seconds'] = 300  # Standard timeout

        self.current_batch_size = self.adaptive_settings['batch_size']

        return self.adaptive_settings


def create_resilient_evaluation_strategy(strategy_class, resilience_config=None):
    """Factory to create resilient evaluation strategies"""
    if resilience_config is None:
        resilience_config = {
            'max_retries': 3,
            'retry_delay': 2.0,
            'memory_cleanup_threshold': 0.8
        }

    class ResilientEvaluationStrategy(strategy_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.resilient_evaluator = ResilientEvaluator(**resilience_config)
            self.progressive_eval = ProgressiveEvaluation()

        def evaluate_dataset(self, dataset_name, config):
            """Override with resilient evaluation"""
            context = {
                'dataset_name': dataset_name,
                'config': config,
                'trainer': getattr(self, 'trainer', None)
            }

            return self.resilient_evaluator.resilient_operation(
                super().evaluate_dataset,
                context,
                dataset_name,
                config
            )

        def process_dataset(self, dataset_splits, num_files, dataset_name):
            """Override with adaptive processing"""
            # Adapt to current system conditions
            adaptive_settings = self.progressive_eval.adapt_to_system_load()

            # Update trainer batch size if available
            if hasattr(self, 'trainer') and hasattr(self.trainer.args, 'per_device_eval_batch_size'):
                self.trainer.args.per_device_eval_batch_size = adaptive_settings['batch_size']

            context = {
                'dataset_name': dataset_name,
                'file_count': num_files,
                'trainer': getattr(self, 'trainer', None)
            }

            return self.resilient_evaluator.resilient_operation(
                super().process_dataset,
                context,
                dataset_splits,
                num_files,
                dataset_name
            )

        def get_resilience_report(self):
            """Get report on resilience performance"""
            return self.resilient_evaluator.get_recovery_report()

    return ResilientEvaluationStrategy
