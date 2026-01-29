#!/usr/bin/env python3
"""
Multi-model ensemble evaluation for transformer model comparison.
This module enables simultaneous evaluation of multiple models and ensemble methods.
"""
import numpy as np
import time
import json
import os
import sys
from typing import List, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Fix imports to work when called as module
try:
    from .evaluation_strategy_factory import run_evaluation, EvaluationStrategyFactory
    from .evaluation_common import calculate_performance_metrics
except ImportError:
    from evaluation_strategy_factory import run_evaluation, EvaluationStrategyFactory
    from evaluation_common import calculate_performance_metrics


@dataclass
class ModelConfig:
    """Configuration for a single model in ensemble"""
    name: str
    model_path: str
    weight: float = 1.0
    strategy_type: str = "file"
    confidence_threshold: float = 0.5


class EnsembleMethod:
    """Base class for ensemble methods"""

    def combine_predictions(self, predictions_list: List[np.ndarray], weights: Optional[List[float]] = None) -> np.ndarray:
        raise NotImplementedError


class VotingEnsemble(EnsembleMethod):
    """Majority voting ensemble"""

    def combine_predictions(self, predictions_list: List[np.ndarray], weights: Optional[List[float]] = None) -> np.ndarray:
        if weights is None:
            weights = [1.0] * len(predictions_list)

        # Convert to binary predictions if needed
        binary_preds = []
        for preds in predictions_list:
            if preds.ndim > 1:  # Logits or probabilities
                binary_preds.append((preds[:, 1] > 0.5).astype(int))
            else:
                binary_preds.append(preds.astype(int))

        # Weighted voting
        weighted_votes = np.zeros_like(binary_preds[0], dtype=float)
        total_weight = sum(weights)

        for preds, weight in zip(binary_preds, weights):
            weighted_votes += preds * weight

        return (weighted_votes / total_weight > 0.5).astype(int)


class AveragingEnsemble(EnsembleMethod):
    """Probability averaging ensemble"""

    def combine_predictions(self, predictions_list: List[np.ndarray], weights: Optional[List[float]] = None) -> np.ndarray:
        if weights is None:
            weights = [1.0] * len(predictions_list)

        # Extract probabilities
        probabilities = []
        for preds in predictions_list:
            if preds.ndim > 1:  # Logits - convert to probabilities
                from scipy.special import softmax
                probs = softmax(preds, axis=1)
                probabilities.append(probs[:, 1])  # P(drone)
            else:
                probabilities.append(preds)  # Already probabilities

        # Weighted average
        weighted_avg = np.zeros_like(probabilities[0])
        total_weight = sum(weights)

        for probs, weight in zip(probabilities, weights):
            weighted_avg += probs * weight

        weighted_avg /= total_weight
        return (weighted_avg > 0.5).astype(int)


class StackingEnsemble(EnsembleMethod):
    """Stacking ensemble with meta-learner"""

    def __init__(self):
        self.meta_learner = None
        self.is_trained = False

    def train_meta_learner(self, predictions_list: List[np.ndarray], true_labels: np.ndarray):
        """Train meta-learner on predictions from base models"""
        from sklearn.linear_model import LogisticRegression

        # Stack predictions as features
        X = np.column_stack([
            preds[:, 1] if preds.ndim > 1 else preds
            for preds in predictions_list
        ])

        self.meta_learner = LogisticRegression()
        self.meta_learner.fit(X, true_labels)
        self.is_trained = True

    def combine_predictions(self, predictions_list: List[np.ndarray], weights: Optional[List[float]] = None) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Meta-learner must be trained before making predictions")

        # Stack predictions as features
        X = np.column_stack([
            preds[:, 1] if preds.ndim > 1 else preds
            for preds in predictions_list
        ])

        return self.meta_learner.predict(X)


class WeightedEnsemble(EnsembleMethod):
    """Dynamic weighted ensemble that adjusts weights based on model confidence"""

    def __init__(self, confidence_threshold=0.7):
        self.confidence_threshold = confidence_threshold
        self.adaptive_weights = True

    def combine_predictions(self, predictions_list: List[np.ndarray], weights: Optional[List[float]] = None) -> np.ndarray:
        if weights is None:
            weights = [1.0] * len(predictions_list)

        # Extract probabilities
        probabilities = []
        confidences = []

        for preds in predictions_list:
            if preds.ndim > 1:  # Logits - convert to probabilities
                from scipy.special import softmax
                probs = softmax(preds, axis=1)
                prob_values = probs[:, 1]  # P(drone)
            else:
                prob_values = preds  # Already probabilities

            probabilities.append(prob_values)
            # Calculate confidence as distance from decision boundary
            confidences.append(np.abs(prob_values - 0.5))

        # Adaptive weighting based on confidence
        if self.adaptive_weights:
            final_predictions = []
            for i in range(len(probabilities[0])):
                sample_confidences = [conf[i] for conf in confidences]
                sample_probs = [prob[i] for prob in probabilities]

                # Weight models more heavily when they are confident
                adaptive_weights = []
                for j, conf in enumerate(sample_confidences):
                    if conf > self.confidence_threshold:
                        adaptive_weights.append(weights[j] * 2.0)  # Boost confident predictions
                    else:
                        adaptive_weights.append(weights[j] * 0.5)  # Reduce uncertain predictions

                # Normalize weights
                total_weight = sum(adaptive_weights)
                if total_weight > 0:
                    adaptive_weights = [w / total_weight for w in adaptive_weights]
                else:
                    adaptive_weights = [1.0 / len(weights)] * len(weights)

                # Weighted average for this sample
                weighted_prob = sum(prob * weight for prob, weight in zip(sample_probs, adaptive_weights))
                final_predictions.append(1 if weighted_prob > 0.5 else 0)

            return np.array(final_predictions)
        else:
            # Standard weighted average
            weighted_avg = np.zeros_like(probabilities[0])
            total_weight = sum(weights)

            for probs, weight in zip(probabilities, weights):
                weighted_avg += probs * weight

            weighted_avg /= total_weight
            return (weighted_avg > 0.5).astype(int)


class MetaLearnerEnsemble(EnsembleMethod):
    """Advanced meta-learner ensemble with feature engineering"""

    def __init__(self, meta_model_type='xgboost'):
        self.meta_model_type = meta_model_type
        self.meta_learner = None
        self.is_trained = False
        self.feature_names = []

    def _engineer_features(self, predictions_list: List[np.ndarray]) -> np.ndarray:
        """Engineer features from base model predictions"""
        features = []

        # Extract probabilities
        probabilities = []
        for preds in predictions_list:
            if preds.ndim > 1:
                from scipy.special import softmax
                probs = softmax(preds, axis=1)
                probabilities.append(probs[:, 1])
            else:
                probabilities.append(preds)

        probabilities = np.array(probabilities).T  # Shape: (n_samples, n_models)

        # Basic probability features
        features.append(probabilities)  # Raw probabilities
        self.feature_names.extend([f'model_{i}_prob' for i in range(len(predictions_list))])

        # Statistical features
        features.append(np.mean(probabilities, axis=1, keepdims=True))  # Mean probability
        features.append(np.std(probabilities, axis=1, keepdims=True))   # Std of probabilities
        features.append(np.max(probabilities, axis=1, keepdims=True))   # Max probability
        features.append(np.min(probabilities, axis=1, keepdims=True))   # Min probability
        self.feature_names.extend(['prob_mean', 'prob_std', 'prob_max', 'prob_min'])

        # Confidence features
        confidences = np.abs(probabilities - 0.5)
        features.append(np.mean(confidences, axis=1, keepdims=True))    # Mean confidence
        features.append(np.max(confidences, axis=1, keepdims=True))     # Max confidence
        self.feature_names.extend(['conf_mean', 'conf_max'])

        # Agreement features
        predictions_binary = (probabilities > 0.5).astype(int)
        agreement = np.mean(predictions_binary, axis=1, keepdims=True)  # Agreement ratio
        features.append(agreement)
        self.feature_names.append('agreement_ratio')

        # Disagreement features
        disagreement = np.std(predictions_binary.astype(float), axis=1, keepdims=True)
        features.append(disagreement)
        self.feature_names.append('disagreement_std')

        return np.concatenate(features, axis=1)

    def train_meta_learner(self, predictions_list: List[np.ndarray], true_labels: np.ndarray):
        """Train meta-learner with engineered features"""
        X = self._engineer_features(predictions_list)

        if self.meta_model_type == 'xgboost':
            try:
                import xgboost as xgb
                self.meta_learner = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42
                )
            except ImportError:
                print("XGBoost not available, falling back to Random Forest")
                from sklearn.ensemble import RandomForestClassifier
                self.meta_learner = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                )
        elif self.meta_model_type == 'random_forest':
            from sklearn.ensemble import RandomForestClassifier
            self.meta_learner = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        else:  # Default to logistic regression
            from sklearn.linear_model import LogisticRegression
            self.meta_learner = LogisticRegression(random_state=42)

        self.meta_learner.fit(X, true_labels)
        self.is_trained = True

        # Print feature importance if available
        if hasattr(self.meta_learner, 'feature_importances_'):
            importances = self.meta_learner.feature_importances_
            feature_importance = list(zip(self.feature_names, importances))
            feature_importance.sort(key=lambda x: x[1], reverse=True)

            print("Top 5 most important features for meta-learner:")
            for name, importance in feature_importance[:5]:
                print(f"  {name}: {importance:.4f}")

    def combine_predictions(self, predictions_list: List[np.ndarray], weights: Optional[List[float]] = None) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Meta-learner must be trained before making predictions")

        X = self._engineer_features(predictions_list)
        return self.meta_learner.predict(X)


class MultiModelEvaluator:
    """Evaluates multiple models and their ensembles"""

    def __init__(self, models: List[ModelConfig], ensemble_methods: Optional[List[EnsembleMethod]] = None,
                 output_base_dir: str = "./multi_model_results",
                 strategy_params: Optional[dict] = None,
                 calibration_options: Optional[dict] = None):
        self.models = models
        self.ensemble_methods = ensemble_methods or [
            VotingEnsemble(),
            AveragingEnsemble()
        ]
        self.individual_results = {}
        self.ensemble_results = {}
        self.output_base_dir = output_base_dir
        self.strategy_params = strategy_params or {}
        self.calibration_options = calibration_options or {
            'calibrate': False,
            'calibration_key': 'Calibration',
            'force_recalibrate': False,
            'calibration_fraction': 1.0
        }

    def evaluate_individual_models(self, parallel=True, max_workers=2):
        """Evaluate each model individually"""
        print(f"\n{'='*60}")
        print("MULTI-MODEL INDIVIDUAL EVALUATION")
        print(f"{'='*60}")

        if parallel and len(self.models) > 1:
            return self._evaluate_parallel(max_workers)
        else:
            return self._evaluate_sequential()

    def _evaluate_sequential(self):
        """Sequential evaluation of models"""

        for i, model_config in enumerate(self.models):
            print(f"\n[{i+1}/{len(self.models)}] Evaluating {model_config.name}...")

            try:
                start_time = time.time()

                model_output_dir = f"{self.output_base_dir}/{model_config.name}"

                # Ensure output directory exists
                os.makedirs(model_output_dir, exist_ok=True)
                print(f"  📁 Output directory: {model_output_dir}")

                # Create strategy with calibration support
                strategy = EvaluationStrategyFactory.create_strategy(
                    model_config.strategy_type,
                    model_config.model_path,
                    output_dir=model_output_dir,
                    **self.strategy_params
                )
                strategy.setup()

                # Verify output_dir is set on strategy
                if hasattr(strategy, 'output_dir'):
                    print(f"  📁 Strategy output_dir: {strategy.output_dir}")
                else:
                    print(f"  ⚠️ Strategy has no output_dir attribute, setting it manually")
                    strategy.output_dir = model_output_dir

                # Handle calibration
                default_threshold_path = os.path.join(model_config.model_path, 'threshold.json')

                if self.calibration_options.get('calibrate', False):
                    if not self.calibration_options.get('force_recalibrate', False):
                        loaded = strategy.load_saved_threshold(default_threshold_path)
                        if loaded:
                            print(f"  ✓ Using saved calibrated threshold: {strategy.threshold:.6f}")
                        else:
                            print(f"  🎯 Calibrating {model_config.name}...")
                            strategy.calibrate_threshold(
                                calibration_key=self.calibration_options.get('calibration_key', 'Calibration'),
                                threshold_file=default_threshold_path,
                                calibration_fraction=self.calibration_options.get('calibration_fraction', 1.0)
                            )
                    else:
                        print(f"  🎯 Force recalibrating {model_config.name}...")
                        strategy.calibrate_threshold(
                            calibration_key=self.calibration_options.get('calibration_key', 'Calibration'),
                            threshold_file=default_threshold_path,
                            calibration_fraction=self.calibration_options.get('calibration_fraction', 1.0)
                        )
                else:
                    # Try to load existing threshold
                    _ = strategy.load_saved_threshold(default_threshold_path)

                # Run evaluation
                print(f"  📊 Running evaluation for {model_config.name}...")
                results = strategy.evaluate_all_datasets()

                print(f"  💾 Generating outputs to: {model_output_dir}")
                print(f"  📁 Strategy output_dir is: {strategy.output_dir}")
                strategy.generate_outputs(results)
                print(f"  ✅ Outputs generated for {model_config.name}")

                eval_time = time.time() - start_time

                self.individual_results[model_config.name] = {
                    'results': results,
                    'evaluation_time': eval_time,
                    'model_config': model_config,
                    'threshold': getattr(strategy, 'threshold', 0.5)
                }

                print(f"✅ {model_config.name} completed in {eval_time:.1f}s (threshold: {strategy.threshold:.4f})")

            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"❌ {model_config.name} failed: {e}")
                self.individual_results[model_config.name] = {
                    'error': str(e),
                    'model_config': model_config
                }

        return self.individual_results

    def _evaluate_parallel(self, max_workers):
        """Parallel evaluation of models"""

        print(f"Running {len(self.models)} models in parallel (max {max_workers} workers)...")

        def evaluate_model(model_config):
            try:
                start_time = time.time()

                model_output_dir = f"{self.output_base_dir}/{model_config.name}"

                # Ensure output directory exists
                os.makedirs(model_output_dir, exist_ok=True)

                # Create strategy with calibration support
                strategy = EvaluationStrategyFactory.create_strategy(
                    model_config.strategy_type,
                    model_config.model_path,
                    output_dir=model_output_dir,
                    **self.strategy_params
                )
                strategy.setup()

                # Verify output_dir is set on strategy
                if not hasattr(strategy, 'output_dir') or strategy.output_dir != model_output_dir:
                    strategy.output_dir = model_output_dir

                # Handle calibration
                default_threshold_path = os.path.join(model_config.model_path, 'threshold.json')
                threshold_used = 0.5

                if self.calibration_options.get('calibrate', False):
                    if not self.calibration_options.get('force_recalibrate', False):
                        loaded = strategy.load_saved_threshold(default_threshold_path)
                        if loaded:
                            print(f"  ✓ [{model_config.name}] Using saved threshold: {strategy.threshold:.6f}")
                        else:
                            print(f"  🎯 [{model_config.name}] Calibrating...")
                            strategy.calibrate_threshold(
                                calibration_key=self.calibration_options.get('calibration_key', 'Calibration'),
                                threshold_file=default_threshold_path,
                                calibration_fraction=self.calibration_options.get('calibration_fraction', 1.0)
                            )
                    else:
                        print(f"  🎯 [{model_config.name}] Force recalibrating...")
                        strategy.calibrate_threshold(
                            calibration_key=self.calibration_options.get('calibration_key', 'Calibration'),
                            threshold_file=default_threshold_path,
                            calibration_fraction=self.calibration_options.get('calibration_fraction', 1.0)
                        )
                else:
                    _ = strategy.load_saved_threshold(default_threshold_path)

                threshold_used = getattr(strategy, 'threshold', 0.5)

                # Run evaluation
                results = strategy.evaluate_all_datasets()
                strategy.generate_outputs(results)

                eval_time = time.time() - start_time

                return model_config.name, {
                    'results': results,
                    'evaluation_time': eval_time,
                    'model_config': model_config,
                    'threshold': threshold_used
                }
            except Exception as e:
                import traceback
                traceback.print_exc()
                return model_config.name, {
                    'error': str(e),
                    'model_config': model_config
                }

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(evaluate_model, model): model.name for model in self.models}

            for future in as_completed(futures):
                model_name = futures[future]
                try:
                    name, result = future.result()
                    self.individual_results[name] = result

                    if 'error' in result:
                        print(f"❌ {name} failed: {result['error']}")
                    else:
                        print(f"✅ {name} completed in {result['evaluation_time']:.1f}s")

                except Exception as e:
                    print(f"❌ {model_name} crashed: {e}")
                    self.individual_results[model_name] = {'error': str(e)}

        return self.individual_results

    def evaluate_ensembles(self):
        """Evaluate ensemble methods using individual model results"""
        print(f"\n{'='*60}")
        print("ENSEMBLE EVALUATION")
        print(f"{'='*60}")

        # Check if we have enough successful individual results
        successful_models = [
            name for name, result in self.individual_results.items()
            if 'results' in result and result['results']
        ]

        if len(successful_models) < 2:
            print(f"❌ Need at least 2 successful models for ensemble, got {len(successful_models)}")
            return {}

        print(f"Creating ensembles from {len(successful_models)} models: {successful_models}")

        # Collect predictions from all successful models
        ensemble_predictions = {}
        common_datasets = None

        for model_name in successful_models:
            model_results = self.individual_results[model_name]['results']
            model_predictions = {}

            for dataset_name, dataset_result in model_results.items():
                if 'file_predictions' in dataset_result and 'file_labels' in dataset_result:
                    model_predictions[dataset_name] = {
                        'predictions': np.array(dataset_result['file_predictions']),
                        'labels': np.array(dataset_result['file_labels']),
                        'probabilities': np.array(dataset_result.get('probabilities', []))
                    }

            if model_predictions:
                ensemble_predictions[model_name] = model_predictions

                # Find common datasets across all models
                if common_datasets is None:
                    common_datasets = set(model_predictions.keys())
                else:
                    common_datasets &= set(model_predictions.keys())

        if not common_datasets:
            print("❌ No common datasets found across models")
            return {}

        print(f"Found {len(common_datasets)} common datasets: {list(common_datasets)}")

        # Evaluate each ensemble method
        for ensemble_method in self.ensemble_methods:
            method_name = ensemble_method.__class__.__name__
            print(f"\nEvaluating {method_name}...")

            ensemble_results = {}

            for dataset_name in common_datasets:
                print(f"  Processing {dataset_name}...")

                # Collect predictions for this dataset from all models
                dataset_predictions = []
                weights = []
                true_labels = None

                for model_name in successful_models:
                    if dataset_name in ensemble_predictions[model_name]:
                        preds = ensemble_predictions[model_name][dataset_name]['predictions']
                        dataset_predictions.append(preds)

                        # Get model weight
                        model_config = self.individual_results[model_name]['model_config']
                        weights.append(model_config.weight)

                        # Set true labels (should be same across all models)
                        if true_labels is None:
                            true_labels = ensemble_predictions[model_name][dataset_name]['labels']

                if len(dataset_predictions) >= 2:
                    try:
                        # Train stacking ensemble if needed
                        if isinstance(ensemble_method, StackingEnsemble) and not ensemble_method.is_trained:
                            # Use a subset for training meta-learner
                            train_size = min(len(true_labels) // 2, 1000)
                            if train_size > 10:
                                train_preds = [p[:train_size] for p in dataset_predictions]
                                ensemble_method.train_meta_learner(train_preds, true_labels[:train_size])

                        # Get ensemble predictions
                        ensemble_preds = ensemble_method.combine_predictions(dataset_predictions, weights)

                        # Calculate metrics
                        metrics = calculate_performance_metrics(true_labels, ensemble_preds, ensemble_preds.astype(float))
                        ensemble_results[dataset_name] = metrics

                        print(f"    {method_name} F1: {metrics['f1']:.3f}")

                    except Exception as e:
                        print(f"    ❌ {method_name} failed for {dataset_name}: {e}")
                        continue

            if ensemble_results:
                self.ensemble_results[method_name] = ensemble_results

        return self.ensemble_results

    def generate_comparison_report(self, save_path="model_comparison_report.json"):
        """Generate comprehensive comparison report"""
        print(f"\n{'='*60}")
        print("GENERATING COMPARISON REPORT")
        print(f"{'='*60}")

        # Analyze individual model performance
        individual_summary = {}
        for model_name, result in self.individual_results.items():
            if 'results' in result:
                model_results = result['results']

                # Calculate average metrics across datasets
                f1_scores = []
                accuracies = []

                for dataset_result in model_results.values():
                    if 'file_level_f1' in dataset_result:
                        f1_scores.append(dataset_result['file_level_f1'])
                    if 'file_level_accuracy' in dataset_result:
                        accuracies.append(dataset_result['file_level_accuracy'])

                individual_summary[model_name] = {
                    'avg_f1': np.mean(f1_scores) if f1_scores else 0,
                    'avg_accuracy': np.mean(accuracies) if accuracies else 0,
                    'num_datasets': len(model_results),
                    'evaluation_time': result.get('evaluation_time', 0)
                }
            else:
                individual_summary[model_name] = {
                    'error': result.get('error', 'Unknown error'),
                    'evaluation_time': 0
                }

        # Analyze ensemble performance
        ensemble_summary = {}
        for method_name, ensemble_result in self.ensemble_results.items():
            f1_scores = [metrics['f1'] for metrics in ensemble_result.values()]
            accuracies = [metrics['accuracy'] for metrics in ensemble_result.values()]

            ensemble_summary[method_name] = {
                'avg_f1': np.mean(f1_scores) if f1_scores else 0,
                'avg_accuracy': np.mean(accuracies) if accuracies else 0,
                'num_datasets': len(ensemble_result)
            }

        # Find best performing approach
        all_approaches = {}
        all_approaches.update({f"Individual_{k}": v for k, v in individual_summary.items() if 'avg_f1' in v})
        all_approaches.update({f"Ensemble_{k}": v for k, v in ensemble_summary.items()})

        best_f1 = max(all_approaches.items(), key=lambda x: x[1].get('avg_f1', 0)) if all_approaches else None
        best_accuracy = max(all_approaches.items(), key=lambda x: x[1].get('avg_accuracy', 0)) if all_approaches else None

        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_models_evaluated': len(self.models),
            'successful_models': len([r for r in self.individual_results.values() if 'results' in r]),
            'ensemble_methods_tested': len(self.ensemble_methods),
            'individual_model_summary': individual_summary,
            'ensemble_summary': ensemble_summary,
            'best_performing': {
                'by_f1': best_f1,
                'by_accuracy': best_accuracy
            },
            'recommendations': self._generate_recommendations(individual_summary, ensemble_summary)
        }

        # Save report
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"📊 Comparison report saved to: {save_path}")

        # Print summary
        self._print_comparison_summary(report)

        return report

    def _generate_recommendations(self, individual_summary, ensemble_summary):
        """Generate performance recommendations"""
        recommendations = []

        # Find best individual model
        individual_models = {k: v for k, v in individual_summary.items() if 'avg_f1' in v}
        if individual_models:
            best_individual = max(individual_models.items(), key=lambda x: x[1]['avg_f1'])
            recommendations.append(f"Best individual model: {best_individual[0]} (F1: {best_individual[1]['avg_f1']:.3f})")

        # Compare ensemble vs individual
        if ensemble_summary and individual_models:
            best_ensemble_f1 = max(ensemble_summary.values(), key=lambda x: x['avg_f1'])['avg_f1']
            best_individual_f1 = max(individual_models.values(), key=lambda x: x['avg_f1'])['avg_f1']

            if best_ensemble_f1 > best_individual_f1:
                improvement = (best_ensemble_f1 - best_individual_f1) / best_individual_f1 * 100
                recommendations.append(f"Ensemble methods improve F1 by {improvement:.1f}% over best individual model")
            else:
                recommendations.append("Individual models perform better than ensembles - consider model selection over ensembling")

        # Efficiency recommendations
        efficient_models = [
            (k, v) for k, v in individual_summary.items()
            if 'avg_f1' in v and v.get('evaluation_time', float('inf')) < 60
        ]
        if efficient_models:
            efficient_models.sort(key=lambda x: x[1]['avg_f1'], reverse=True)
            recommendations.append(f"Most efficient high-performing model: {efficient_models[0][0]}")

        return recommendations

    def _print_comparison_summary(self, report):
        """Print a nice summary of the comparison results"""
        print(f"\n{'='*60}")
        print("MODEL COMPARISON SUMMARY")
        print(f"{'='*60}")

        print(f"Models evaluated: {report['total_models_evaluated']}")
        print(f"Successful evaluations: {report['successful_models']}")
        print(f"Ensemble methods tested: {report['ensemble_methods_tested']}")

        print(f"\n📊 INDIVIDUAL MODEL PERFORMANCE:")
        for model_name, metrics in report['individual_model_summary'].items():
            if 'avg_f1' in metrics:
                print(f"  {model_name}: F1={metrics['avg_f1']:.3f}, Acc={metrics['avg_accuracy']:.3f}, Time={metrics['evaluation_time']:.1f}s")
            else:
                print(f"  {model_name}: ❌ {metrics.get('error', 'Failed')}")

        if report['ensemble_summary']:
            print(f"\n🤝 ENSEMBLE PERFORMANCE:")
            for method_name, metrics in report['ensemble_summary'].items():
                print(f"  {method_name}: F1={metrics['avg_f1']:.3f}, Acc={metrics['avg_accuracy']:.3f}")

        if report['best_performing']['by_f1']:
            best = report['best_performing']['by_f1']
            print(f"\n🏆 BEST PERFORMING: {best[0]} (F1: {best[1]['avg_f1']:.3f})")

        if report['recommendations']:
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in report['recommendations']:
                print(f"  • {rec}")


def create_multi_model_config_from_checkpoints(checkpoint_dir: str, strategy_type: str = "file") -> List[ModelConfig]:
    """Auto-discover model checkpoints and create configs"""
    checkpoint_path = Path(checkpoint_dir)
    models = []

    if checkpoint_path.exists():
        # Look for checkpoint directories
        for checkpoint in checkpoint_path.glob("checkpoint-*"):
            if checkpoint.is_dir():
                model_name = f"Model_{checkpoint.name}"
                models.append(ModelConfig(
                    name=model_name,
                    model_path=str(checkpoint),
                    strategy_type=strategy_type
                ))

    return models
