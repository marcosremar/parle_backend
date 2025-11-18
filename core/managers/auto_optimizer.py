"""
Auto Optimizer
Automatic parameter tuning using Bayesian optimization
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from scipy.stats import norm
from scipy.optimize import minimize

from .profile_optimizer import ProfileOptimizer, OptimizationResult
from .gpu_profile_manager import get_gpu_profile_manager
from .structured_logger import get_logger
from src.core.exceptions import UltravoxError, wrap_exception

logger = logging.getLogger(__name__)
structured_logger = get_logger("AutoOptimizer")


@dataclass
class ParameterSpace:
    """Define parameter search space"""
    name: str
    min_value: float
    max_value: float
    is_integer: bool = False


@dataclass
class BayesianState:
    """State of Bayesian optimization"""
    iteration: int
    tested_params: List[Dict[str, float]]
    observed_scores: List[float]
    best_params: Dict[str, float]
    best_score: float


class AutoOptimizer:
    """
    Automatic optimizer using Bayesian optimization

    Uses Gaussian Process-based Bayesian optimization to efficiently
    search the parameter space and find optimal configurations.

    Features:
    - Bayesian optimization (more efficient than grid search)
    - Exploration vs exploitation balance
    - Automatic convergence detection
    - Multi-objective optimization support
    """

    def __init__(
        self,
        profile_id: str,
        exploration_factor: float = 0.2
    ):
        """
        Initialize auto optimizer

        Args:
            profile_id: GPU profile to optimize
            exploration_factor: Balance between exploration and exploitation (0-1)
        """
        self.profile_id = profile_id
        self.exploration_factor = exploration_factor

        self.profile_manager = get_gpu_profile_manager()
        self.optimizer = ProfileOptimizer(profile_id)

        # Get optimization config
        self.opt_config = self.profile_manager.get_optimization_config()

        # Define parameter space
        self.param_space = self._define_parameter_space()

        # Bayesian state
        self.state = BayesianState(
            iteration=0,
            tested_params=[],
            observed_scores=[],
            best_params={},
            best_score=0.0
        )

        structured_logger.info(
            f"🤖 AutoOptimizer initialized for profile '{profile_id}'",
            metadata={
                'exploration_factor': exploration_factor,
                'param_space_dims': len(self.param_space)
            }
        )

    def _define_parameter_space(self) -> List[ParameterSpace]:
        """Define parameter search space from config"""
        test_matrix = self.opt_config.get('test_matrix', {})

        param_space = []

        # GPU memory utilization for LLM
        llm_utils = test_matrix.get('gpu_memory_utilization', {}).get('llm', [0.65, 0.85])
        param_space.append(ParameterSpace(
            name='gpu_memory_utilization_llm',
            min_value=min(llm_utils),
            max_value=max(llm_utils),
            is_integer=False
        ))

        # GPU memory utilization for STT (if applicable)
        stt_utils = test_matrix.get('gpu_memory_utilization', {}).get('stt', [])
        if stt_utils:
            param_space.append(ParameterSpace(
                name='gpu_memory_utilization_stt',
                min_value=min(stt_utils),
                max_value=max(stt_utils),
                is_integer=False
            ))

        # Batch size
        batch_sizes = test_matrix.get('batch_sizes', [1, 32])
        param_space.append(ParameterSpace(
            name='batch_size',
            min_value=min(batch_sizes),
            max_value=max(batch_sizes),
            is_integer=True
        ))

        # Sequence length
        seq_lengths = test_matrix.get('sequence_lengths', [64, 1024])
        param_space.append(ParameterSpace(
            name='sequence_length',
            min_value=min(seq_lengths),
            max_value=max(seq_lengths),
            is_integer=True
        ))

        # Concurrency
        concurrency_levels = test_matrix.get('concurrent_requests', [1, 16])
        param_space.append(ParameterSpace(
            name='concurrency',
            min_value=min(concurrency_levels),
            max_value=max(concurrency_levels),
            is_integer=True
        ))

        return param_space

    def _normalize_params(self, params: Dict[str, float]) -> np.ndarray:
        """Normalize parameters to [0, 1] range"""
        normalized = []

        for param_def in self.param_space:
            value = params.get(param_def.name, param_def.min_value)
            norm_value = (value - param_def.min_value) / (param_def.max_value - param_def.min_value)
            normalized.append(norm_value)

        return np.array(normalized)

    def _denormalize_params(self, normalized: np.ndarray) -> Dict[str, float]:
        """Denormalize parameters from [0, 1] to original range"""
        params = {}

        for i, param_def in enumerate(self.param_space):
            value = normalized[i] * (param_def.max_value - param_def.min_value) + param_def.min_value

            if param_def.is_integer:
                value = int(round(value))

            params[param_def.name] = value

        return params

    def _gaussian_process_predict(
        self,
        x: np.ndarray,
        X_train: np.ndarray,
        y_train: np.ndarray,
        kernel_scale: float = 1.0,
        noise: float = 0.1
    ) -> Tuple[float, float]:
        """
        Simple Gaussian Process prediction

        Args:
            x: Point to predict at
            X_train: Training points
            y_train: Training values
            kernel_scale: RBF kernel scale
            noise: Observation noise

        Returns:
            (mean, std_dev) predictions
        """
        if len(X_train) == 0:
            return 0.5, 1.0  # Return high uncertainty for unexplored space

        # RBF kernel
        def kernel(x1, x2):
            return np.exp(-np.sum((x1 - x2) ** 2) / (2 * kernel_scale ** 2))

        # Compute kernel matrix
        K = np.zeros((len(X_train), len(X_train)))
        for i in range(len(X_train)):
            for j in range(len(X_train)):
                K[i, j] = kernel(X_train[i], X_train[j])

        K += noise * np.eye(len(X_train))  # Add noise

        # Kernel vector for test point
        k = np.array([kernel(x, x_train) for x_train in X_train])

        # GP prediction
        try:
            K_inv = np.linalg.inv(K)
            mean = k.T @ K_inv @ y_train
            var = kernel(x, x) - k.T @ K_inv @ k
            std = np.sqrt(max(var, 0))
        except Exception as e:
            # Fallback if matrix is singular
            mean = np.mean(y_train)
            std = np.std(y_train)

        return mean, std

    def _acquisition_function(
        self,
        x: np.ndarray,
        X_train: np.ndarray,
        y_train: np.ndarray,
        best_y: float
    ) -> float:
        """
        Upper Confidence Bound (UCB) acquisition function

        UCB = mean + exploration_factor * std

        Higher UCB = more promising to explore
        """
        mean, std = self._gaussian_process_predict(x, X_train, y_train)

        # UCB acquisition
        ucb = mean + self.exploration_factor * std

        return -ucb  # Negative for minimization

    def _suggest_next_params(self) -> Dict[str, float]:
        """Suggest next parameters to test using Bayesian optimization"""

        if len(self.state.tested_params) == 0:
            # First iteration: random sample
            params = {}
            for param_def in self.param_space:
                value = np.random.uniform(param_def.min_value, param_def.max_value)
                if param_def.is_integer:
                    value = int(round(value))
                params[param_def.name] = value

            structured_logger.info("🎲 First iteration: random sampling", metadata=params)
            return params

        # Convert to normalized arrays
        X_train = np.array([self._normalize_params(p) for p in self.state.tested_params])
        y_train = np.array(self.state.observed_scores) / 100.0  # Normalize scores to [0, 1]

        best_y = max(y_train)

        # Optimize acquisition function to find next best point
        best_acq = float('inf')
        best_x = None

        # Random search over acquisition function
        for _ in range(100):
            x_random = np.random.uniform(0, 1, len(self.param_space))

            acq_value = self._acquisition_function(x_random, X_train, y_train, best_y)

            if acq_value < best_acq:
                best_acq = acq_value
                best_x = x_random

        # Denormalize to get actual parameters
        next_params = self._denormalize_params(best_x)

        structured_logger.info(
            "🎯 Bayesian optimization suggests next parameters",
            metadata=next_params
        )

        return next_params

    async def optimize(
        self,
        max_iterations: int = 50,
        convergence_threshold: float = 0.01,
        min_iterations: int = 10
    ) -> Dict[str, Any]:
        """
        Run automatic optimization

        Args:
            max_iterations: Maximum optimization iterations
            convergence_threshold: Stop if improvement < threshold for N iterations
            min_iterations: Minimum iterations before convergence check

        Returns:
            Optimization results with best configuration
        """
        structured_logger.info(
            f"🚀 Starting automatic optimization",
            metadata={
                'profile': self.profile_id,
                'max_iterations': max_iterations,
                'exploration_factor': self.exploration_factor
            }
        )

        no_improvement_count = 0
        previous_best = 0.0

        for i in range(max_iterations):
            self.state.iteration = i + 1

            # Suggest next parameters
            test_params = self._suggest_next_params()

            # Run optimization iteration
            result = await self.optimizer.run_optimization_iteration(
                test_params,
                executor=None  # TODO: Connect to actual services
            )

            # Update state
            self.state.tested_params.append(test_params)
            self.state.observed_scores.append(result.quality_score)

            # Check if new best
            if result.quality_score > self.state.best_score:
                self.state.best_score = result.quality_score
                self.state.best_params = test_params

                structured_logger.info(
                    f"🎉 New best configuration found!",
                    metadata={
                        'iteration': i + 1,
                        'score': result.quality_score,
                        'params': test_params
                    }
                )

                no_improvement_count = 0
            else:
                no_improvement_count += 1

            # Log progress
            structured_logger.info(
                f"📊 Iteration {i+1}/{max_iterations}",
                metadata={
                    'current_score': result.quality_score,
                    'best_score': self.state.best_score,
                    'no_improvement_count': no_improvement_count
                }
            )

            # Convergence check (after minimum iterations)
            if i >= min_iterations:
                improvement = self.state.best_score - previous_best

                if improvement < convergence_threshold and no_improvement_count >= 5:
                    structured_logger.info(
                        f"✅ Converged after {i+1} iterations",
                        metadata={
                            'improvement': improvement,
                            'threshold': convergence_threshold
                        }
                    )
                    break

                previous_best = self.state.best_score

        # Final summary
        summary = {
            'profile_id': self.profile_id,
            'total_iterations': self.state.iteration,
            'best_score': self.state.best_score,
            'best_params': self.state.best_params,
            'convergence_history': {
                'tested_params': self.state.tested_params,
                'observed_scores': self.state.observed_scores
            },
            'best_configuration': self.optimizer.get_best_configuration()
        }

        structured_logger.info(
            f"🏁 Automatic optimization complete",
            metadata={
                'iterations': self.state.iteration,
                'best_score': self.state.best_score
            }
        )

        return summary

    def get_optimization_curve(self) -> Dict[str, List]:
        """Get optimization convergence curve"""
        return {
            'iterations': list(range(1, len(self.state.observed_scores) + 1)),
            'scores': self.state.observed_scores,
            'best_scores_cumulative': [
                max(self.state.observed_scores[:i+1])
                for i in range(len(self.state.observed_scores))
            ]
        }

    def get_parameter_importance(self) -> Dict[str, float]:
        """
        Estimate parameter importance based on variance

        Higher variance in scores across parameter values = more important
        """
        if len(self.state.tested_params) < 5:
            return {}

        importance = {}

        for param_def in self.param_space:
            param_name = param_def.name

            # Get all tested values for this parameter (skip if not present in all)
            try:
                param_values = [p[param_name] for p in self.state.tested_params]
            except KeyError:
                # Parameter not present in all tested configs (e.g., gpu_memory_utilization_stt in minimal_gpu)
                importance[param_name] = 0.0
                continue

            param_scores = self.state.observed_scores

            # Calculate correlation between parameter and score
            if len(set(param_values)) > 1:
                correlation = np.corrcoef(param_values, param_scores)[0, 1]
                importance[param_name] = abs(correlation)
            else:
                importance[param_name] = 0.0

        # Normalize to [0, 1]
        if importance:
            max_importance = max(importance.values())
            if max_importance > 0:
                importance = {k: v / max_importance for k, v in importance.items()}

        return importance

    def export_best_config_to_profile(self) -> bool:
        """
        Export best configuration back to GPU profile

        Returns:
            Success status
        """
        if not self.state.best_params:
            structured_logger.error("❌ No best configuration to export")
            return False

        try:
            # Get profile
            profile = self.profile_manager.get_profile(self.profile_id)

            if not profile:
                structured_logger.error(f"❌ Profile '{self.profile_id}' not found")
                return False

            # Update service configs with best parameters
            if 'gpu_memory_utilization_llm' in self.state.best_params:
                profile.services['llm'].gpu_memory_utilization = self.state.best_params['gpu_memory_utilization_llm']

            if 'gpu_memory_utilization_stt' in self.state.best_params:
                if 'stt' in profile.services:
                    profile.services['stt'].gpu_memory_utilization = self.state.best_params['gpu_memory_utilization_stt']

            # TODO: Update profile YAML file with new values

            structured_logger.info(
                "✅ Best configuration exported to profile",
                metadata={
                    'profile_id': self.profile_id,
                    'best_params': self.state.best_params,
                    'score': self.state.best_score
                }
            )

            return True

        except Exception as e:
            structured_logger.error("❌ Failed to export configuration", exception=e)
            return False
