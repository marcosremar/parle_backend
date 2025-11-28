"""
Calibration Manager
Calibrates model predictions with human annotations.
Based on Byun et al. (2025) - LLM-as-a-Grader.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.linear_model import LinearRegression
from loguru import logger


class CalibrationManager:
    """
    Calibrate model predictions with human annotations.
    
    Based on Byun et al. (2025) - LLM-as-a-Grader:
    - Learns correction weights from validation dataset
    - Applies calibration to adjust systematic biases
    - Improves alignment with human evaluators
    """
    
    def __init__(self, calibration_file: Optional[str] = None):
        """
        Initialize calibration manager.
        
        Args:
            calibration_file: Path to save/load calibration weights
                             If None, uses default location
        """
        if calibration_file is None:
            # Default location: data/calibration_weights.json
            project_root = Path(__file__).parent.parent.parent.parent
            calibration_file = project_root / "data" / "calibration_weights.json"
        
        self.calibration_file = Path(calibration_file)
        self.weights: Optional[Dict[str, float]] = None
        
        # Create data directory if it doesn't exist
        self.calibration_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing weights if available
        self._load_weights()
    
    def _load_weights(self) -> None:
        """Load calibration weights from file"""
        if self.calibration_file.exists():
            try:
                with open(self.calibration_file, 'r') as f:
                    data = json.load(f)
                    self.weights = data.get("weights", {})
                    logger.info(f"Loaded calibration weights from {self.calibration_file}")
            except Exception as e:
                logger.warning(f"Failed to load calibration weights: {e}")
                self.weights = None
        else:
            self.weights = None
            logger.info("No existing calibration weights found")
    
    def _save_weights(self) -> None:
        """Save calibration weights to file"""
        try:
            data = {
                "weights": self.weights,
                "version": "1.0"
            }
            with open(self.calibration_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved calibration weights to {self.calibration_file}")
        except Exception as e:
            logger.error(f"Failed to save calibration weights: {e}")
            raise
    
    def _calculate_cefr_distance(
        self, 
        predicted: str, 
        actual: str
    ) -> float:
        """
        Calculate distance between CEFR levels.
        
        Args:
            predicted: Predicted CEFR level (A1, A2, B1, B2, C1, C2)
            actual: Actual CEFR level
            
        Returns:
            Distance in levels (0 = same, 1 = adjacent, etc.)
        """
        levels = ["A1", "A2", "B1", "B2", "C1", "C2"]
        
        try:
            pred_idx = levels.index(predicted.upper())
            actual_idx = levels.index(actual.upper())
            return abs(pred_idx - actual_idx)
        except ValueError:
            # Invalid level, return max distance
            return 3.0
    
    def _get_model_prediction(
        self, 
        text: str, 
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get model prediction for text.
        
        This is a placeholder - in real implementation, would call
        the actual complexity analyzer.
        
        Args:
            text: Text to analyze
            user_id: User ID for AKT integration
            
        Returns:
            Model prediction dictionary
        """
        # TODO: Integrate with actual ComplexityAnalyzer
        # For now, return placeholder
        return {
            "level": "B1",
            "breakdown": {
                "fluency": 3.5,
                "grammar": 4.0,
                "vocabulary": 3.8,
                "coherence": 4.2
            }
        }
    
    def calibrate(
        self, 
        validation_dataset: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Learn calibration weights from validation dataset.
        
        Args:
            validation_dataset: List of validation samples with format:
                [
                    {
                        "text": "...",
                        "user_id": "...",
                        "human_cefr": "B1",
                        "human_scores": {
                            "fluency": 3.5,
                            "grammar": 4.0,
                            "vocabulary": 3.8,
                            "coherence": 4.2
                        }
                    },
                    ...
                ]
        
        Returns:
            Dictionary with calibration results
        """
        if not validation_dataset:
            logger.warning("Empty validation dataset")
            return {"status": "error", "message": "Empty dataset"}
        
        logger.info(f"Calibrating with {len(validation_dataset)} samples")
        
        errors = []
        
        for i, sample in enumerate(validation_dataset):
            try:
                # Get model prediction
                model_pred = self._get_model_prediction(
                    sample["text"], 
                    sample.get("user_id")
                )
                
                # Calculate CEFR error
                cefr_error = self._calculate_cefr_distance(
                    model_pred["level"], 
                    sample["human_cefr"]
                )
                
                # Calculate score errors for each aspect
                score_errors = {}
                for aspect in ["fluency", "grammar", "vocabulary", "coherence"]:
                    model_score = model_pred["breakdown"].get(aspect, 0.0)
                    human_score = sample["human_scores"].get(aspect, 0.0)
                    score_errors[aspect] = human_score - model_score
                
                errors.append({
                    "cefr_error": cefr_error,
                    "score_errors": score_errors
                })
                
            except Exception as e:
                logger.warning(f"Error processing sample {i}: {e}")
                continue
        
        if not errors:
            logger.error("No valid samples processed")
            return {"status": "error", "message": "No valid samples"}
        
        # Learn correction weights using linear regression
        self.weights = self._learn_correction_weights(errors)
        
        # Save weights
        self._save_weights()
        
        # Calculate statistics
        avg_cefr_error = np.mean([e["cefr_error"] for e in errors])
        avg_score_errors = {
            aspect: np.mean([e["score_errors"][aspect] for e in errors])
            for aspect in ["fluency", "grammar", "vocabulary", "coherence"]
        }
        
        return {
            "status": "success",
            "num_samples": len(errors),
            "avg_cefr_error": float(avg_cefr_error),
            "avg_score_errors": {k: float(v) for k, v in avg_score_errors.items()},
            "weights": self.weights
        }
    
    def _learn_correction_weights(
        self, 
        errors: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Learn correction weights from errors.
        
        Uses simple linear regression to learn offset corrections.
        
        Args:
            errors: List of error dictionaries
            
        Returns:
            Dictionary of correction weights per aspect
        """
        # Simple approach: average error per aspect
        # More sophisticated: could use linear regression with features
        
        weights = {}
        
        for aspect in ["fluency", "grammar", "vocabulary", "coherence"]:
            aspect_errors = [e["score_errors"][aspect] for e in errors]
            weights[aspect] = float(np.mean(aspect_errors))
        
        logger.info(f"Learned calibration weights: {weights}")
        
        return weights
    
    def apply_calibration(
        self, 
        raw_scores: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Apply learned calibration to raw scores.
        
        Args:
            raw_scores: Raw scores from model (e.g., {"fluency": 3.5, ...})
            
        Returns:
            Calibrated scores
        """
        if not self.weights:
            logger.debug("No calibration weights available, returning raw scores")
            return raw_scores
        
        calibrated = {}
        
        for aspect, score in raw_scores.items():
            if aspect in self.weights:
                # Apply correction (add offset)
                calibrated[aspect] = score + self.weights[aspect]
                # Clamp to valid range [0, 5]
                calibrated[aspect] = max(0.0, min(5.0, calibrated[aspect]))
            else:
                calibrated[aspect] = score
        
        return calibrated
    
    def has_weights(self) -> bool:
        """Check if calibration weights are available"""
        return self.weights is not None and len(self.weights) > 0
