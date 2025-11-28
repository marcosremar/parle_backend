"""
Pairwise Classification
Implements 15 binary classifiers (A1-A2, A1-B1, ..., B2-C2) for CEFR classification
Based on: Vajjala & Rama (2021), Arnold et al. (2018)
"""

import os
import json
import pickle
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import numpy as np
from loguru import logger

# Try to import scikit-learn (optional dependency)
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available. Pairwise classifiers will use heuristics.")


class PairwiseClassifier:
    """Pairwise binary classifier for CEFR level pairs"""
    
    # All 15 pairwise comparisons
    PAIRS = [
        ("A1", "A2"), ("A1", "B1"), ("A1", "B2"), ("A1", "C1"), ("A1", "C2"),
        ("A2", "B1"), ("A2", "B2"), ("A2", "C1"), ("A2", "C2"),
        ("B1", "B2"), ("B1", "C1"), ("B1", "C2"),
        ("B2", "C1"), ("B2", "C2"),
        ("C1", "C2")
    ]
    
    def __init__(self, models_dir: Optional[Path] = None):
        """
        Initialize pairwise classifier
        
        Args:
            models_dir: Directory containing trained models (optional)
        """
        if models_dir is None:
            project_root = Path(__file__).parent.parent.parent
            models_dir = project_root / "data" / "models" / "pairwise_classifiers"
        
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.classifiers = {}
        self.scalers = {}
        self._load_models()
    
    def _load_models(self):
        """Load pre-trained models if available"""
        for level1, level2 in self.PAIRS:
            pair_name = f"{level1}_{level2}"
            model_path = self.models_dir / f"{pair_name}_model.pkl"
            scaler_path = self.models_dir / f"{pair_name}_scaler.pkl"
            
            if model_path.exists() and scaler_path.exists():
                try:
                    with open(model_path, 'rb') as f:
                        self.classifiers[pair_name] = pickle.load(f)
                    with open(scaler_path, 'rb') as f:
                        self.scalers[pair_name] = pickle.load(f)
                    logger.info(f"✅ Loaded model for {pair_name}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load model for {pair_name}: {e}")
    
    def extract_features(self, quantitative_metrics: Dict[str, Any], 
                        llm_confidence: Dict[str, float]) -> np.ndarray:
        """
        Extract features for pairwise classification
        
        Args:
            quantitative_metrics: Dictionary with quantitative metrics
            llm_confidence: Dictionary with LLM confidence scores per level
            
        Returns:
            Feature vector
        """
        features = []
        
        # Quantitative metrics
        basic_syntactic = quantitative_metrics.get("basic_syntactic", {})
        basic_lexical = quantitative_metrics.get("basic_lexical", {})
        lexical_div = quantitative_metrics.get("lexical_diversity", {})
        syntactic_comp = quantitative_metrics.get("syntactic_complexity", {})
        speech_feat = quantitative_metrics.get("speech_features", {})
        
        # MOST IMPORTANT: Word Tokens and Word Types (Arnold et al., 2018)
        # These should be the FIRST features as they are the most predictive
        features.append(basic_lexical.get("word_tokens", basic_lexical.get("total_words", 0)))
        features.append(basic_lexical.get("word_types", basic_lexical.get("vocabulary_size", 0)))
        features.append(basic_lexical.get("tokens_types_ratio", 0.0))
        
        # Basic syntactic features
        features.append(basic_syntactic.get("avg_words_per_sentence", 0.0))
        features.append(basic_syntactic.get("subordination_ratio", 0.0))
        features.append(basic_syntactic.get("passive_voice_ratio", 0.0))
        features.append(basic_syntactic.get("subjunctive_ratio", 0.0))
        features.append(basic_syntactic.get("yngve_mean_depth", 0.0))
        features.append(basic_syntactic.get("frazier_mean_depth", 0.0))
        features.append(basic_syntactic.get("subordination_index", 0.0))
        
        # Basic lexical features
        features.append(basic_lexical.get("type_token_ratio", 0.0))
        features.append(basic_lexical.get("avg_word_length", 0.0))
        features.append(basic_lexical.get("mtld", 0.0))
        features.append(basic_lexical.get("mattr", 0.0))
        features.append(basic_lexical.get("zipf_ttr", 0.0))
        features.append(basic_lexical.get("hapax_percentage", 0.0))
        
        # Advanced syntactic features
        features.append(syntactic_comp.get("yngve_mean_depth", 0.0))
        features.append(syntactic_comp.get("frazier_mean_depth", 0.0))
        features.append(syntactic_comp.get("num_t_units", 0))
        features.append(syntactic_comp.get("avg_words_per_tunit", 0.0))
        features.append(syntactic_comp.get("subordination_index", 0.0))
        
        # Speech features
        mws = speech_feat.get("mean_word_span", {})
        features.append(mws.get("mws", 0.0) if isinstance(mws, dict) else 0.0)
        
        reps = speech_feat.get("repetitions", {})
        features.append(reps.get("repetition_rate", 0.0) if isinstance(reps, dict) else 0.0)
        
        disflu = speech_feat.get("disfluencies", {})
        features.append(disflu.get("disfluency_rate", 0.0) if isinstance(disflu, dict) else 0.0)
        
        # LLM confidence scores (6 levels)
        levels = ["A1", "A2", "B1", "B2", "C1", "C2"]
        for level in levels:
            features.append(llm_confidence.get(level, 0.0))
        
        return np.array(features, dtype=np.float32)
    
    def classify_pair(self, features: np.ndarray, level1: str, level2: str) -> Tuple[str, float]:
        """
        Classify text as belonging to level1 or level2
        
        Args:
            features: Feature vector
            level1: First level
            level2: Second level
            
        Returns:
            (predicted_level, confidence)
        """
        pair_name = f"{level1}_{level2}"
        
        # Check if model exists
        if pair_name in self.classifiers and SKLEARN_AVAILABLE:
            try:
                # Scale features
                scaler = self.scalers.get(pair_name)
                if scaler:
                    features_scaled = scaler.transform(features.reshape(1, -1))
                else:
                    features_scaled = features.reshape(1, -1)
                
                # Predict
                classifier = self.classifiers[pair_name]
                proba = classifier.predict_proba(features_scaled)[0]
                
                # Return level with higher probability
                if proba[1] > proba[0]:
                    return level2, float(proba[1])
                else:
                    return level1, float(proba[0])
            except Exception as e:
                logger.warning(f"⚠️ Error in pairwise classification for {pair_name}: {e}")
        
        # Fallback: heuristic based on features
        return self._heuristic_classify(features, level1, level2)
    
    def _heuristic_classify(self, features: np.ndarray, level1: str, level2: str) -> Tuple[str, float]:
        """
        Heuristic classification when model is not available
        
        Uses simple thresholds based on feature values
        """
        # Level order for comparison
        level_order = {"A1": 1, "A2": 2, "B1": 3, "B2": 4, "C1": 5, "C2": 6}
        level1_order = level_order.get(level1, 0)
        level2_order = level_order.get(level2, 0)
        
        # Higher level should have higher complexity
        if level2_order > level1_order:
            # Use features that indicate higher complexity
            # avg_words_per_sentence (index 0), subordination (1), yngve (4), mtld (10)
            complexity_score = (
                features[0] * 0.3 +  # sentence length
                features[1] * 0.3 +  # subordination
                features[4] * 0.2 +  # yngve depth
                features[10] * 0.2   # MTLD
            )
            
            # Threshold: if complexity > 0.5, predict level2
            if complexity_score > 0.5:
                return level2, 0.6
            else:
                return level1, 0.6
        else:
            # level1 > level2 (shouldn't happen in our pairs, but handle it)
            return level1, 0.5
    
    def classify_all_pairs(self, features: np.ndarray) -> Dict[str, Tuple[str, float]]:
        """
        Classify text using all pairwise comparisons
        
        Args:
            features: Feature vector
            
        Returns:
            Dictionary mapping pair names to (predicted_level, confidence)
        """
        results = {}
        
        for level1, level2 in self.PAIRS:
            predicted, confidence = self.classify_pair(features, level1, level2)
            pair_name = f"{level1}_{level2}"
            results[pair_name] = (predicted, confidence)
        
        return results
    
    def aggregate_votes(self, pair_results: Dict[str, Tuple[str, float]]) -> Dict[str, float]:
        """
        Aggregate votes from all pairwise comparisons
        
        Args:
            pair_results: Results from classify_all_pairs
            
        Returns:
            Dictionary with vote counts per level
        """
        votes = {"A1": 0.0, "A2": 0.0, "B1": 0.0, "B2": 0.0, "C1": 0.0, "C2": 0.0}
        
        for pair_name, (predicted, confidence) in pair_results.items():
            votes[predicted] += confidence
        
        # Normalize to probabilities
        total = sum(votes.values())
        if total > 0:
            for level in votes:
                votes[level] = votes[level] / total
        
        return votes
    
    def predict_level(self, quantitative_metrics: Dict[str, Any], 
                     llm_confidence: Dict[str, float]) -> Dict[str, Any]:
        """
        Predict CEFR level using pairwise classification
        
        Args:
            quantitative_metrics: Quantitative metrics from analyze_text_features_quantitative
            llm_confidence: LLM confidence scores per level
            
        Returns:
            Dictionary with predicted level and vote distribution
        """
        # Extract features
        features = self.extract_features(quantitative_metrics, llm_confidence)
        
        # Classify all pairs
        pair_results = self.classify_all_pairs(features)
        
        # Aggregate votes
        votes = self.aggregate_votes(pair_results)
        
        # Predicted level is the one with most votes
        predicted_level = max(votes.items(), key=lambda x: x[1])[0]
        confidence = votes[predicted_level]
        
        return {
            "predicted_level": predicted_level,
            "confidence": confidence,
            "vote_distribution": votes,
            "pair_results": pair_results
        }
    
    def train_model(self, pair_name: str, X: np.ndarray, y: np.ndarray):
        """
        Train a pairwise classifier (for future use with training data)
        
        Args:
            pair_name: Name of the pair (e.g., "A1_A2")
            X: Feature matrix
            y: Labels (0 for first level, 1 for second level)
        """
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available. Cannot train models.")
            return
        
        try:
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train classifier
            classifier = LogisticRegression(max_iter=1000, random_state=42)
            classifier.fit(X_scaled, y)
            
            # Save model
            model_path = self.models_dir / f"{pair_name}_model.pkl"
            scaler_path = self.models_dir / f"{pair_name}_scaler.pkl"
            
            with open(model_path, 'wb') as f:
                pickle.dump(classifier, f)
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            
            self.classifiers[pair_name] = classifier
            self.scalers[pair_name] = scaler
            
            logger.info(f"✅ Trained and saved model for {pair_name}")
        except Exception as e:
            logger.error(f"❌ Error training model for {pair_name}: {e}")

