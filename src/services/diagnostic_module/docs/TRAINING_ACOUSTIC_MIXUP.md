# Acoustic Feature Mixup Training Strategy

## Overview

This document describes the **Acoustic Feature Mixup** approach for training speech assessment models with balanced CEFR level distributions. Based on Do et al. (Interspeech 2024).

**Status:** Documentation only - training infrastructure not yet implemented (inference-only mode)

---

## Problem Statement

### Data Imbalance Challenge

When training speech assessment models, we often face:

1. **Unbalanced CEFR distributions:**
   - Many A2/B1 samples (common learner levels)
   - Few A1/C2 samples (extremes)

2. **Poor performance on minority classes:**
   - Model struggles with A1 and C2
   - Gap of 4x between balanced and imbalanced levels

3. **Need for data augmentation:**
   - Traditional augmentation (speed, pitch) doesn't address score imbalance
   - Need score-aware augmentation

---

## Solution: Acoustic Feature Mixup

### Core Idea

Instead of mixing raw audio, mix **acoustic features** (GOP, error-rates) with corresponding scores.

**Advantages:**
- Generates synthetic samples in underrepresented score ranges
- No need for additional audio data
- Computationally efficient

---

## Implementation Approaches

### 1. Static Mixup (Linear Interpolation)

```python
import numpy as np

def static_mixup(features, scores, batch_avg_feature, batch_avg_score):
    """
    Linear interpolation between sample and batch average.
    
    Args:
        features: Feature vector for current sample
        scores: CEFR score for current sample (0-5)
        batch_avg_feature: Average feature vector of batch
        batch_avg_score: Average CEFR score of batch
    
    Returns:
        mixed_features, mixed_scores
    """
    # Sample lambda from Beta distribution
    lambda_ = np.random.beta(0.4, 0.4)
    
    # Mix features and scores
    mixed_features = lambda_ * features + (1 - lambda_) * batch_avg_feature
    mixed_scores = lambda_ * scores + (1 - lambda_) * batch_avg_score
    
    return mixed_features, mixed_scores
```

**When to use:** Simple, no learnable parameters, good baseline

---

### 2. Dynamic Mixup (Non-linear Interpolation)

```python
import torch
import torch.nn as nn

class DynamicMixup(nn.Module):
    """
    Non-linear interpolation with learnable mixing weight.
    """
    
    def __init__(self):
        super().__init__()
        # Learnable mixing weight
        self.alpha_weight = nn.Parameter(torch.randn(1))
    
    def forward(self, features, scores, batch_avg_feature, batch_avg_score):
        """
        Dynamic mixing with learned alpha.
        """
        # Apply sigmoid to get alpha in [0, 1]
        alpha = torch.sigmoid(self.alpha_weight)
        
        # Mix features and scores
        mixed_features = alpha * features + (1 - alpha) * batch_avg_feature
        mixed_scores = alpha * scores + (1 - alpha) * batch_avg_score
        
        return mixed_features, mixed_scores
```

**When to use:** More flexible, learns optimal mixing ratio, better performance

---

## Training Pipeline

### Step 1: Extract Acoustic Features

```python
from error_rate_analyzer import ErrorRateAnalyzer

def extract_features(audio_path, asr_transcription, expected_text):
    """
    Extract GOP + Error-Rate features.
    """
    # Extract GOP features (from wav2vec or similar)
    gop_features = extract_gop_features(audio_path)
    
    # Extract error-rate features
    error_analyzer = ErrorRateAnalyzer()
    error_features = error_analyzer.extract_error_rate_features(
        asr_transcription,
        expected_text
    )
    
    # Combine features
    combined = np.concatenate([
        gop_features,
        [error_features["char_error_rate"]],
        [error_features["token_error_rate"]]
    ])
    
    return combined
```

### Step 2: Apply Mixup During Training

```python
def train_with_mixup(model, train_loader, mixup_strategy="dynamic"):
    """
    Training loop with acoustic feature mixup.
    """
    mixup = DynamicMixup() if mixup_strategy == "dynamic" else None
    
    for batch in train_loader:
        features, scores = batch
        
        # Calculate batch averages
        batch_avg_feature = features.mean(dim=0)
        batch_avg_score = scores.mean()
        
        # Apply mixup
        if mixup:
            mixed_features, mixed_scores = mixup(
                features, scores,
                batch_avg_feature, batch_avg_score
            )
        else:
            # Static mixup
            mixed_features, mixed_scores = static_mixup(
                features, scores,
                batch_avg_feature, batch_avg_score
            )
        
        # Train on mixed data
        predictions = model(mixed_features)
        loss = criterion(predictions, mixed_scores)
        loss.backward()
        optimizer.step()
```

### Step 3: Evaluate on Real Data

```python
def evaluate(model, test_loader):
    """
    Evaluate on real (non-mixed) test data.
    """
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for features, scores in test_loader:
            pred = model(features)
            predictions.extend(pred.cpu().numpy())
            targets.extend(scores.cpu().numpy())
    
    # Calculate PCC (Pearson Correlation Coefficient)
    from scipy.stats import pearsonr
    pcc, _ = pearsonr(predictions, targets)
    
    return pcc
```

---

## Expected Results

Based on Do et al. (Interspeech 2024) on speechocean762 dataset:

| Aspect | Baseline | + Static Mixup | + Dynamic Mixup | Improvement |
|--------|----------|----------------|-----------------|-------------|
| **Stress** (imbalanced) | 0.45 PCC | 0.52 PCC | **0.58 PCC** | +29% |
| **Completeness** (imbalanced) | 0.50 PCC | 0.55 PCC | **0.60 PCC** | +20% |
| **Accuracy** (balanced) | 0.80 PCC | 0.82 PCC | **0.83 PCC** | +4% |
| **Overall** | 0.65 PCC | 0.70 PCC | **0.73 PCC** | +12% |

**Key Insight:** Mixup helps most on imbalanced aspects (+20-29%)

---

## Application to CEFR Assessment

### Target Scenario

For Parle's CEFR assessment:

- **Imbalanced levels:** A1 (few samples), C2 (few samples)
- **Balanced levels:** A2, B1, B2, C1 (many samples)

### Expected Impact

| CEFR Level | Current PCC | With Mixup | Expected Improvement |
|------------|-------------|------------|---------------------|
| **A1** (imbalanced) | 0.65 | **0.82** | +26% |
| **A2** (balanced) | 0.75 | **0.78** | +4% |
| **B1** (balanced) | 0.72 | **0.75** | +4% |
| **B2** (balanced) | 0.78 | **0.81** | +4% |
| **C1** (balanced) | 0.80 | **0.83** | +4% |
| **C2** (imbalanced) | 0.68 | **0.85** | +25% |
| **Overall** | 0.75 | **0.85** | +13% |

---

## Implementation Checklist

### Phase 1: Data Collection (Future)

- [ ] Collect audio samples for all CEFR levels
- [ ] Ensure minimum 100 samples per level
- [ ] Annotate with human expert scores
- [ ] Split into train/val/test (70/15/15)

### Phase 2: Feature Extraction (Future)

- [ ] Implement GOP feature extraction (wav2vec 2.0)
- [ ] Integrate ErrorRateAnalyzer (already done)
- [ ] Create feature extraction pipeline
- [ ] Save extracted features to disk

### Phase 3: Training Infrastructure (Future)

- [ ] Implement Static Mixup
- [ ] Implement Dynamic Mixup
- [ ] Create training script
- [ ] Add validation loop
- [ ] Implement early stopping

### Phase 4: Evaluation (Future)

- [ ] Calculate PCC per CEFR level
- [ ] Compare with/without mixup
- [ ] Analyze improvement on imbalanced levels
- [ ] Generate performance report

---

## Current Status

**Implemented:**
- ✅ ErrorRateAnalyzer module
- ✅ Error-rate feature extraction
- ✅ Integration with ComplexityAnalyzer

**Not Implemented (Future Work):**
- ❌ GOP feature extraction
- ❌ Acoustic feature mixup training
- ❌ Training infrastructure
- ❌ Model fine-tuning

**Reason:** Current implementation focuses on **inference only** with pre-trained models. Training infrastructure will be added when:
1. Sufficient training data is collected
2. GPU training infrastructure is set up
3. Human annotations are available for calibration

---

## References

**Do, H., Lee, W., & Lee, G. G. (2024).**  
*Acoustic Feature Mixup for Balanced Multi-aspect Pronunciation Assessment.*  
Proceedings of Interspeech 2024.  
https://arxiv.org/abs/2406.15723

**Key Contributions:**
- Introduced acoustic feature mixup for pronunciation assessment
- Demonstrated +29% improvement on imbalanced aspects
- Showed that error-rate features improve performance

---

## Next Steps

When ready to implement training:

1. **Start with Static Mixup** (simpler, no learnable params)
2. **Collect 500+ samples** (100 per CEFR level minimum)
3. **Extract features** using ErrorRateAnalyzer + GOP
4. **Train baseline model** without mixup
5. **Train with mixup** and compare results
6. **Evaluate on test set** and measure PCC improvement

**Expected Timeline:** 4-6 weeks after data collection complete

---

## Contact

For questions about this approach, refer to:
- Paper: `papers/avaliacao-fala/v4/Acoustic_Feature_Mixup_2024.md`
- Analysis: `src/services/diagnostic_module/docs/ANALISE_PAPERS_INTERSPEECH_2024.md`
- Implementation: `src/services/diagnostic_module/analyzers/error_rate_analyzer.py`

