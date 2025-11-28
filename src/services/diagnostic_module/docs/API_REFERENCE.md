# Speech Grader API Reference

**Service:** `speech_grader` (formerly `diagnostic_module`)  
**Port:** 8960 (default)  
**Base URL:** `http://localhost:8960`

---

## Overview

The Speech Grader service provides comprehensive analysis of student speech and text, including:

- **CEFR Level Estimation**: Multi-aspect assessment (fluency, grammar, vocabulary, coherence)
- **Error Analysis**: Grammar, vocabulary, and pronunciation errors
- **Acoustic Features**: Wav2Vec 2.0 embeddings for pronunciation assessment
- **Task Relevance**: SBERT-based semantic similarity
- **Structured Feedback**: Pedagogical feedback with strengths, weaknesses, and next steps
- **Session Dynamics**: Consistency, trajectory, and engagement analysis

---

## Endpoints

### 1. Health Check

**GET** `/health`

Check service health.

**Response:**
```json
{
  "status": "healthy",
  "service": "speech_grader"
}
```

---

### 2. Estimate CEFR Level

**POST** `/api/diagnostic/estimate_level`

Estimate CEFR level with optional multi-aspect analysis.

**Request Body:**
```json
{
  "text": "Eu gosto de estudar português",
  "language": "pt-BR",
  "user_id": "user_123",
  "question": "Por que você estuda português?",
  "exemplar": "Eu estudo português porque quero me comunicar melhor",
  "audio_path": "/path/to/audio.wav",
  "asr_transcription": "Eu gosto de estudar portugues",
  "expected_text": "Eu gosto de estudar português",
  "asr_metadata": {
    "text": "...",
    "words": [
      {"word": "Eu", "start": 0.0, "end": 0.2, "confidence": 0.98}
    ],
    "duration": 2.0
  }
}
```

**Response:**
```json
{
  "cefr_level": "B1",
  "confidence": 0.85,
  "indicators": ["...", "..."],
  "reasoning": "Análise automática baseada em...",
  "breakdown": {
    "fluency": 3.5,
    "grammar": 4.0,
    "vocabulary": 3.8,
    "coherence": 4.2
  },
  "error_rate_features": {
    "char_error_rate": 0.05,
    "token_error_rate": 0.0,
    "char_distance": 1,
    "token_distance": 0
  },
  "pronunciation_score": 0.95,
  "acoustic_features": [0.123, 0.456, ...],
  "acoustic_metadata": {
    "native_embedding_shape": [100, 768],
    "learner_embedding_shape": [100, 768],
    "fused_shape": [768]
  },
  "asr_metadata_analysis": {
    "avg_confidence": 0.94,
    "speech_rate_wpm": 120.0,
    "num_pauses": 0,
    "fluency_score": 0.88
  },
  "task_relevance": {
    "task_relevance": 0.85,
    "exemplar_similarity": 0.78
  },
  "topic_coverage": {
    "coverage_score": 0.85,
    "coverage_level": "good"
  },
  "feedback": {
    "strengths": ["Excelente gramática", "Boa fluência"],
    "weaknesses": [
      {
        "type": "pronunciation",
        "category": "pronunciation",
        "severity": "medium",
        "description": "Pronúncia precisa melhorar"
      }
    ],
    "next_steps": ["Praticar pronúncia", "Expandir vocabulário"],
    "priority": "medium",
    "estimated_time_minutes": 30
  },
  "akt_alignment": {
    "akt_estimated_level": "B1",
    "alignment_score": 0.90,
    "confidence_adjustment": 0.05
  }
}
```

**Features by Phase:**

- **Phase 1 (Error-Rate):** `error_rate_features`, `pronunciation_score`, `error_positions`
- **Phase 2 (Acoustic):** `acoustic_features`, `acoustic_metadata`
- **Phase 3 (ASR Metadata):** `asr_metadata_analysis`
- **Phase 3 (Task Relevance):** `task_relevance`, `topic_coverage`
- **Phase 3 (Feedback):** `feedback`
- **AKT Integration:** `akt_alignment`

---

### 3. Analyze Session

**POST** `/api/diagnostic/analyze_session`

Analyze complete session with aggregated metrics and dynamics.

**Request Body:**
```json
{
  "session_turns": [
    {
      "analysis": {
        "estimated_cefr_level": "B1",
        "confidence": 0.75,
        "errors": []
      }
    }
  ],
  "conversation_history": [],
  "user_id": "user_123"
}
```

**Response:**
```json
{
  "session_summary": "Sessão com 3 turnos. Média de 0.5 erros por turno.",
  "total_turns": 3,
  "total_errors": 1,
  "avg_errors_per_turn": 0.33,
  "improving": true,
  "skill_progress": {
    "verb_conjugation_past": {
      "total_uses": 5,
      "correct_count": 4,
      "error_count": 1,
      "error_rate": 0.2,
      "success_rate": 0.8
    }
  },
  "session_dynamics": {
    "consistency": 0.85,
    "trajectory": "improving",
    "engagement": 0.90,
    "anomalies": [],
    "progress_rate": 0.15
  }
}
```

---

### 4. Calibrate Model

**POST** `/api/diagnostic/calibrate`

Calibrate model predictions with human annotations.

**Request Body:**
```json
{
  "validation_dataset": [
    {
      "text": "Eu gosto de estudar português",
      "user_id": "user_123",
      "human_cefr": "B1",
      "human_scores": {
        "fluency": 3.5,
        "grammar": 4.0,
        "vocabulary": 3.8,
        "coherence": 4.2
      }
    }
  ]
}
```

**Response:**
```json
{
  "status": "success",
  "num_samples": 10,
  "avg_cefr_error": 0.3,
  "avg_score_errors": {
    "fluency": 0.1,
    "grammar": -0.05,
    "vocabulary": 0.2,
    "coherence": 0.0
  },
  "weights": {
    "fluency": 0.1,
    "grammar": -0.05,
    "vocabulary": 0.2,
    "coherence": 0.0
  }
}
```

---

## Acoustic Features Service

**Service:** `acoustic_features`  
**Port:** 8970 (default)  
**Base URL:** `http://localhost:8970`

### Health Check

**GET** `/health`

**Response:**
```json
{
  "status": "healthy",
  "service": "acoustic_features",
  "version": "1.0.0",
  "device": "mps",
  "models_loaded": true
}
```

### Extract Features

**POST** `/api/acoustic/extract_features`

Extract acoustic features from audio file.

**Request:** Multipart form data with `audio` file

**Response:**
```json
{
  "features": [0.123, 0.456, ...],
  "native_embedding_shape": [100, 768],
  "learner_embedding_shape": [100, 768],
  "fused_shape": [768],
  "model_info": {
    "model_name": "facebook/wav2vec2-large-xlsr-53",
    "device": "mps"
  }
}
```

---

## Error Codes

- **200**: Success
- **400**: Bad Request (invalid input)
- **500**: Internal Server Error
- **503**: Service Unavailable (models not loaded)

---

## References

- **Phase 1:** Do et al. (Interspeech 2024) - Acoustic Feature Mixup
- **Phase 2:** Lee et al. (Interspeech 2024) - Multi-Embedding Wav2Vec
- **Phase 3:** 
  - Lu et al. (2025) - Multi-aspect feedback
  - Reimers & Gurevych (2019) - Sentence-BERT
  - Byun et al. (2025) - LLM-as-a-Grader
  - Mohammadi et al. (2025) - ASR metadata analysis

---

## Examples

### Python (aiohttp)

```python
import aiohttp
import asyncio

async def estimate_level():
    async with aiohttp.ClientSession() as session:
        request = {
            "text": "Eu gosto de estudar português",
            "user_id": "user_123",
            "question": "Por que você estuda português?"
        }
        
        async with session.post(
            "http://localhost:8960/api/diagnostic/estimate_level",
            json=request
        ) as resp:
            data = await resp.json()
            print(f"CEFR Level: {data['cefr_level']}")
            print(f"Confidence: {data['confidence']}")
            if data.get("feedback"):
                print(f"Strengths: {data['feedback']['strengths']}")

asyncio.run(estimate_level())
```

### cURL

```bash
curl -X POST http://localhost:8960/api/diagnostic/estimate_level \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Eu gosto de estudar português",
    "user_id": "user_123"
  }'
```

---

## Performance

**Expected Latency:**
- Text-only analysis: 200-500ms
- With ASR metadata: 300-600ms
- With acoustic features: 500-1000ms (M1), 200-400ms (GPU)
- With all features: 800-1500ms (M1), 400-700ms (GPU)

**Throughput:**
- Text-only: ~10-20 requests/second
- With acoustic features: ~2-5 requests/second (M1), ~10-15 requests/second (GPU)


