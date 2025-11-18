# Thought Process Framework - Integration Quick Reference

**Quick Guide to Implement Language Teaching Thought Processes**

---

## 🎯 The Complete Picture (One Diagram)

```
STUDENT INPUT (Speech/Text)
        │
        ▼
┌──────────────────────────────────────────────────────┐
│          ORCHESTRATOR SERVICE                        │
│  (src/services/orchestrator/service.py)              │
└──────────────────┬───────────────────────────────────┘
                   │
                   │ 1. Load System Prompt
                   │
                   ▼
┌──────────────────────────────────────────────────────┐
│    SYSTEM_PROMPT_LANGUAGE_TEACHING.md                │
│                                                      │
│  - Context: Learning environment                   │
│  - Language: [Portuguese, Spanish, etc.]           │
│  - Level: [A1, A2, B1, B2, C1, C2]                 │
│  - Instructs 9 Thought Processes                   │
│  - Specifies JSON output format                    │
│  - Guides pedagogy (RECAST, SCAFFOLDING, etc.)     │
└──────────────────┬───────────────────────────────────┘
                   │
                   │ 2. Send to LLM
                   │
                   ▼
┌──────────────────────────────────────────────────────┐
│         LLM SERVICE                                  │
│  (external_llm or llm service)                      │
│                                                      │
│  Receives:                                          │
│  - System Prompt (with Thought Process instructions)│
│  - Student input                                    │
│                                                      │
│  Generates:                                         │
│  - Main Response (conversational)                   │
│  - Response Metadata (9 JSON Thought Processes)    │
└──────────────────┬───────────────────────────────────┘
                   │
                   │ 3. Process Output
                   │
                   ▼
┌──────────────────────────────────────────────────────┐
│  RESPONSE METADATA & THOUGHT PROCESS FRAMEWORK       │
│  (RESPONSE_METADATA_AND_THOUGHT_PROCESS_            │
│   FRAMEWORK.md)                                      │
│                                                      │
│  Validates & Structures:                            │
│  - Process 1: Error Detection                       │
│  - Process 2: Grammar Analysis                      │
│  - Process 3: Vocabulary Assessment                 │
│  - Process 4: Pedagogical Strategy                  │
│  - Process 5: Conversation Flow                     │
│  - Process 6: Pronunciation Evaluation              │
│  - Process 7: Learning Progress                     │
│  - Process 8: Cultural Context                      │
│  - Process 9: Learning Recommendation               │
└──────────────────┬───────────────────────────────────┘
                   │
                   │ 4. Serialize
                   │
                   ▼
┌──────────────────────────────────────────────────────┐
│  STREAMING_RESPONSE_MODELS                          │
│  (src/core/streaming_response_models.py)            │
│                                                      │
│  Pydantic models for type-safe JSON:               │
│  - LatencyMetrics                                   │
│  - ComponentOutputsResponse                         │
│  - StreamingConversationResponse                    │
│  - NDJSON Format (one JSON per line)               │
└──────────────────┬───────────────────────────────────┘
                   │
                   │ 5. Stream Response
                   │
                   ▼
┌──────────────────────────────────────────────────────┐
│  API GATEWAY                                         │
│  (src/services/api_gateway/routers/process.py)      │
│                                                      │
│  /stream/process endpoint                           │
│  Returns: NDJSON stream                             │
└──────────────────┬───────────────────────────────────┘
                   │
                   │ 6. Display to User
                   │
                   ▼
┌──────────────────────────────────────────────────────┐
│  CLIENT SDK (TypeScript)                            │
│  (client-sdk/                                        │
│                                                      │
│  Shows to student/teacher:                          │
│  - Main Response (conversational)                   │
│  - Latency Metrics (performance)                    │
│  - Thought Processes (transparency)                 │
│  - Learning Recommendations (guidance)              │
└──────────────────────────────────────────────────────┘
                   │
                   ▼
              STUDENT SEES
         Language Learning Platform
          with Full Transparency
```

---

## 📋 Implementation Checklist

### Step 1: System Prompt Setup ✅
- [ ] Read: `SYSTEM_PROMPT_LANGUAGE_TEACHING.md`
- [ ] Understand: Why it's critical
- [ ] Customize: Fill in language and level placeholders
- [ ] Store: Make accessible to Orchestrator

### Step 2: Orchestrator Integration ✅
- [ ] Location: `src/services/orchestrator/service.py`
- [ ] Update: `process_conversation()` method
- [ ] Add: Load and customize system prompt before LLM call
- [ ] Pass: `system_prompt` parameter to LLM

```python
# In orchestrator/routes.py
async def process_conversation(request: ConversationRequest):
    # Load and customize system prompt
    SYSTEM_PROMPT = load_file("SYSTEM_PROMPT_LANGUAGE_TEACHING.md")
    SYSTEM_PROMPT = customize_prompt(
        SYSTEM_PROMPT,
        language="Portuguese",
        level=student_level,
        session_number=turn_number
    )

    # Call LLM with system prompt
    response = await self.llm_service.generate(
        system_prompt=SYSTEM_PROMPT,  # ← CRITICAL
        user_input=request.audio_text,
        session_id=request.session_id
    )

    # Response should contain:
    # - llm_response (main response)
    # - response_metadata (9 Thought Processes)
    return response
```

### Step 3: Response Validation ✅
- [ ] Verify: All 9 Thought Processes are generated
- [ ] Validate: JSON structure matches framework
- [ ] Check: "analysis" field present in each process
- [ ] Test: With different student levels (A1, A2, B1, B2)

### Step 4: Streaming Setup ✅
- [ ] Location: `src/services/api_gateway/routers/process.py`
- [ ] Endpoint: `/stream/process`
- [ ] Format: NDJSON (one JSON object per line)
- [ ] Serialization: Use `StreamingResponseBuilder.to_ndjson()`

### Step 5: Client Integration ✅
- [ ] Parse NDJSON stream (one JSON per line)
- [ ] Display main response to user
- [ ] Show latency metrics
- [ ] Optionally display Thought Processes (transparency)
- [ ] Use learning recommendations for follow-up

### Step 6: Testing ✅
- [ ] Run: `pytest tests/integration/test_perceived_latency_integration.py -v`
- [ ] Verify: All 9 processes in response
- [ ] Test: With actual student inputs (errors, complexities)
- [ ] Monitor: Latency (should be < 1200ms total)

### Step 7: Deployment ✅
- [ ] Stage: Deploy to staging environment
- [ ] Monitor: Check thought process generation
- [ ] Metrics: Set up dashboards for performance
- [ ] Production: Deploy with monitoring active

---

## 🔄 Data Flow Examples

### Example 1: A2 Student Makes Grammar Error

```
INPUT:
  User: "Eu vai para praia"  [Grammar error: vai→vou]

SYSTEM PROMPT GUIDES:
  - Analyze error
  - Choose pedagogy: RECAST (implicit correction)
  - Generate 9 Thought Processes
  - Output JSON with process details

LLM RESPONSE:
{
  "llm_response": "Ah, você VAI para a praia! Que legal!",
  "response_metadata": {
    "thought_process": {
      "processes": [
        {
          "id": 1,
          "name": "Error Detection",
          "content": {
            "errors_found": [
              {
                "type": "grammar",
                "error": "vai (3rd person) → vou (1st person)",
                "severity": "major"
              }
            ]
          }
        },
        {
          "id": 4,
          "name": "Pedagogical Strategy",
          "content": {
            "strategy": "recast",
            "implementation": "Naturally repeat correct form",
            "learning_objective": "Verb conjugation awareness"
          }
        },
        ...  // Processes 2,3,5,6,7,8,9
      ]
    }
  }
}

CLIENT DISPLAYS:
  Response: "Ah, você VAI para a praia! Que legal!"

  Optional metadata panel:
  - Error detected: "vai" should be "vou"
  - Teaching method: RECAST (implicit correction)
  - Learning objective: Verb conjugation
```

### Example 2: B1 Student Uses Complex Structure

```
INPUT:
  User: "Enquanto eu estava na praia, vi meu amigo que não via há muito tempo"

SYSTEM PROMPT GUIDES:
  - Recognize: Advanced past narrative
  - Choose pedagogy: EXPANSION + PROMPTING
  - Generate 9 processes with deeper analysis

LLM RESPONSE:
{
  "llm_response": "Que encontro especial! Como se sentiram ao se reencontrarem?",
  "response_metadata": {
    "thought_process": {
      "processes": [
        {
          "id": 2,
          "name": "Grammar Analysis",
          "content": {
            "sentence_structure": "Complex with temporal subordination",
            "main_verb": "Past continuous + simple",
            "grammar_score": 1.0
          }
        },
        {
          "id": 4,
          "name": "Pedagogical Strategy",
          "content": {
            "strategy": "expansion",
            "implementation": "Expand topic, ask for more details",
            "learning_objective": "Develop narrative skills"
          }
        },
        {
          "id": 9,
          "name": "Learning Recommendation",
          "content": {
            "recommended_practice": "Temporal connectors",
            "focus": "então, depois, enquanto",
            "priority": "medium"
          }
        },
        ...  // Other 6 processes
      ]
    }
  }
}
```

---

## 📊 Files You Need to Know

| File | Purpose | Status |
|------|---------|--------|
| `SYSTEM_PROMPT_LANGUAGE_TEACHING.md` | **Instructions for LLM** | ✅ Ready |
| `RESPONSE_METADATA_AND_THOUGHT_PROCESS_FRAMEWORK.md` | Define 9 processes & structure | ✅ Ready |
| `src/core/streaming_response_models.py` | Pydantic models | ✅ Ready |
| `src/core/perceived_latency_manager.py` | Latency tracking | ✅ Ready |
| `orchestrator/routes.py` | **INTEGRATE PROMPT HERE** | 📝 Action needed |
| `api_gateway/routers/process.py` | Streaming endpoint | 📝 Action needed |
| `client-sdk/` | Display responses | 📝 Action needed |
| `tests/integration/test_perceived_latency_integration.py` | Validation | ✅ Ready |

---

## ⚠️ Critical Success Factors

### Without System Prompt
❌ LLM won't know about Thought Processes
❌ Response Metadata won't be generated
❌ System acts as general chatbot, not language teacher
❌ 9 Processes missing
❌ Pedagogical strategies not applied

### With System Prompt (Properly Integrated)
✅ LLM understands learning context
✅ All 9 Thought Processes generated automatically
✅ System acts as intelligent language teacher
✅ Students see transparent reasoning
✅ Pedagogical strategies applied implicitly
✅ Learning recommendations provided

---

## 🚀 Immediate Next Steps (Priority Order)

1. **Read** the System Prompt: `SYSTEM_PROMPT_LANGUAGE_TEACHING.md`
2. **Understand** Why it's critical (System Prompt Integration section)
3. **Integrate** into Orchestrator (load and pass to LLM)
4. **Test** with sample student inputs
5. **Verify** all 9 processes are generated
6. **Deploy** to staging
7. **Monitor** and refine

---

## 🔗 See Also

- `SYSTEM_PROMPT_LANGUAGE_TEACHING.md` - **READ THIS FIRST** ⚠️
- `RESPONSE_METADATA_AND_THOUGHT_PROCESS_FRAMEWORK.md` - Detailed process definitions
- `PERCEIVED_LATENCY_SYSTEM_INTEGRATION.md` - Latency framework
- `PERCEIVED_LATENCY_INTEGRATION_SUMMARY.md` - Overview

---

**Status:** 🟢 **Complete and Ready for Integration**

The foundation is ready. The critical piece is ensuring the System Prompt is properly integrated into the Orchestrator's LLM pipeline.

**Version:** 1.0
**Last Updated:** October 26, 2025
