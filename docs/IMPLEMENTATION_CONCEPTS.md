# Intelligent Tutoring System - Implementation Concepts Tracker

This document tracks all pedagogical and AI concepts implemented in the system, mapping them to specific services and academic references.

**Database**: `docs/implementation_tracker.db`  
**Total Concepts**: 135 (23 implemented, 15 partially implemented, 97 planned)

---

## Summary by Service

| Service | Implemented | Partially Implemented | Planned | Total |
|---------|------------|----------------------|---------|-------|
| `student_model` | 6 | 3 | 39 | 48 |
| `pedagogical_policy` | 9 | 3 | 25 | 37 |
| `learning_path` | 3 | 0 | 16 | 19 |
| `orchestrator` | 2 | 3 | 11 | 16 |
| `diagnostic_module` | 3 | 1 | 11 | 15 |

---

## Implemented Concepts

### 🧠 Student Model Service

| Concept | Implementation | Reference |
|---------|---------------|-----------|
| **Bayesian Knowledge Tracing (BKT)** | Fallback algorithm in `knowledge_tracer/bkt_tracer.py`. Tracks P(L), P(T), P(G), P(S) for each skill. | Corbett, A. T., & Anderson, J. R. (1994). Knowledge tracing: Modeling the acquisition of procedural knowledge. User modeling and user-adapted interaction, 4(4), 253-278. |
| **Attentive Knowledge Tracing (AKT)** | Primary algorithm in `knowledge_tracer/akt_tracer.py`. Uses attention mechanisms, temporal decay, and contextual features. | Ghosh, A., Heffernan, N., & Lan, A. S. (2020). Context-aware attentive knowledge tracing. KDD. |
| **CEFR Level Mapping** | `SKILL_CEFR_MAP` in `skill_registry.py` maps skills to CEFR levels (A1-C2). `calculate_cefr_progress` aggregates AKT probabilities. | Council of Europe. (2001). Common European Framework of Reference for Languages. Cambridge University Press. |
| **Deep Learning for Knowledge States** | AKT uses attention mechanisms and temporal features to model complex knowledge states beyond simple probabilities. | Piech, C., et al. (2015). Deep knowledge tracing. NIPS. |
| **Contextual Feature Engineering** | AKT processes contextual features (difficulty, complexity, time_since_last, error_type_severity) for nuanced mastery updates. | Ghosh, A., Heffernan, N., & Lan, A. S. (2020). Context-aware attentive knowledge tracing. KDD. |
| **Automatic Progress Recognition** | `calculate_cefr_progress` detects when student masters a level (>80%) and estimates transition to next CEFR level. | Käser, T., et al. (2017). Modeling exploration strategies to predict student performance. EDM. |

---

### 📚 Pedagogical Policy Service

| Concept | Implementation | Reference |
|---------|---------------|-----------|
| **Scaffolding (ZPD)** | `StrategyLayer` provides adaptive support: Explicit correction for beginners (<30% mastery), Implicit recast for intermediate (30-70%), Minimal for advanced (>70%). | Vygotsky, L. S. (1978). Mind in society: The development of higher psychological processes. Harvard University Press. |
| **Input Hypothesis (i+1)** | CHALLENGE strategy in `PolicyEngine` exposes students to content slightly above current level when mastery > 70%. | Krashen, S. D. (1985). The input hypothesis: Issues and implications. Longman. |
| **Recast-based Correction** | `ScaffoldingType.IMPLICIT` in `strategy_layer.py` provides subtle error correction by reformulating correct responses naturally. | Long, M. H. (2006). Problems in SLA. Lawrence Erlbaum Associates. |
| **Instruction-Grounded Conversations** | `PromptComposer` uses layered architecture (StudentStateLayer, StrategyLayer, FocusLayer, AffectiveLayer) to build pedagogically-informed prompts. | Kumar, G., et al. (2022). Instruction-grounded conversational agents. arXiv:2211.09020. |
| **Emotional State Modulation** | `AffectiveLayer` and `PolicyEngine.get_emotional_modulation` adapt tone and pace based on student emotional state (motivated, frustrated, confused, etc.). | D'Mello, S., & Graesser, A. (2012). Dynamics of affective states during complex learning. Learning and Instruction, 22(2), 145-157. |

---

### 🔍 Diagnostic Module Service

| Concept | Implementation | Reference |
|---------|---------------|-----------|
| **Automatic Error Detection** | `DiagnosticLLMClient.analyze_grammar` uses LLM to identify grammar, vocabulary, and syntax errors with skill tagging. | Rei, M., & Yannakoudakis, H. (2016). Compositional sequence labeling models for error detection. ACL. |
| **Skill Tagging (Granular Assessment)** | LLM prompt updated to identify specific `skill_id` for both errors and correct usages, enabling fine-grained knowledge updates. | Benedetto, L., et al. (2020). R2DE: a NLP approach to estimating IRT parameters. LAK. |
| **Multi-faceted Student Assessment** | Analyzers for grammar, vocabulary, complexity, and progress provide cognitive and linguistic evaluation. | Desmarais, M. C., & Baker, R. S. (2012). A review of recent advances in learner and skill modeling. UMUAI, 22(1-2), 9-38. |

---

### 🗺️ Learning Path Service

| Concept | Implementation | Reference |
|---------|---------------|-----------|
| **Spaced Repetition System (SRS)** | `SpacedRepetitionSystem` in `spaced_repetition.py` calculates optimal review intervals using forgetting curves and mastery levels. | Ebbinghaus, H. (1885). Memory: A contribution to experimental psychology. Teachers College, Columbia University. |
| **Zone of Proximal Development (ZPD) Calculator** | `ZPDCalculator` in `zpd_calculator.py` identifies skills within learner's ZPD based on mastery and prerequisites. | Vygotsky, L. S. (1978). Mind in society. Harvard University Press. |
| **Adaptive Learning Path Navigation** | `LearningPathNavigator` combines SRS and ZPD to recommend next skills, balancing new content and review. | Clement, B., et al. (2015). Multi-armed bandits for intelligent tutoring systems. JEDM, 7(2), 20-48. |

---

### 🎯 Orchestrator Service

| Concept | Implementation | Reference |
|---------|---------------|-----------|
| **Microservices Architecture for ITS** | Orchestrator coordinates `student_model`, `pedagogical_policy`, `diagnostic_module`, and `learning_path` as independent services. | Aleven, V., et al. (2016). Instruction based on adaptive learning technologies. Handbook of research on learning and instruction. |
| **Background Knowledge Update (Non-blocking)** | `_analyze_and_update_knowledge` runs as asyncio background task to avoid blocking conversation flow. | VanLehn, K. (2011). The relative effectiveness of human tutoring, intelligent tutoring systems. Educational Psychologist, 46(4), 197-221. |

---

## 🔄 Partially Implemented Concepts

### Pedagogical Policy Service

| Concept | Current State | Next Steps | Reference |
|---------|--------------|------------|-----------|
| **Output Hypothesis (Pushed Output)** | CHALLENGE strategy pushes students beyond comfort zone. | Add explicit "Try to use..." prompts for target structures. | Swain, M. (1985). Communicative competence: Some roles of comprehensible input and comprehensible output. Newbury House. |
| **Cognitive Load Theory** | CEFR-based adaptation manages difficulty. | Add explicit load metrics (intrinsic/extraneous/germane) to PolicyEngine. | Sweller, J. (1988). Cognitive load during problem solving: Effects on learning. Cognitive Science, 12(2), 257-285. |

### Orchestrator Service

| Concept | Current State | Next Steps | Reference |
|---------|--------------|------------|-----------|
| **Task-Based Language Teaching (TBLT)** | Scenarios support task-based design. | Add task complexity metrics and sequencing. | Ellis, R. (2003). Task-based language learning and teaching. Oxford University Press. |
| **Mobile-Assisted Language Learning (MALL)** | REST API supports mobile access. | Add push notifications for spaced repetition reminders. | Kukulska-Hulme, A., & Shield, L. (2008). An overview of mobile assisted language learning. ReCALL, 20(3), 271-289. |
| **Dialogue Management for SLA** | Conversation history and context managed. | Add explicit dialogue acts (question, clarification, correction) tracking. | Griol, D., et al. (2014). A statistical approach to spoken dialog systems design. Speech Communication, 60, 1-20. |

### Student Model Service

| Concept | Current State | Next Steps | Reference |
|---------|--------------|------------|-----------|
| **Gamification in Language Learning** | Progress tracking (CEFR, mastery) provides gamification elements. | Add badges, streaks, and leaderboards using mastery data. | Reinders, H., & Wattana, S. (2015). Affect and willingness to communicate in digital game-based learning. ReCALL, 27(1), 38-57. |

### Diagnostic Module Service

| Concept | Current State | Next Steps | Reference |
|---------|--------------|------------|-----------|
| **Error Analysis & Interlanguage** | Errors are identified. | Track interlanguage development patterns (systematic errors indicating learning stage). | Corder, S. P. (1967). The significance of learners' errors. IRAL, 5(4), 161-170. |

---

## 🚧 Planned Concepts

### Pedagogical Policy Service

| Concept | Planned Implementation | Reference |
|---------|----------------------|-----------|
| **Reinforcement Learning Policy (Thompson Sampling)** | Multi-Armed Bandit using Thompson Sampling to dynamically select optimal teaching strategies based on student feedback. | Clement, B., et al. (2015). Multi-armed bandits for intelligent tutoring systems. JEDM, 7(2), 20-48. |
| **Contextual Bandits for Personalization** | Context-aware RL that learns per-CEFR-level policies (e.g., "A1 students respond better to TEACH"). | Mandel, T., et al. (2014). Offline policy evaluation across representations. AAMAS. |
| **Interaction Hypothesis (Negotiation of Meaning)** | Facilitate negotiation through clarification requests, confirmation checks, and comprehension checks. | Long, M. H. (1996). The role of the linguistic environment in second language acquisition. Handbook of SLA. |
| **Comprehension Checks** | Periodically ask "Do you understand?" or "Can you explain what I just said?" to verify comprehension. | Long, M. H. (1983). Native speaker/non-native speaker conversation and the negotiation of comprehensible input. Applied Linguistics, 4(2), 126-141. |
| **Noticing Hypothesis** | Highlight grammatical structures in responses (e.g., bold key phrases) to draw attention. | Schmidt, R. (1990). The role of consciousness in second language learning. Applied Linguistics, 11(2), 129-158. |
| **Input Enhancement** | Make target linguistic features more salient through typographical enhancement (bold, italics) or repetition. | Sharwood Smith, M. (1993). Input enhancement in instructed SLA: Theoretical bases. SSLA, 15(2), 165-179. |
| **Adaptive Feedback Timing** | Adjust feedback timing based on student state: immediate for beginners, delayed for advanced. | Shute, V. J. (2008). Focus on formative feedback. Review of Educational Research, 78(1), 153-189. |
| **Incidental Vocabulary Learning** | Introduce new vocabulary naturally in conversation with vocabulary tracking and frequency-based selection. | Nation, I. S. P. (2001). Learning vocabulary in another language. Cambridge University Press. |
| **Learning Style Adaptation** | Adapt to visual, auditory, or kinesthetic preferences based on user profile. | Oxford, R. L. (2003). Language learning styles and strategies: An overview. GALA, 1-25. |
| **Aptitude-Treatment Interaction** | Match teaching strategies to individual aptitudes (e.g., analytical learners benefit from explicit rules). | Robinson, P. (2002). Individual differences and instructed language learning. John Benjamins. |
| **Motivation Maintenance** | Provide encouragement messages and celebrate milestones when emotional_state is MOTIVATED or CONFIDENT. | Dörnyei, Z. (2001). Motivational strategies in the language classroom. Cambridge University Press. |

### Learning Path Service

| Concept | Planned Implementation | Reference |
|---------|----------------------|-----------|
| **Pushed Output Tasks** | Design tasks that require students to use specific grammatical structures. | Swain, M. (1995). Three functions of output in second language learning. Oxford University Press. |
| **Task Complexity Sequencing** | Sequence tasks from simple to complex based on cognitive load using difficulty scores. | Robinson, P. (2001). Task complexity, task difficulty, and task production. Applied Linguistics, 22(1), 27-57. |
| **Vocabulary Frequency Lists** | Prioritize high-frequency words in vocabulary_basic skills with frequency data integration. | Nation, I. S. P., & Waring, R. (1997). Vocabulary size, text coverage and word lists. Cambridge University Press. |

### Orchestrator Service

| Concept | Planned Implementation | Reference |
|---------|----------------------|-----------|
| **Interaction Hypothesis (Negotiation of Meaning)** | Detect confusion and ask "Did you mean...?" or "Could you clarify?" to facilitate negotiation. | Long, M. H. (1996). The role of the linguistic environment in second language acquisition. Handbook of SLA. |
| **Communication Breakdown Repair** | Detect unfinished utterances (timeouts) and repair communication by asking 'Would you like to finish your thought?' | Avouris, N. (2025). AI Conversational Tutors in Foreign Language Learning: A Mixed-Methods Evaluation Study. |
| **Scene Construction** | Adapt virtual environment context based on student profile. | Yang, J., & Zhang, B. (2019). Artificial Intelligence in Intelligent Tutoring Robots. Applied Sciences. |
| **Voice Interaction Modes** | Support both automatic (hands-free) and manual (button-press) voice interaction modes. | Avouris, N. (2025). AI Conversational Tutors in Foreign Language Learning. |
| **Perception-Planning-Action Framework** | Transform teaching-learning relationship into perception (diagnose), planning (select strategy), action (execute). | Yang, J., & Zhang, B. (2019). Artificial Intelligence in Intelligent Tutoring Robots. |

### Pedagogical Policy Service (Additional from Papers)

| Concept | Planned Implementation | Reference |
|---------|----------------------|-----------|
| **Teaching Action Codes** | Explicit action codes: [Correction], [Confirmation], [Others] for dialogue action. | Chae, H., et al. (2023). TUTORING: Instruction-Grounded Conversational Agent for Language Learners. AAAI. |
| **Instruction Transition Detection** | Monitor when current instruction should transition to next based on mastery and dialogue turns. | Chae, H., et al. (2023). TUTORING: Instruction-Grounded Conversational Agent for Language Learners. AAAI. |
| **Contextual Error Correction with Explanations** | Provide corrections with: (1) appraisal, (2) error remarks, (3) re-phrasing suggestion, (4) follow-up question. | Avouris, N. (2025). AI Conversational Tutors in Foreign Language Learning. |
| **Cultural Authenticity in Conversations** | Ensure AI responses reflect cultural nuances and authentic language use. | Avouris, N. (2025). AI Conversational Tutors in Foreign Language Learning. |
| **Interventional Messages** | Personalized messages based on affective state (e.g., 'I see you're frustrated, let's try a different approach'). | Park, M., et al. (2024). Empowering Personalized Learning through a Conversation-based Tutoring System. CHI EA '24. |
| **Felder-Silverman Learning Style Model** | Classify learning styles into 16 categories (Perception, Processing, Understanding dimensions). | Park, M., et al. (2024). Empowering Personalized Learning through a Conversation-based Tutoring System. CHI EA '24. |
| **Alternative Phrasing Suggestions** | Suggest alternative ways to express student's message when errors are detected. | Avouris, N. (2025). AI Conversational Tutors in Foreign Language Learning. |
| **Critical Thinking Guidance** | Focus on critical thinking over learner's argument, guiding into further reasoning. | Avouris, N. (2025). AI Conversational Tutors in Foreign Language Learning. |
| **Step-based vs Substep-based Tutoring** | Provide feedback at appropriate granularity levels (step-based has 0.76 effect size). | Alkhatlan, A., & Kalita, J. K. (2018). Intelligent Tutoring Systems: A Comprehensive Historical Survey. |

### Student Model Service (Additional from Papers)

| Concept | Planned Implementation | Reference |
|---------|----------------------|-----------|
| **Item Response Theory (IRT) for Proficiency Assessment** | Use IRT model to assess proficiency levels in knowledge concepts, complementing AKT. | Park, M., et al. (2024). Empowering Personalized Learning through a Conversation-based Tutoring System. CHI EA '24. |
| **Metacognition Tracking** | Track self-reported self-assessment vs actual proficiency to measure metacognitive awareness. | Park, M., et al. (2024). Empowering Personalized Learning through a Conversation-based Tutoring System. CHI EA '24. |
| **Learning Gain Calculation** | Calculate learning gain as difference between current and previous knowledge level (APR_t - APR_{t-1}). | Chen, J.-Y., et al. (2023). Adaptive Learning Path Navigation Based on Knowledge Tracing and Reinforcement Learning. |
| **Average Performance Rate (APR)** | Calculate average probability of correct answers across all available exercises as overall knowledge metric. | Chen, J.-Y., et al. (2023). Adaptive Learning Path Navigation Based on Knowledge Tracing and Reinforcement Learning. |
| **Progress Recognition Tasks** | Multi-task learning to infer teaching action and progress simultaneously. | Chae, H., et al. (2023). TUTORING: Instruction-Grounded Conversational Agent for Language Learners. AAAI. |
| **Knowledge Graph Integration** | Use knowledge graphs to represent domain knowledge and student knowledge state. | Yang, J., & Zhang, B. (2019). Artificial Intelligence in Intelligent Tutoring Robots. Applied Sciences. |
| **Daily/Weekly Feedback Reports** | Generate summary reports with observations, suggestions, and overall score after each session. | Avouris, N. (2025). AI Conversational Tutors in Foreign Language Learning. |
| **Model Tracing** | Monitor student's problem-solving steps incrementally and intervene when mistakes are made. | Alkhatlan, A., & Kalita, J. K. (2018). Intelligent Tutoring Systems: A Comprehensive Historical Survey. |

### Diagnostic Module Service (Additional from Papers)

| Concept | Planned Implementation | Reference |
|---------|----------------------|-----------|
| **Session-End Summaries** | Generate LLM-based summaries at end of session evaluating cognitive, affective state, and learning style. | Park, M., et al. (2024). Empowering Personalized Learning through a Conversation-based Tutoring System. CHI EA '24. |
| **Gamification Elements (Error Indicators)** | Visual indicators (green/yellow signs) for error-free vs error-containing utterances. | Avouris, N. (2025). AI Conversational Tutors in Foreign Language Learning. |
| **Confidence Scoring for Speech Recognition** | Measure confidence in interpretation of student speech (spoken precision index). | Avouris, N. (2025). AI Conversational Tutors in Foreign Language Learning. |
| **Annotated Chat Transcripts** | Provide annotated transcripts with error markings, corrections, and explanations. | Avouris, N. (2025). AI Conversational Tutors in Foreign Language Learning. |

### Learning Path Service (Additional from Papers)

| Concept | Planned Implementation | Reference |
|---------|----------------------|-----------|
| **Entropy-enhanced Proximal Policy Optimization (EPPO)** | Enhanced RL algorithm for learning path recommendation with better exploration capabilities. | Chen, J.-Y., et al. (2023). Adaptive Learning Path Navigation Based on Knowledge Tracing and Reinforcement Learning. |
| **Diversity Penalty in Learning Paths** | Ensure diversity in recommended materials by penalizing repetition in reward function. | Chen, J.-Y., et al. (2023). Adaptive Learning Path Navigation Based on Knowledge Tracing and Reinforcement Learning. |

---

## Querying the Database

The SQLite database can be queried for detailed information:

```bash
# Open database
sqlite3 docs/implementation_tracker.db

# View all implemented concepts
SELECT concept, service, status FROM implementation_concepts WHERE status='implemented';

# View concepts by service
SELECT concept, reference FROM implementation_concepts WHERE service='pedagogical_policy';

# Search by keyword
SELECT concept, service FROM implementation_concepts WHERE concept LIKE '%Knowledge Tracing%';

# Get full details for a concept
SELECT * FROM implementation_concepts WHERE concept='Attentive Knowledge Tracing (AKT)';
```

---

## Database Schema

```sql
CREATE TABLE implementation_concepts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    concept TEXT NOT NULL,
    service TEXT NOT NULL,
    implementation_details TEXT,
    reference TEXT NOT NULL,
    status TEXT DEFAULT 'implemented',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```

---

## Academic Impact

This implementation synthesizes research from:
- **Cognitive Psychology**: Vygotsky (ZPD), Ebbinghaus (Forgetting Curve), Sweller (Cognitive Load)
- **Second Language Acquisition**: Krashen (Input Hypothesis, Affective Filter), Long (Interaction Hypothesis), Swain (Output Hypothesis), Schmidt (Noticing Hypothesis)
- **Educational Data Mining**: BKT, DKT, AKT algorithms
- **Reinforcement Learning**: Multi-Armed Bandits for ITS
- **Natural Language Processing**: Error detection, complexity analysis
- **Task-Based Learning**: Ellis, Robinson (Task Complexity)
- **Assessment**: Black & Wiliam (Formative Assessment), Shute (Adaptive Feedback)
- **Motivation & Individual Differences**: Dörnyei (Motivation), Oxford (Learning Styles), Robinson (Aptitude)

**Total References**: 60+ peer-reviewed papers and seminal works in education, SLA, AI, and ITS research.

**New Concepts from Paper Analysis**: 
- First pass: 29 concepts identified from initial analysis
- Second pass: 36 additional technical concepts from deeper analysis
- Third pass: 26 concepts from AKT-related papers (FoLiBi, SINKT, Vocabulary KT, AKT original)
- **Total new concepts**: 91 concepts from 10 papers
- **Grand total**: 135 concepts (23 implemented, 15 partially implemented, 97 planned)

---

**Last Updated**: 2025-01-21  
**Maintainer**: ITS Development Team  
**New Concepts Added**: 
- 23 concepts from SLA theory and CALL research
- 91 concepts from paper analysis:
  - First pass: 29 high-level concepts (6 papers)
  - Second pass: 36 technical/detailed concepts (same 6 papers, deeper analysis)
  - Third pass: 26 AKT-specific concepts (FoLiBi, SINKT, Vocabulary KT, AKT original)
- **Papers analyzed**: 10 papers total

