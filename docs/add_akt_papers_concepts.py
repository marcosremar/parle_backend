#!/usr/bin/env python3
"""
Script to add concepts discovered from analyzing AKT-related papers:
- FoLiBi: Forgetting-aware Linear Bias for AKT (2023)
- SINKT: Structure-Aware Inductive Knowledge Tracing with LLM (2024)
- Knowledge Tracing in Sequential Learning of Inflected Vocabulary (2017)
- Adaptive Learning Path Navigation (ALPN) (2023) - já analisado, mas adicionar conceitos específicos
"""

import sqlite3
from pathlib import Path

# Database path
db_path = Path(__file__).parent / "implementation_tracker.db"

# Create connection
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Concepts from AKT-related papers
akt_concepts = [
    # From FoLiBi: Forgetting-aware Linear Bias for AKT (2023)
    {
        "concept": "Forgetting-aware Linear Bias (FoLiBi)",
        "service": "student_model",
        "implementation_details": "Decouple forgetting behavior from question correlations using linear bias β_t = m_h · [1, ..., t-1]. Penalizes positional decay independently of question similarity. Can be added to AKT implementation to improve performance by up to 2.58% AUC.",
        "reference": "Im, Y., Choi, E., Kook, H., & Lee, J. (2023). Forgetting-aware Linear Bias for Attentive Knowledge Tracing. CIKM '23.",
        "status": "planned"
    },
    {
        "concept": "Disentanglement of Forgetting and Question Correlation",
        "service": "student_model",
        "implementation_details": "Separate modeling of forgetting behavior from question correlations in attention mechanism. FoLiBi shows this improves performance especially for longer interaction histories.",
        "reference": "Im, Y., Choi, E., Kook, H., & Lee, J. (2023). Forgetting-aware Linear Bias for Attentive Knowledge Tracing. CIKM '23.",
        "status": "planned"
    },
    {
        "concept": "Linear Bias for Positional Decay",
        "service": "student_model",
        "implementation_details": "Use linear bias proportional to relative distance to penalize forgetting, independent of question correlation. Formula: β_t = m_h · [1, ..., t-1] where m_h adjusts importance per attention head.",
        "reference": "Im, Y., Choi, E., Kook, H., & Lee, J. (2023). Forgetting-aware Linear Bias for Attentive Knowledge Tracing. CIKM '23.",
        "status": "planned"
    },
    {
        "concept": "Robustness Against Sequence Length Variation",
        "service": "student_model",
        "implementation_details": "FoLiBi shows robustness against various sequence lengths, making it suitable for systems with varying interaction history lengths.",
        "reference": "Im, Y., Choi, E., Kook, H., & Lee, J. (2023). Forgetting-aware Linear Bias for Attentive Knowledge Tracing. CIKM '23.",
        "status": "planned"
    },
    
    # From SINKT: Structure-Aware Inductive Knowledge Tracing with LLM (2024)
    {
        "concept": "Inductive Knowledge Tracing",
        "service": "student_model",
        "implementation_details": "Predict responses to new questions/concepts not seen in training data. SINKT uses LLMs to encode semantic information instead of ID embeddings, enabling generalization to unseen questions.",
        "reference": "Fu, L., et al. (2024). SINKT: A Structure-Aware Inductive Knowledge Tracing Model with Large Language Model. CIKM '24.",
        "status": "planned"
    },
    {
        "concept": "LLM-based Semantic Encoding for Questions",
        "service": "student_model",
        "implementation_details": "Use Pretrained Language Models (PLMs) to encode semantic information of questions and concepts instead of training ID embeddings. Enables handling of new questions without retraining.",
        "reference": "Fu, L., et al. (2024). SINKT: A Structure-Aware Inductive Knowledge Tracing Model with Large Language Model. CIKM '24.",
        "status": "planned"
    },
    {
        "concept": "Concept-Question Heterogeneous Graph",
        "service": "student_model",
        "implementation_details": "Use LLMs to generate heterogeneous graph containing structural relationships between concepts and questions. Can be used to model dependencies (e.g., addition → subtraction → multiplication).",
        "reference": "Fu, L., et al. (2024). SINKT: A Structure-Aware Inductive Knowledge Tracing Model with Large Language Model. CIKM '24.",
        "status": "planned"
    },
    {
        "concept": "Structural Information Encoder",
        "service": "student_model",
        "implementation_details": "Encode concept-question graph using carefully designed structural information encoder to capture topological relations among questions and concepts.",
        "reference": "Fu, L., et al. (2024). SINKT: A Structure-Aware Inductive Knowledge Tracing Model with Large Language Model. CIKM '24.",
        "status": "planned"
    },
    {
        "concept": "Open-world Knowledge Integration via LLMs",
        "service": "student_model",
        "implementation_details": "Integrate open-world semantic and structural information by LLMs, enabling dynamic updates and expansion of knowledge base with minimal manual intervention.",
        "reference": "Fu, L., et al. (2024). SINKT: A Structure-Aware Inductive Knowledge Tracing Model with Large Language Model. CIKM '24.",
        "status": "planned"
    },
    {
        "concept": "Cold Start Problem Solution for New Questions",
        "service": "student_model",
        "implementation_details": "Address cold start problem where new questions/concepts lack interaction data. SINKT uses semantic encoding to predict responses to unseen questions.",
        "reference": "Fu, L., et al. (2024). SINKT: A Structure-Aware Inductive Knowledge Tracing Model with Large Language Model. CIKM '24.",
        "status": "planned"
    },
    
    # From Knowledge Tracing in Sequential Learning of Inflected Vocabulary (2017)
    {
        "concept": "Parametric Knowledge Tracing (PKT)",
        "service": "student_model",
        "implementation_details": "Represent student knowledge as vector of prediction parameters (feature weights) rather than binary skill bits. Uses log-linear model with feature functions for sub-skills.",
        "reference": "Renduchintala, A., Koehn, P., & Eisner, J. (2017). Knowledge Tracing in Sequential Learning of Inflected Vocabulary. CoNLL 2017.",
        "status": "planned"
    },
    {
        "concept": "Neural Gating Mechanism for Knowledge Updates",
        "service": "student_model",
        "implementation_details": "Use neural gating mechanism to model how student updates log-linear parameters in response to feedback. Allows learning complex patterns of retention and acquisition for each feature.",
        "reference": "Renduchintala, A., Koehn, P., & Eisner, J. (2017). Knowledge Tracing in Sequential Learning of Inflected Vocabulary. CoNLL 2017.",
        "status": "planned"
    },
    {
        "concept": "Feature-rich Knowledge State Representation",
        "service": "student_model",
        "implementation_details": "Represent knowledge state using feature vectors (phrasal, word, suffix, prefix features) with interpretable weights. Enables understanding of which sub-skills student has mastered.",
        "reference": "Renduchintala, A., Koehn, P., & Eisner, J. (2017). Knowledge Tracing in Sequential Learning of Inflected Vocabulary. CoNLL 2017.",
        "status": "planned"
    },
    {
        "concept": "Sub-skill Interaction Modeling",
        "service": "student_model",
        "implementation_details": "Model interaction between sub-skills (e.g., verb conjugation suffixes, word recognition) using log-linear formulation with feature functions. Critical for vocabulary learning with inflection.",
        "reference": "Renduchintala, A., Koehn, P., & Eisner, J. (2017). Knowledge Tracing in Sequential Learning of Inflected Vocabulary. CoNLL 2017.",
        "status": "planned"
    },
    {
        "concept": "Interpretable Knowledge State for Feedback",
        "service": "student_model",
        "implementation_details": "Maintain interpretable knowledge state (feature weights) to provide useful feedback to students and teachers, and construct educational stimuli targeted at improving particular sub-skills.",
        "reference": "Renduchintala, A., Koehn, P., & Eisner, J. (2017). Knowledge Tracing in Sequential Learning of Inflected Vocabulary. CoNLL 2017.",
        "status": "planned"
    },
    {
        "concept": "Verb Conjugation Task Modeling",
        "service": "student_model",
        "implementation_details": "Model verb conjugation learning where student needs to deploy sub-word features and generalize to new examples. Uses flash card system with example, multiple-choice, and typing cards.",
        "reference": "Renduchintala, A., Koehn, P., & Eisner, J. (2017). Knowledge Tracing in Sequential Learning of Inflected Vocabulary. CoNLL 2017.",
        "status": "planned"
    },
    {
        "concept": "Retention and Acquisition Pattern Learning",
        "service": "student_model",
        "implementation_details": "Learn complex patterns of retention and acquisition for each feature using neural gating mechanism. Captures how knowledge is retained/forgotten over time for different sub-skills.",
        "reference": "Renduchintala, A., Koehn, P., & Eisner, J. (2017). Knowledge Tracing in Sequential Learning of Inflected Vocabulary. CoNLL 2017.",
        "status": "planned"
    },
    {
        "concept": "Log-linear Model for Structured Prediction",
        "service": "student_model",
        "implementation_details": "Use log-linear distribution parameterized by knowledge state θ: p(y|a;θ) = exp(θ·φ(x,y)) / Σ exp(θ·φ(x,y')). Enables structured prediction for vocabulary learning.",
        "reference": "Renduchintala, A., Koehn, P., & Eisner, J. (2017). Knowledge Tracing in Sequential Learning of Inflected Vocabulary. CoNLL 2017.",
        "status": "planned"
    },
    
    # From ALPN (2023) - Additional concepts not captured before
    {
        "concept": "AKT as Environment for RL Agent",
        "service": "learning_path",
        "implementation_details": "Use pre-trained AKT model as environment, allowing RL agent to explore diverse learning paths. AKT provides knowledge state transitions for RL training.",
        "reference": "Chen, J.-Y., Saeedvand, S., & Lai, I.-W. (2023). Adaptive Learning Path Navigation Based on Knowledge Tracing and Reinforcement Learning. PRIME AI.",
        "status": "planned"
    },
    {
        "concept": "Probabilistic Formulation for Learning Paths",
        "service": "learning_path",
        "implementation_details": "Formulate learning path generation as probability: P[τ] = P[s1] · Π π(a_t|s_t) · P[c_t|s_t,a_t] · P[s_{t+1}|s_t,a_t,c_t]. Maximize expected cumulative reward.",
        "reference": "Chen, J.-Y., Saeedvand, S., & Lai, I.-W. (2023). Adaptive Learning Path Navigation Based on Knowledge Tracing and Reinforcement Learning. PRIME AI.",
        "status": "planned"
    },
    {
        "concept": "Knowledge State Transition Determinism",
        "service": "student_model",
        "implementation_details": "Knowledge state transition P[s_{t+1}|s_t,a_t,c_t] is deterministic for given (s_t, a_t, c_t) pair when using trained AKT model. This simplifies RL gradient computation.",
        "reference": "Chen, J.-Y., Saeedvand, S., & Lai, I.-W. (2023). Adaptive Learning Path Navigation Based on Knowledge Tracing and Reinforcement Learning. PRIME AI.",
        "status": "planned"
    },
    
    # From Context-Aware Attentive Knowledge Tracing (2020) - Original AKT paper
    {
        "concept": "Monotonic Attention Mechanism",
        "service": "student_model",
        "implementation_details": "Use exponential decay curve to down-weight importance of questions in distant past. Inspired by cognitive science findings on forgetting mechanisms. Can be enhanced with FoLiBi.",
        "reference": "Ghosh, A., Heffernan, N., & Lan, A. S. (2020). Context-Aware Attentive Knowledge Tracing. KDD '20.",
        "status": "partially_implemented"
    },
    {
        "concept": "Context-aware Relative Distance Measure",
        "service": "student_model",
        "implementation_details": "Develop context-aware measure to characterize time distance between questions. Uses relative distance instead of absolute position embeddings to handle time outliers.",
        "reference": "Ghosh, A., Heffernan, N., & Lan, A. S. (2020). Context-Aware Attentive Knowledge Tracing. KDD '20.",
        "status": "partially_implemented"
    },
    {
        "concept": "Rasch Model-based Embeddings",
        "service": "student_model",
        "implementation_details": "Use Rasch model (simple IRT model) to regularize concept and question embeddings. Captures individual differences among questions without excessive parameters.",
        "reference": "Ghosh, A., Heffernan, N., & Lan, A. S. (2020). Context-Aware Attentive Knowledge Tracing. KDD '20.",
        "status": "planned"
    },
    {
        "concept": "Context-aware Representations",
        "service": "student_model",
        "implementation_details": "Put raw embeddings into context using learner's entire practice history. Creates context-aware representations of past questions and responses.",
        "reference": "Ghosh, A., Heffernan, N., & Lan, A. S. (2020). Context-Aware Attentive Knowledge Tracing. KDD '20.",
        "status": "partially_implemented"
    },
    {
        "concept": "Interpretability for Automated Feedback",
        "service": "student_model",
        "implementation_details": "AKT exhibits excellent interpretability, enabling automated feedback and practice question recommendation. Attention weights show which past interactions are most relevant.",
        "reference": "Ghosh, A., Heffernan, N., & Lan, A. S. (2020). Context-Aware Attentive Knowledge Tracing. KDD '20.",
        "status": "planned"
    }
]

# Check if concept already exists and insert
added_count = 0
skipped_count = 0

for concept_data in akt_concepts:
    cursor.execute("""
        SELECT id FROM implementation_concepts 
        WHERE concept = ? AND service = ?
    """, (concept_data["concept"], concept_data["service"]))
    
    if cursor.fetchone() is None:
        cursor.execute("""
            INSERT INTO implementation_concepts 
            (concept, service, implementation_details, reference, status)
            VALUES (?, ?, ?, ?, ?)
        """, (
            concept_data["concept"],
            concept_data["service"],
            concept_data["implementation_details"],
            concept_data["reference"],
            concept_data["status"]
        ))
        print(f"✅ Added: {concept_data['concept']} ({concept_data['service']})")
        added_count += 1
    else:
        print(f"⏭️  Skipped (exists): {concept_data['concept']}")
        skipped_count += 1

# Commit
conn.commit()

# Print summary
cursor.execute("""
    SELECT status, COUNT(*) as count 
    FROM implementation_concepts 
    GROUP BY status 
    ORDER BY count DESC
""")
print(f"\n=== Summary ===")
print(f"Added: {added_count} new concepts")
print(f"Skipped: {skipped_count} (already exist)")
print(f"\nTotal concepts by status:")
for row in cursor.fetchall():
    print(f"  {row[0]}: {row[1]} concepts")

cursor.execute("""
    SELECT service, COUNT(*) as count 
    FROM implementation_concepts 
    GROUP BY service 
    ORDER BY count DESC
""")
print(f"\nConcepts by Service:")
for row in cursor.fetchall():
    print(f"  {row[0]}: {row[1]} concepts")

total = cursor.execute("SELECT COUNT(*) FROM implementation_concepts").fetchone()[0]
print(f"\n✅ Database updated! Total concepts: {total}")

conn.close()

