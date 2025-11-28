#!/usr/bin/env python3
"""
Script to create SQLite database tracking ITS implementation concepts
Maps pedagogical concepts to services and academic references
"""

import sqlite3
from pathlib import Path

# Database path
db_path = Path(__file__).parent / "implementation_tracker.db"

# Create connection
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Create table
cursor.execute("""
CREATE TABLE IF NOT EXISTS implementation_concepts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    concept TEXT NOT NULL,
    service TEXT NOT NULL,
    implementation_details TEXT,
    reference TEXT NOT NULL,
    status TEXT DEFAULT 'implemented',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")

# Insert data based on implemented features
implementations = [
    # Knowledge Tracing
    {
        "concept": "Bayesian Knowledge Tracing (BKT)",
        "service": "student_model",
        "implementation_details": "Fallback algorithm in knowledge_tracer/bkt_tracer.py. Tracks P(L), P(T), P(G), P(S) for each skill.",
        "reference": "Corbett, A. T., & Anderson, J. R. (1994). Knowledge tracing: Modeling the acquisition of procedural knowledge. User modeling and user-adapted interaction, 4(4), 253-278.",
        "status": "implemented"
    },
    {
        "concept": "Attentive Knowledge Tracing (AKT)",
        "service": "student_model",
        "implementation_details": "Primary algorithm in knowledge_tracer/akt_tracer.py. Uses attention mechanisms, temporal decay, and contextual features.",
        "reference": "Ghosh, A., Heffernan, N., & Lan, A. S. (2020). Context-aware attentive knowledge tracing. In Proceedings of the 26th ACM SIGKDD (pp. 2330-2339).",
        "status": "implemented"
    },
    
    # CEFR Integration
    {
        "concept": "CEFR Level Mapping",
        "service": "student_model",
        "implementation_details": "SKILL_CEFR_MAP in skill_registry.py maps skills to CEFR levels (A1-C2). calculate_cefr_progress aggregates AKT probabilities.",
        "reference": "Council of Europe. (2001). Common European Framework of Reference for Languages: Learning, teaching, assessment. Cambridge University Press.",
        "status": "implemented"
    },
    
    # Pedagogical Strategies
    {
        "concept": "Scaffolding (Zone of Proximal Development)",
        "service": "pedagogical_policy",
        "implementation_details": "StrategyLayer provides adaptive support: Explicit correction for beginners (<30% mastery), Implicit recast for intermediate (30-70%), Minimal for advanced (>70%).",
        "reference": "Vygotsky, L. S. (1978). Mind in society: The development of higher psychological processes. Harvard University Press.",
        "status": "implemented"
    },
    {
        "concept": "Input Hypothesis (i+1)",
        "service": "pedagogical_policy",
        "implementation_details": "CHALLENGE strategy in PolicyEngine exposes students to content slightly above current level when mastery > 70%.",
        "reference": "Krashen, S. D. (1985). The input hypothesis: Issues and implications. Longman.",
        "status": "implemented"
    },
    {
        "concept": "Recast-based Correction",
        "service": "pedagogical_policy",
        "implementation_details": "ScaffoldingType.IMPLICIT in strategy_layer.py provides subtle error correction by reformulating correct responses naturally.",
        "reference": "Long, M. H. (2006). Problems in SLA. Lawrence Erlbaum Associates.",
        "status": "implemented"
    },
    {
        "concept": "Instruction-Grounded Conversations",
        "service": "pedagogical_policy",
        "implementation_details": "PromptComposer uses layered architecture (StudentStateLayer, StrategyLayer, FocusLayer, AffectiveLayer) to build pedagogically-informed prompts.",
        "reference": "Kumar, G., et al. (2022). Instruction-grounded conversational agents. arXiv preprint arXiv:2211.09020.",
        "status": "implemented"
    },
    
    # Diagnostic & Analysis
    {
        "concept": "Automatic Error Detection",
        "service": "diagnostic_module",
        "implementation_details": "DiagnosticLLMClient.analyze_grammar uses LLM to identify grammar, vocabulary, and syntax errors with skill tagging.",
        "reference": "Rei, M., & Yannakoudakis, H. (2016). Compositional sequence labeling models for error detection. In Proceedings of ACL (pp. 1181-1191).",
        "status": "implemented"
    },
    {
        "concept": "Skill Tagging (Granular Assessment)",
        "service": "diagnostic_module",
        "implementation_details": "LLM prompt updated to identify specific skill_id for both errors and correct usages, enabling fine-grained knowledge updates.",
        "reference": "Benedetto, L., et al. (2020). R2DE: a NLP approach to estimating IRT parameters of newly generated questions. In Proceedings of LAK (pp. 412-421).",
        "status": "implemented"
    },
    {
        "concept": "Multi-faceted Student Assessment",
        "service": "diagnostic_module",
        "implementation_details": "Analyzers for grammar, vocabulary, complexity, and progress provide cognitive and linguistic evaluation.",
        "reference": "Desmarais, M. C., & Baker, R. S. (2012). A review of recent advances in learner and skill modeling in intelligent learning environments. User Modeling and User-Adapted Interaction, 22(1-2), 9-38.",
        "status": "implemented"
    },
    
    # Learning Path & Spaced Repetition
    {
        "concept": "Spaced Repetition System (SRS)",
        "service": "learning_path",
        "implementation_details": "SpacedRepetitionSystem in spaced_repetition.py calculates optimal review intervals using forgetting curves and mastery levels.",
        "reference": "Ebbinghaus, H. (1885). Memory: A contribution to experimental psychology. Teachers College, Columbia University.",
        "status": "implemented"
    },
    {
        "concept": "Zone of Proximal Development (ZPD) Calculator",
        "service": "learning_path",
        "implementation_details": "ZPDCalculator in zpd_calculator.py identifies skills within learner's ZPD based on mastery and prerequisites.",
        "reference": "Vygotsky, L. S. (1978). Mind in society: The development of higher psychological processes. Harvard University Press.",
        "status": "implemented"
    },
    {
        "concept": "Adaptive Learning Path Navigation",
        "service": "learning_path",
        "implementation_details": "LearningPathNavigator combines SRS and ZPD to recommend next skills, balancing new content and review.",
        "reference": "Clement, B., et al. (2015). Multi-armed bandits for intelligent tutoring systems. Journal of Educational Data Mining, 7(2), 20-48.",
        "status": "implemented"
    },
    
    # Deep Learning for Student Modeling
    {
        "concept": "Deep Learning for Knowledge States",
        "service": "student_model",
        "implementation_details": "AKT uses attention mechanisms and temporal features to model complex knowledge states beyond simple probabilities.",
        "reference": "Piech, C., et al. (2015). Deep knowledge tracing. In Advances in neural information processing systems (pp. 505-513).",
        "status": "implemented"
    },
    {
        "concept": "Contextual Feature Engineering",
        "service": "student_model",
        "implementation_details": "AKT processes contextual features (difficulty, complexity, time_since_last, error_type_severity) for nuanced mastery updates.",
        "reference": "Ghosh, A., Heffernan, N., & Lan, A. S. (2020). Context-aware attentive knowledge tracing. KDD.",
        "status": "implemented"
    },
    
    # Affective Computing
    {
        "concept": "Emotional State Modulation",
        "service": "pedagogical_policy",
        "implementation_details": "AffectiveLayer and PolicyEngine.get_emotional_modulation adapt tone and pace based on student emotional state (motivated, frustrated, confused, etc.).",
        "reference": "D'Mello, S., & Graesser, A. (2012). Dynamics of affective states during complex learning. Learning and Instruction, 22(2), 145-157.",
        "status": "implemented"
    },
    
    # Reinforcement Learning (Planned)
    {
        "concept": "Reinforcement Learning Policy (Thompson Sampling)",
        "service": "pedagogical_policy",
        "implementation_details": "Planned: Multi-Armed Bandit using Thompson Sampling to dynamically select optimal teaching strategies based on student feedback.",
        "reference": "Clement, B., et al. (2015). Multi-armed bandits for intelligent tutoring systems. JEDM, 7(2), 20-48.",
        "status": "planned"
    },
    {
        "concept": "Contextual Bandits for Personalization",
        "service": "pedagogical_policy",
        "implementation_details": "Planned: Context-aware RL that learns per-CEFR-level policies (e.g., 'A1 students respond better to TEACH').",
        "reference": "Mandel, T., et al. (2014). Offline policy evaluation across representations with applications to educational games. AAMAS.",
        "status": "planned"
    },
    
    # Progress Recognition
    {
        "concept": "Automatic Progress Recognition",
        "service": "student_model",
        "implementation_details": "calculate_cefr_progress detects when student masters a level (>80%) and estimates transition to next CEFR level.",
        "reference": "Käser, T., et al. (2017). Modeling exploration strategies to predict student performance. In EDM (pp. 31-40).",
        "status": "implemented"
    },
    
    # Orchestration
    {
        "concept": "Microservices Architecture for ITS",
        "service": "orchestrator",
        "implementation_details": "Orchestrator coordinates student_model, pedagogical_policy, diagnostic_module, and learning_path as independent services.",
        "reference": "Aleven, V., et al. (2016). Instruction based on adaptive learning technologies. In Handbook of research on learning and instruction (pp. 522-560).",
        "status": "implemented"
    },
    {
        "concept": "Background Knowledge Update (Non-blocking)",
        "service": "orchestrator",
        "implementation_details": "_analyze_and_update_knowledge runs as asyncio background task to avoid blocking conversation flow.",
        "reference": "VanLehn, K. (2011). The relative effectiveness of human tutoring, intelligent tutoring systems, and other tutoring systems. Educational Psychologist, 46(4), 197-221.",
        "status": "implemented"
    }
]

# Insert all implementations
for impl in implementations:
    cursor.execute("""
        INSERT INTO implementation_concepts 
        (concept, service, implementation_details, reference, status)
        VALUES (?, ?, ?, ?, ?)
    """, (
        impl["concept"],
        impl["service"],
        impl["implementation_details"],
        impl["reference"],
        impl["status"]
    ))

# Create indexes for faster queries
cursor.execute("CREATE INDEX IF NOT EXISTS idx_service ON implementation_concepts(service)")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_status ON implementation_concepts(status)")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_concept ON implementation_concepts(concept)")

# Commit and close
conn.commit()

# Print summary
cursor.execute("SELECT service, COUNT(*) as count FROM implementation_concepts GROUP BY service ORDER BY count DESC")
print("\n=== Implementation Tracker Database Created ===")
print(f"Location: {db_path}")
print("\nConcepts by Service:")
for row in cursor.fetchall():
    print(f"  {row[0]}: {row[1]} concepts")

cursor.execute("SELECT status, COUNT(*) as count FROM implementation_concepts GROUP BY status")
print("\nStatus Summary:")
for row in cursor.fetchall():
    print(f"  {row[0]}: {row[1]} concepts")

print("\n✅ Database created successfully!")
print(f"\nTo query: sqlite3 {db_path}")
print("Example queries:")
print("  SELECT concept, service FROM implementation_concepts WHERE status='implemented';")
print("  SELECT * FROM implementation_concepts WHERE service='pedagogical_policy';")

conn.close()

