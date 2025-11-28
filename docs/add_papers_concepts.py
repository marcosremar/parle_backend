#!/usr/bin/env python3
"""
Script to add concepts discovered from analyzing papers in the papers/ directory
These concepts are specifically related to language learning and can improve our ITS
"""

import sqlite3
from pathlib import Path

# Database path
db_path = Path(__file__).parent / "implementation_tracker.db"

# Create connection
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# New concepts from paper analysis
paper_concepts = [
    # From AI Conversational Tutors (2025)
    {
        "concept": "Communication Breakdown Repair",
        "service": "orchestrator",
        "implementation_details": "System detects unfinished utterances (timeouts) and repairs communication by asking 'Would you like to finish your thought?' or suggesting completions. Can be implemented in orchestrator by detecting silence/pause patterns.",
        "reference": "Avouris, N. (2025). AI Conversational Tutors in Foreign Language Learning: A Mixed-Methods Evaluation Study. Proceedings 14th Panhellenic Conference ICT in Education.",
        "status": "planned"
    },
    {
        "concept": "Contextual Error Correction with Explanations",
        "service": "pedagogical_policy",
        "implementation_details": "Provide corrections that include: (1) appraisal/encouragement, (2) remarks on errors, (3) suggestion for re-phrasing, (4) follow-up question. Can be enhanced in StrategyLayer for explicit correction mode.",
        "reference": "Avouris, N. (2025). AI Conversational Tutors in Foreign Language Learning: A Mixed-Methods Evaluation Study. Proceedings 14th Panhellenic Conference ICT in Education.",
        "status": "partially_implemented"
    },
    {
        "concept": "Daily/Weekly Feedback Reports",
        "service": "student_model",
        "implementation_details": "Generate summary reports with observations, suggestions, and overall score after each session. Can be implemented as endpoint that aggregates interaction history and mastery data.",
        "reference": "Avouris, N. (2025). AI Conversational Tutors in Foreign Language Learning: A Mixed-Methods Evaluation Study. Proceedings 14th Panhellenic Conference ICT in Education.",
        "status": "planned"
    },
    {
        "concept": "Gamification Elements (Error Indicators)",
        "service": "diagnostic_module",
        "implementation_details": "Visual indicators (green/yellow signs) for error-free vs error-containing utterances. Can be added to diagnostic response and displayed in UI.",
        "reference": "Avouris, N. (2025). AI Conversational Tutors in Foreign Language Learning: A Mixed-Methods Evaluation Study. Proceedings 14th Panhellenic Conference ICT in Education.",
        "status": "planned"
    },
    {
        "concept": "Cultural Authenticity in Conversations",
        "service": "pedagogical_policy",
        "implementation_details": "Ensure AI responses reflect cultural nuances and authentic language use. Can be added to PromptComposer with cultural context layer.",
        "reference": "Avouris, N. (2025). AI Conversational Tutors in Foreign Language Learning: A Mixed-Methods Evaluation Study. Proceedings 14th Panhellenic Conference ICT in Education.",
        "status": "planned"
    },
    
    # From TUTORING: Instruction-Grounded (2023)
    {
        "concept": "Instruction-Grounded Response Generation",
        "service": "pedagogical_policy",
        "implementation_details": "Generate tutor responses grounded to educational instructions. Current PromptComposer already does this via scenario system, but can be enhanced with explicit instruction tracking.",
        "reference": "Chae, H., et al. (2023). TUTORING: Instruction-Grounded Conversational Agent for Language Learners. AAAI.",
        "status": "partially_implemented"
    },
    {
        "concept": "Teaching Action Codes",
        "service": "pedagogical_policy",
        "implementation_details": "Explicit action codes: [Correction], [Confirmation], [Others] for dialogue action, and [Transition] for instruction changes. Can be added to PolicyEngine output and used in prompt composition.",
        "reference": "Chae, H., et al. (2023). TUTORING: Instruction-Grounded Conversational Agent for Language Learners. AAAI.",
        "status": "planned"
    },
    {
        "concept": "Instruction Transition Detection",
        "service": "pedagogical_policy",
        "implementation_details": "Monitor when current instruction should transition to next. Can be implemented by tracking mastery progress and dialogue turns per instruction.",
        "reference": "Chae, H., et al. (2023). TUTORING: Instruction-Grounded Conversational Agent for Language Learners. AAAI.",
        "status": "planned"
    },
    {
        "concept": "Progress Recognition Tasks",
        "service": "student_model",
        "implementation_details": "Multi-task learning to infer teaching action and progress simultaneously. Can be added as auxiliary task in knowledge tracing.",
        "reference": "Chae, H., et al. (2023). TUTORING: Instruction-Grounded Conversational Agent for Language Learners. AAAI.",
        "status": "planned"
    },
    
    # From Adaptive Learning Path Navigation (2023)
    {
        "concept": "Entropy-enhanced Proximal Policy Optimization (EPPO)",
        "service": "learning_path",
        "implementation_details": "Enhanced RL algorithm for learning path recommendation with better exploration. Can replace current RL implementation in LearningPathNavigator.",
        "reference": "Chen, J.-Y., Saeedvand, S., & Lai, I.-W. (2023). Adaptive Learning Path Navigation Based on Knowledge Tracing and Reinforcement Learning. PRIME AI.",
        "status": "planned"
    },
    {
        "concept": "Learning Gain Calculation",
        "service": "student_model",
        "implementation_details": "Calculate learning gain as difference between current and previous knowledge level (APR_t - APR_{t-1}). Can be added to StudentModelService to track improvement over time.",
        "reference": "Chen, J.-Y., Saeedvand, S., & Lai, I.-W. (2023). Adaptive Learning Path Navigation Based on Knowledge Tracing and Reinforcement Learning. PRIME AI.",
        "status": "planned"
    },
    {
        "concept": "Diversity Penalty in Learning Paths",
        "service": "learning_path",
        "implementation_details": "Ensure diversity in recommended materials by penalizing repetition. Can be added to LearningPathNavigator reward function.",
        "reference": "Chen, J.-Y., Saeedvand, S., & Lai, I.-W. (2023). Adaptive Learning Path Navigation Based on Knowledge Tracing and Reinforcement Learning. PRIME AI.",
        "status": "planned"
    },
    {
        "concept": "Average Performance Rate (APR)",
        "service": "student_model",
        "implementation_details": "Calculate average probability of correct answers across all available exercises. Can be used as overall knowledge level metric.",
        "reference": "Chen, J.-Y., Saeedvand, S., & Lai, I.-W. (2023). Adaptive Learning Path Navigation Based on Knowledge Tracing and Reinforcement Learning. PRIME AI.",
        "status": "planned"
    },
    
    # From Empowering Personalized Learning (2024)
    {
        "concept": "Item Response Theory (IRT) for Proficiency Assessment",
        "service": "student_model",
        "implementation_details": "Use IRT model to assess proficiency levels in knowledge concepts. Can complement AKT for more accurate assessment.",
        "reference": "Park, M., et al. (2024). Empowering Personalized Learning through a Conversation-based Tutoring System with Student Modeling. CHI EA '24.",
        "status": "planned"
    },
    {
        "concept": "Metacognition Tracking",
        "service": "student_model",
        "implementation_details": "Track self-reported self-assessment vs actual proficiency to measure metacognitive awareness. Can be added to User model and compared with AKT mastery.",
        "reference": "Park, M., et al. (2024). Empowering Personalized Learning through a Conversation-based Tutoring System with Student Modeling. CHI EA '24.",
        "status": "planned"
    },
    {
        "concept": "Session-End Summaries",
        "service": "diagnostic_module",
        "implementation_details": "Generate LLM-based summaries at end of session evaluating cognitive state, affective state, and learning style demonstrated. Can be added to DiagnosticModule.",
        "reference": "Park, M., et al. (2024). Empowering Personalized Learning through a Conversation-based Tutoring System with Student Modeling. CHI EA '24.",
        "status": "planned"
    },
    {
        "concept": "Interventional Messages",
        "service": "pedagogical_policy",
        "implementation_details": "Personalized messages based on affective state (e.g., 'I see you're frustrated, let's try a different approach'). Can be added to AffectiveLayer.",
        "reference": "Park, M., et al. (2024). Empowering Personalized Learning through a Conversation-based Tutoring System with Student Modeling. CHI EA '24.",
        "status": "planned"
    },
    {
        "concept": "Felder-Silverman Learning Style Model",
        "service": "pedagogical_policy",
        "implementation_details": "Classify learning styles into 16 categories based on Perception (sensory/intuitive), Processing (active/reflective), Understanding (sequential/global). Can replace current simple learning style adaptation.",
        "reference": "Park, M., et al. (2024). Empowering Personalized Learning through a Conversation-based Tutoring System with Student Modeling. CHI EA '24.",
        "status": "planned"
    },
    {
        "concept": "Adaptive Exercise Selection",
        "service": "learning_path",
        "implementation_details": "Select exercises based on student assessment (cognitive, affective, learning style). Can be enhanced in LearningPathNavigator.",
        "reference": "Park, M., et al. (2024). Empowering Personalized Learning through a Conversation-based Tutoring System with Student Modeling. CHI EA '24.",
        "status": "partially_implemented"
    },
    
    # From AI Intelligent Tutoring Robots (2019)
    {
        "concept": "Scene Construction",
        "service": "orchestrator",
        "implementation_details": "Adapt virtual environment context based on student profile. Can be integrated into scenario system to dynamically adjust context.",
        "reference": "Yang, J., & Zhang, B. (2019). Artificial Intelligence in Intelligent Tutoring Robots: A Systematic Review and Design Guidelines. Applied Sciences, 9(19), 3993.",
        "status": "planned"
    },
    {
        "concept": "Knowledge Graph Integration",
        "service": "student_model",
        "implementation_details": "Use knowledge graphs to represent domain knowledge and student knowledge state. Can enhance skill_registry with graph relationships.",
        "reference": "Yang, J., & Zhang, B. (2019). Artificial Intelligence in Intelligent Tutoring Robots: A Systematic Review and Design Guidelines. Applied Sciences, 9(19), 3993.",
        "status": "planned"
    },
    {
        "concept": "Perception-Planning-Action Framework",
        "service": "orchestrator",
        "implementation_details": "Transform teaching-learning relationship into perception (diagnose student), planning (select strategy), action (execute teaching). Current orchestrator already follows this pattern but can be made more explicit.",
        "reference": "Yang, J., & Zhang, B. (2019). Artificial Intelligence in Intelligent Tutoring Robots: A Systematic Review and Design Guidelines. Applied Sciences, 9(19), 3993.",
        "status": "partially_implemented"
    },
    
    # From ITS Survey (2018)
    {
        "concept": "Model Tracing",
        "service": "student_model",
        "implementation_details": "Monitor student's problem-solving steps incrementally and intervene when mistakes are made. Can be enhanced in DiagnosticModule to track solution steps.",
        "reference": "Alkhatlan, A., & Kalita, J. K. (2018). Intelligent Tutoring Systems: A Comprehensive Historical Survey with Recent Developments. ACM Computing Surveys.",
        "status": "partially_implemented"
    },
    {
        "concept": "Step-based vs Substep-based Tutoring",
        "service": "pedagogical_policy",
        "implementation_details": "Step-based tutoring (0.76 effect size) is almost as effective as human tutoring. Can be implemented by providing feedback at appropriate granularity levels.",
        "reference": "Alkhatlan, A., & Kalita, J. K. (2018). Intelligent Tutoring Systems: A Comprehensive Historical Survey with Recent Developments. ACM Computing Surveys.",
        "status": "planned"
    },
    {
        "concept": "Confidence Scoring for Speech Recognition",
        "service": "diagnostic_module",
        "implementation_details": "Measure confidence in interpretation of student speech (spoken precision index). Can be added to STT response and used to trigger clarification requests.",
        "reference": "Avouris, N. (2025). AI Conversational Tutors in Foreign Language Learning: A Mixed-Methods Evaluation Study. Proceedings 14th Panhellenic Conference ICT in Education.",
        "status": "planned"
    },
    {
        "concept": "Voice Interaction Modes (Hands-free vs Manual)",
        "service": "orchestrator",
        "implementation_details": "Support both automatic (hands-free) and manual (button-press) voice interaction modes. Can be added as session configuration option.",
        "reference": "Avouris, N. (2025). AI Conversational Tutors in Foreign Language Learning: A Mixed-Methods Evaluation Study. Proceedings 14th Panhellenic Conference ICT in Education.",
        "status": "planned"
    },
    {
        "concept": "Annotated Chat Transcripts",
        "service": "diagnostic_module",
        "implementation_details": "Provide annotated transcripts with error markings, corrections, and explanations. Can be generated by DiagnosticModule and stored for review.",
        "reference": "Avouris, N. (2025). AI Conversational Tutors in Foreign Language Learning: A Mixed-Methods Evaluation Study. Proceedings 14th Panhellenic Conference ICT in Education.",
        "status": "planned"
    },
    {
        "concept": "Alternative Phrasing Suggestions",
        "service": "pedagogical_policy",
        "implementation_details": "Suggest alternative ways to express student's message. Can be added to DiagnosticModule response when errors are detected.",
        "reference": "Avouris, N. (2025). AI Conversational Tutors in Foreign Language Learning: A Mixed-Methods Evaluation Study. Proceedings 14th Panhellenic Conference ICT in Education.",
        "status": "planned"
    },
    {
        "concept": "Critical Thinking Guidance",
        "service": "pedagogical_policy",
        "implementation_details": "Focus on critical thinking over learner's argument, guiding into further reasoning. Can be added to CHALLENGE strategy prompts.",
        "reference": "Avouris, N. (2025). AI Conversational Tutors in Foreign Language Learning: A Mixed-Methods Evaluation Study. Proceedings 14th Panhellenic Conference ICT in Education.",
        "status": "planned"
    }
]

# Check if concept already exists and insert
added_count = 0
skipped_count = 0

for concept_data in paper_concepts:
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

