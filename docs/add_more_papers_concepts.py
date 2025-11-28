#!/usr/bin/env python3
"""
Script to add additional concepts discovered from deeper analysis of the 6 papers
These are more granular and technical concepts that may have been missed in the first pass
"""

import sqlite3
from pathlib import Path

# Database path
db_path = Path(__file__).parent / "implementation_tracker.db"

# Create connection
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Additional concepts from deeper paper analysis
additional_concepts = [
    # From AI Conversational Tutors (2025) - Deeper analysis
    {
        "concept": "10-Dimensional Quality Evaluation Framework",
        "service": "diagnostic_module",
        "implementation_details": "Evaluate tutors on: Value of Learning, Supportive Style, Quality of Communication, Quality of Interaction, Coherence of Dialogue, Tutor Initiative, Richness of Vocabulary, Value of Feedback, Learner Initiative, Learner Richness Vocabulary. Can be used to assess system quality.",
        "reference": "Avouris, N. (2025). AI Conversational Tutors in Foreign Language Learning: A Mixed-Methods Evaluation Study. Proceedings 14th Panhellenic Conference ICT in Education.",
        "status": "planned"
    },
    {
        "concept": "Automated Quality Assessment with LLM Evaluators",
        "service": "diagnostic_module",
        "implementation_details": "Use multiple LLMs as artificial experts to evaluate tutor quality with inter-rater reliability (Cronbach's Alpha). Can be implemented as periodic quality checks.",
        "reference": "Avouris, N. (2025). AI Conversational Tutors in Foreign Language Learning: A Mixed-Methods Evaluation Study. Proceedings 14th Panhellenic Conference ICT in Education.",
        "status": "planned"
    },
    {
        "concept": "Dialogue Representation Modes",
        "service": "orchestrator",
        "implementation_details": "Support multiple dialogue views: full chat sequence, last turn only, or no text. Can be added as UI configuration option.",
        "reference": "Avouris, N. (2025). AI Conversational Tutors in Foreign Language Learning: A Mixed-Methods Evaluation Study. Proceedings 14th Panhellenic Conference ICT in Education.",
        "status": "planned"
    },
    {
        "concept": "Pause Detection and Timeout Handling",
        "service": "orchestrator",
        "implementation_details": "Detect long pauses in hands-free mode and handle timeouts appropriately. Can be implemented in audio processing pipeline.",
        "reference": "Avouris, N. (2025). AI Conversational Tutors in Foreign Language Learning: A Mixed-Methods Evaluation Study. Proceedings 14th Panhellenic Conference ICT in Education.",
        "status": "planned"
    },
    {
        "concept": "Selectable Word Translation",
        "service": "pedagogical_policy",
        "implementation_details": "Make words in tutor responses selectable for translation. Can be added to UI layer for vocabulary learning.",
        "reference": "Avouris, N. (2025). AI Conversational Tutors in Foreign Language Learning: A Mixed-Methods Evaluation Study. Proceedings 14th Panhellenic Conference ICT in Education.",
        "status": "planned"
    },
    {
        "concept": "Chat History Export and Review",
        "service": "orchestrator",
        "implementation_details": "Allow students to export chat history and review previous conversations. Can be added to conversation_store service.",
        "reference": "Avouris, N. (2025). AI Conversational Tutors in Foreign Language Learning: A Mixed-Methods Evaluation Study. Proceedings 14th Panhellenic Conference ICT in Education.",
        "status": "planned"
    },
    {
        "concept": "Privacy Consent for Voice Data",
        "service": "orchestrator",
        "implementation_details": "Request explicit consent for storing and using voice conversational data. Can be added to onboarding flow.",
        "reference": "Avouris, N. (2025). AI Conversational Tutors in Foreign Language Learning: A Mixed-Methods Evaluation Study. Proceedings 14th Panhellenic Conference ICT in Education.",
        "status": "planned"
    },
    
    # From TUTORING: Instruction-Grounded (2023) - Deeper analysis
    {
        "concept": "Global and Local Progress Recognition",
        "service": "student_model",
        "implementation_details": "Global progress: which instruction is current (index). Local progress: fraction of dialogue turns completed for current instruction (0-1). Can be added to progress tracking.",
        "reference": "Chae, H., et al. (2023). TUTORING: Instruction-Grounded Conversational Agent for Language Learners. AAAI.",
        "status": "planned"
    },
    {
        "concept": "Multi-task Learning for Tutoring",
        "service": "pedagogical_policy",
        "implementation_details": "Jointly learn response generation, action codes, and progress recognition. Can be implemented as auxiliary tasks in prompt composition.",
        "reference": "Chae, H., et al. (2023). TUTORING: Instruction-Grounded Conversational Agent for Language Learners. AAAI.",
        "status": "planned"
    },
    {
        "concept": "Instruction Sequence Management",
        "service": "learning_path",
        "implementation_details": "Manage fixed sequence of N instructions, where each turn aligns with one instruction. Can be integrated into learning path navigation.",
        "reference": "Chae, H., et al. (2023). TUTORING: Instruction-Grounded Conversational Agent for Language Learners. AAAI.",
        "status": "planned"
    },
    {
        "concept": "Debugging Tools for Action Codes",
        "service": "pedagogical_policy",
        "implementation_details": "Visualization tool to identify generated action codes and progress recognition results. Can be added as developer/admin feature.",
        "reference": "Chae, H., et al. (2023). TUTORING: Instruction-Grounded Conversational Agent for Language Learners. AAAI.",
        "status": "planned"
    },
    
    # From Adaptive Learning Path Navigation (2023) - Deeper analysis
    {
        "concept": "Actor-Critic Network Architecture",
        "service": "learning_path",
        "implementation_details": "Actor network determines policy π(·|s), critic network estimates value function V(s). Can be used in RL implementation for learning paths.",
        "reference": "Chen, J.-Y., Saeedvand, S., & Lai, I.-W. (2023). Adaptive Learning Path Navigation Based on Knowledge Tracing and Reinforcement Learning. PRIME AI.",
        "status": "planned"
    },
    {
        "concept": "Replay Buffer for Policy Learning",
        "service": "learning_path",
        "implementation_details": "Store past experiences (state, action, reward, next_state, entropy) for sample-efficient learning. Can be added to RL agent.",
        "reference": "Chen, J.-Y., Saeedvand, S., & Lai, I.-W. (2023). Adaptive Learning Path Navigation Based on Knowledge Tracing and Reinforcement Learning. PRIME AI.",
        "status": "planned"
    },
    {
        "concept": "Clipped Surrogate Objective Function",
        "service": "learning_path",
        "implementation_details": "PPO's clipped objective to prevent large policy updates. Can be used in RL training to ensure stability.",
        "reference": "Chen, J.-Y., Saeedvand, S., & Lai, I.-W. (2023). Adaptive Learning Path Navigation Based on Knowledge Tracing and Reinforcement Learning. PRIME AI.",
        "status": "planned"
    },
    {
        "concept": "Advantage Function Calculation",
        "service": "learning_path",
        "implementation_details": "Calculate A(s,a) = Q(s,a) - V(s) using discounted cumulative rewards. Can be used to improve policy gradient estimates.",
        "reference": "Chen, J.-Y., Saeedvand, S., & Lai, I.-W. (2023). Adaptive Learning Path Navigation Based on Knowledge Tracing and Reinforcement Learning. PRIME AI.",
        "status": "planned"
    },
    {
        "concept": "Distance Parameter in Reward Function",
        "service": "learning_path",
        "implementation_details": "Calculate distance d_t = β - APR_t between current knowledge and learning goal. Used to scale reward magnitude.",
        "reference": "Chen, J.-Y., Saeedvand, S., & Lai, I.-W. (2023). Adaptive Learning Path Navigation Based on Knowledge Tracing and Reinforcement Learning. PRIME AI.",
        "status": "planned"
    },
    {
        "concept": "Learning Path Diversity Metric (DIV)",
        "service": "learning_path",
        "implementation_details": "Calculate diversity as 1 - (intersection/union) of learning paths across students. Can be used to evaluate system diversity.",
        "reference": "Chen, J.-Y., Saeedvand, S., & Lai, I.-W. (2023). Adaptive Learning Path Navigation Based on Knowledge Tracing and Reinforcement Learning. PRIME AI.",
        "status": "planned"
    },
    {
        "concept": "Knowledge State Vector Representation",
        "service": "student_model",
        "implementation_details": "Represent knowledge state as vector s_t = [s_1,t, s_2,t, ..., s_J,t] where each element is probability of correct answer for exercise j. Can enhance AKT output format.",
        "reference": "Chen, J.-Y., Saeedvand, S., & Lai, I.-W. (2023). Adaptive Learning Path Navigation Based on Knowledge Tracing and Reinforcement Learning. PRIME AI.",
        "status": "planned"
    },
    
    # From Empowering Personalized Learning (2024) - Deeper analysis
    {
        "concept": "2PL IRT Model for Exercise Selection",
        "service": "learning_path",
        "implementation_details": "Use 2-parameter logistic IRT: P(correct|a,d,θ) = 1/(1+exp(-a(θ-d))). Select exercises with P≈0.5 for optimal challenge. Can replace current exercise selection.",
        "reference": "Park, M., et al. (2024). Empowering Personalized Learning through a Conversation-based Tutoring System with Student Modeling. CHI EA '24.",
        "status": "planned"
    },
    {
        "concept": "Dual-Prompt Architecture (Base + Personalized)",
        "service": "pedagogical_policy",
        "implementation_details": "Base prompt provides foundational guidelines, personalized prompt adapts to student state. Can be implemented in PromptComposer layers.",
        "reference": "Park, M., et al. (2024). Empowering Personalized Learning through a Conversation-based Tutoring System with Student Modeling. CHI EA '24.",
        "status": "planned"
    },
    {
        "concept": "Cyclical Framework for Prompt Updates",
        "service": "pedagogical_policy",
        "implementation_details": "Student assessment → System prompt → Tutoring session → Summary → Updated assessment. Can be implemented as feedback loop.",
        "reference": "Park, M., et al. (2024). Empowering Personalized Learning through a Conversation-based Tutoring System with Student Modeling. CHI EA '24.",
        "status": "planned"
    },
    {
        "concept": "Summary Prompt for Session Analysis",
        "service": "diagnostic_module",
        "implementation_details": "Generate session summaries with: specific topics covered, action items for response level, action items for learning style. Can be added to diagnostic service.",
        "reference": "Park, M., et al. (2024). Empowering Personalized Learning through a Conversation-based Tutoring System with Student Modeling. CHI EA '24.",
        "status": "planned"
    },
    {
        "concept": "Tutor Action Labeling (10 Categories)",
        "service": "pedagogical_policy",
        "implementation_details": "Label tutor utterances: Greetings, Engagement, Scaffolding, Asking for explanation, Summary, etc. Can be used for analysis and improvement.",
        "reference": "Park, M., et al. (2024). Empowering Personalized Learning through a Conversation-based Tutoring System with Student Modeling. CHI EA '24.",
        "status": "planned"
    },
    {
        "concept": "Onboarding Survey for Initial Profile",
        "service": "student_model",
        "implementation_details": "Collect learning style preferences, confidence levels, demographic data before first session. Can be added to user onboarding.",
        "reference": "Park, M., et al. (2024). Empowering Personalized Learning through a Conversation-based Tutoring System with Student Modeling. CHI EA '24.",
        "status": "planned"
    },
    {
        "concept": "Pre-test and Post-test Comparison",
        "service": "student_model",
        "implementation_details": "Administer pre-test before tutoring, post-test after, compare proficiency levels (θ_pre vs θ_post). Can be added to assessment flow.",
        "reference": "Park, M., et al. (2024). Empowering Personalized Learning through a Conversation-based Tutoring System with Student Modeling. CHI EA '24.",
        "status": "planned"
    },
    
    # From AI Intelligent Tutoring Robots (2019) - Deeper analysis
    {
        "concept": "Multi-Modal Data Fusion",
        "service": "diagnostic_module",
        "implementation_details": "Fuse audio, visual, tactile, inertial sensor data for comprehensive student perception. Can be added to diagnostic pipeline.",
        "reference": "Yang, J., & Zhang, B. (2019). Artificial Intelligence in Intelligent Tutoring Robots: A Systematic Review and Design Guidelines. Applied Sciences, 9(19), 3993.",
        "status": "planned"
    },
    {
        "concept": "Pixel-Level Data Fusion",
        "service": "diagnostic_module",
        "implementation_details": "Fuse multiple visual sensor inputs at pixel level for moving object tracking. Can be used for gesture/expression analysis.",
        "reference": "Yang, J., & Zhang, B. (2019). Artificial Intelligence in Intelligent Tutoring Robots: A Systematic Review and Design Guidelines. Applied Sciences, 9(19), 3993.",
        "status": "planned"
    },
    {
        "concept": "Affective Computing for Emotion Recognition",
        "service": "diagnostic_module",
        "implementation_details": "Use multi-modal fusion (audio-visual) with metabolic variables (heart rate, eye tracking) for emotion detection. Can enhance affective state assessment.",
        "reference": "Yang, J., & Zhang, B. (2019). Artificial Intelligence in Intelligent Tutoring Robots: A Systematic Review and Design Guidelines. Applied Sciences, 9(19), 3993.",
        "status": "planned"
    },
    {
        "concept": "Knowledge Graph Disparity Analysis",
        "service": "student_model",
        "implementation_details": "Compare student's knowledge graph with expert's target graph to identify weak/missing knowledge connections. Can be added to CEFR progress calculation.",
        "reference": "Yang, J., & Zhang, B. (2019). Artificial Intelligence in Intelligent Tutoring Robots: A Systematic Review and Design Guidelines. Applied Sciences, 9(19), 3993.",
        "status": "planned"
    },
    {
        "concept": "Student Model as Virtual Agent",
        "service": "student_model",
        "implementation_details": "Instantiate student model as virtual agent that reacts to tutor actions, allowing simulation of teaching strategies before real interaction.",
        "reference": "Yang, J., & Zhang, B. (2019). Artificial Intelligence in Intelligent Tutoring Robots: A Systematic Review and Design Guidelines. Applied Sciences, 9(19), 3993.",
        "status": "planned"
    },
    {
        "concept": "Multi-Objective Optimization for Teaching",
        "service": "pedagogical_policy",
        "implementation_details": "Balance long-term education objectives, short-term curriculum objectives, and student personal objectives. Can be used in strategy selection.",
        "reference": "Yang, J., & Zhang, B. (2019). Artificial Intelligence in Intelligent Tutoring Robots: A Systematic Review and Design Guidelines. Applied Sciences, 9(19), 3993.",
        "status": "planned"
    },
    {
        "concept": "Virtual Reality Scene Construction",
        "service": "orchestrator",
        "implementation_details": "Generate VR/AR environments with virtual tutors for immersive learning. Can be added as advanced scenario mode.",
        "reference": "Yang, J., & Zhang, B. (2019). Artificial Intelligence in Intelligent Tutoring Robots: A Systematic Review and Design Guidelines. Applied Sciences, 9(19), 3993.",
        "status": "planned"
    },
    {
        "concept": "Constraint-Based Modeling",
        "service": "student_model",
        "implementation_details": "Build annotated domain model showing gap between student and expert knowledge, plus bug library of misconceptions. Can complement AKT.",
        "reference": "Yang, J., & Zhang, B. (2019). Artificial Intelligence in Intelligent Tutoring Robots: A Systematic Review and Design Guidelines. Applied Sciences, 9(19), 3993.",
        "status": "planned"
    },
    
    # From ITS Survey (2018) - Deeper analysis
    {
        "concept": "Effect Size Measurement (0.79 for Human Tutoring)",
        "service": "student_model",
        "implementation_details": "Measure system effectiveness using effect size (standard deviations improvement). Target: approach 0.79 effect size of human tutoring.",
        "reference": "Alkhatlan, A., & Kalita, J. K. (2018). Intelligent Tutoring Systems: A Comprehensive Historical Survey with Recent Developments. ACM Computing Surveys.",
        "status": "planned"
    },
    {
        "concept": "Step-based vs Answer-based Tutoring",
        "service": "pedagogical_policy",
        "implementation_details": "Step-based tutoring (0.76 effect size) monitors solution steps, answer-based only checks final answer. Can be implemented in diagnostic granularity.",
        "reference": "Alkhatlan, A., & Kalita, J. K. (2018). Intelligent Tutoring Systems: A Comprehensive Historical Survey with Recent Developments. ACM Computing Surveys.",
        "status": "planned"
    },
    {
        "concept": "Rule-Based Model Tracing",
        "service": "student_model",
        "implementation_details": "Model students as rule-based agents, trace execution of rules to infer student states. Can be used for step-by-step error detection.",
        "reference": "Alkhatlan, A., & Kalita, J. K. (2018). Intelligent Tutoring Systems: A Comprehensive Historical Survey with Recent Developments. ACM Computing Surveys.",
        "status": "planned"
    }
]

# Check if concept already exists and insert
added_count = 0
skipped_count = 0

for concept_data in additional_concepts:
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

