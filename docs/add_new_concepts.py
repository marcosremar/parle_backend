#!/usr/bin/env python3
"""
Script to add new language learning concepts discovered from research
Adds concepts from SLA theory and CALL research applicable to the ITS
"""

import sqlite3
from pathlib import Path

# Database path
db_path = Path(__file__).parent / "implementation_tracker.db"

# Create connection
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# New concepts from research
new_concepts = [
    # Interaction Hypothesis & Negotiation
    {
        "concept": "Interaction Hypothesis (Negotiation of Meaning)",
        "service": "orchestrator",
        "implementation_details": "System should facilitate negotiation through clarification requests, confirmation checks, and comprehension checks. Can be implemented in conversation flow by detecting confusion and asking 'Did you mean...?' or 'Could you clarify?'",
        "reference": "Long, M. H. (1996). The role of the linguistic environment in second language acquisition. In W. C. Ritchie & T. K. Bhatia (Eds.), Handbook of second language acquisition (pp. 413-468). Academic Press.",
        "status": "planned"
    },
    {
        "concept": "Comprehension Checks",
        "service": "pedagogical_policy",
        "implementation_details": "System can periodically ask 'Do you understand?' or 'Can you explain what I just said?' to verify comprehension. Implemented via StrategyLayer when mastery is low.",
        "reference": "Long, M. H. (1983). Native speaker/non-native speaker conversation and the negotiation of comprehensible input. Applied Linguistics, 4(2), 126-141.",
        "status": "planned"
    },
    
    # Noticing & Input Enhancement
    {
        "concept": "Noticing Hypothesis",
        "service": "pedagogical_policy",
        "implementation_details": "System can highlight grammatical structures in responses (e.g., bold key phrases) to draw attention. Can be implemented in PromptComposer by adding formatting instructions for target structures.",
        "reference": "Schmidt, R. (1990). The role of consciousness in second language learning. Applied Linguistics, 11(2), 129-158.",
        "status": "planned"
    },
    {
        "concept": "Input Enhancement",
        "service": "pedagogical_policy",
        "implementation_details": "Make target linguistic features more salient through typographical enhancement (bold, italics) or repetition. Can be added to StrategyLayer for TEACH mode.",
        "reference": "Sharwood Smith, M. (1993). Input enhancement in instructed SLA: Theoretical bases. Studies in Second Language Acquisition, 15(2), 165-179.",
        "status": "planned"
    },
    
    # Output Hypothesis
    {
        "concept": "Output Hypothesis (Pushed Output)",
        "service": "pedagogical_policy",
        "implementation_details": "System should push students to produce language beyond their comfort zone. CHALLENGE strategy already does this, but can be enhanced with explicit 'Try to use...' prompts.",
        "reference": "Swain, M. (1985). Communicative competence: Some roles of comprehensible input and comprehensible output in its development. In S. Gass & C. Madden (Eds.), Input in second language acquisition (pp. 235-253). Newbury House.",
        "status": "partially_implemented"
    },
    {
        "concept": "Pushed Output Tasks",
        "service": "learning_path",
        "implementation_details": "Design tasks that require students to use specific grammatical structures. Can be integrated into scenario design and LearningPathNavigator recommendations.",
        "reference": "Swain, M. (1995). Three functions of output in second language learning. In G. Cook & B. Seidlhofer (Eds.), Principle and practice in applied linguistics (pp. 125-144). Oxford University Press.",
        "status": "planned"
    },
    
    # Assessment
    {
        "concept": "Formative Assessment",
        "service": "diagnostic_module",
        "implementation_details": "Continuous assessment during learning (not just at end). DiagnosticModule already provides this through real-time error analysis. Can be enhanced with progress dashboards.",
        "reference": "Black, P., & Wiliam, D. (1998). Assessment and classroom learning. Assessment in Education: Principles, Policy & Practice, 5(1), 7-74.",
        "status": "implemented"
    },
    {
        "concept": "Adaptive Feedback Timing",
        "service": "pedagogical_policy",
        "implementation_details": "Adjust feedback timing based on student state: immediate for beginners, delayed for advanced. Can be implemented in PolicyEngine based on mastery level.",
        "reference": "Shute, V. J. (2008). Focus on formative feedback. Review of Educational Research, 78(1), 153-189.",
        "status": "planned"
    },
    
    # Task-Based Learning
    {
        "concept": "Task-Based Language Teaching (TBLT)",
        "service": "orchestrator",
        "implementation_details": "Scenarios should be designed as meaningful tasks (e.g., 'Order food at a restaurant') rather than grammar drills. Current scenario system supports this but can be enhanced with task complexity metrics.",
        "reference": "Ellis, R. (2003). Task-based language learning and teaching. Oxford University Press.",
        "status": "partially_implemented"
    },
    {
        "concept": "Task Complexity Sequencing",
        "service": "learning_path",
        "implementation_details": "Sequence tasks from simple to complex based on cognitive load. Can be integrated into LearningPathNavigator using difficulty scores from skill_registry.",
        "reference": "Robinson, P. (2001). Task complexity, task difficulty, and task production: Exploring interactions in a componential framework. Applied Linguistics, 22(1), 27-57.",
        "status": "planned"
    },
    
    # Mobile & Gamification
    {
        "concept": "Mobile-Assisted Language Learning (MALL)",
        "service": "orchestrator",
        "implementation_details": "System architecture supports mobile access via REST API. Can be enhanced with push notifications for spaced repetition reminders.",
        "reference": "Kukulska-Hulme, A., & Shield, L. (2008). An overview of mobile assisted language learning: From content delivery to supported collaboration and interaction. ReCALL, 20(3), 271-289.",
        "status": "partially_implemented"
    },
    {
        "concept": "Gamification in Language Learning",
        "service": "student_model",
        "implementation_details": "Progress tracking (CEFR levels, mastery scores) provides gamification elements. Can be enhanced with badges, streaks, and leaderboards using existing mastery data.",
        "reference": "Reinders, H., & Wattana, S. (2015). Affect and willingness to communicate in digital game-based learning. ReCALL, 27(1), 38-57.",
        "status": "partially_implemented"
    },
    
    # Affective Factors
    {
        "concept": "Affective Filter Hypothesis",
        "service": "pedagogical_policy",
        "implementation_details": "EmotionalState modulation in AffectiveLayer already addresses this. Can be enhanced with anxiety detection and adaptive difficulty adjustment.",
        "reference": "Krashen, S. D. (1982). Principles and practice in second language acquisition. Pergamon Press.",
        "status": "implemented"
    },
    {
        "concept": "Motivation Maintenance",
        "service": "pedagogical_policy",
        "implementation_details": "System can provide encouragement messages and celebrate milestones. Can be added to AffectiveLayer when emotional_state is MOTIVATED or CONFIDENT.",
        "reference": "Dörnyei, Z. (2001). Motivational strategies in the language classroom. Cambridge University Press.",
        "status": "planned"
    },
    
    # Cognitive Load
    {
        "concept": "Cognitive Load Theory",
        "service": "pedagogical_policy",
        "implementation_details": "System should manage intrinsic (content difficulty), extraneous (presentation), and germane (learning) load. Current CEFR-based adaptation addresses this, but can be refined with explicit load metrics.",
        "reference": "Sweller, J. (1988). Cognitive load during problem solving: Effects on learning. Cognitive Science, 12(2), 257-285.",
        "status": "partially_implemented"
    },
    
    # Conversational Agents
    {
        "concept": "Conversational Agents for Language Learning",
        "service": "orchestrator",
        "implementation_details": "Current system IS a conversational agent. Can be enhanced with persona consistency, topic coherence, and cultural awareness.",
        "reference": "Fryer, L., et al. (2019). Chatbot learning partners: Connecting learning experiences, interest and competence. Computers in Human Behavior, 93, 279-289.",
        "status": "implemented"
    },
    {
        "concept": "Dialogue Management for SLA",
        "service": "orchestrator",
        "implementation_details": "Conversation history and context management already implemented. Can be enhanced with explicit dialogue acts (question, clarification, correction) tracking.",
        "reference": "Griol, D., et al. (2014). A statistical approach to spoken dialog systems design and evaluation. Speech Communication, 60, 1-20.",
        "status": "partially_implemented"
    },
    
    # Vocabulary Acquisition
    {
        "concept": "Incidental Vocabulary Learning",
        "service": "pedagogical_policy",
        "implementation_details": "System can introduce new vocabulary naturally in conversation. Can be enhanced with vocabulary tracking and frequency-based selection.",
        "reference": "Nation, I. S. P. (2001). Learning vocabulary in another language. Cambridge University Press.",
        "status": "planned"
    },
    {
        "concept": "Vocabulary Frequency Lists",
        "service": "learning_path",
        "implementation_details": "Prioritize high-frequency words in vocabulary_basic skills. Can be integrated into skill_registry with frequency data.",
        "reference": "Nation, I. S. P., & Waring, R. (1997). Vocabulary size, text coverage and word lists. In N. Schmitt & M. McCarthy (Eds.), Vocabulary: Description, acquisition and pedagogy (pp. 6-19). Cambridge University Press.",
        "status": "planned"
    },
    
    # Error Analysis
    {
        "concept": "Error Analysis & Interlanguage",
        "service": "diagnostic_module",
        "implementation_details": "DiagnosticModule identifies errors, but can be enhanced to track interlanguage development patterns (systematic errors that indicate learning stage).",
        "reference": "Corder, S. P. (1967). The significance of learners' errors. International Review of Applied Linguistics, 5(4), 161-170.",
        "status": "partially_implemented"
    },
    {
        "concept": "Error Correction Strategies",
        "service": "pedagogical_policy",
        "implementation_details": "Current ScaffoldingType (IMPLICIT/EXPLICIT) addresses this. Can be enhanced with metallinguistic feedback (explaining rules) for explicit mode.",
        "reference": "Lyster, R., & Ranta, L. (1997). Corrective feedback and learner uptake: Negotiation of form in communicative classrooms. Studies in Second Language Acquisition, 19(1), 37-66.",
        "status": "implemented"
    },
    
    # Individual Differences
    {
        "concept": "Learning Style Adaptation",
        "service": "pedagogical_policy",
        "implementation_details": "System can adapt to visual, auditory, or kinesthetic preferences. Can be added to StudentStateLayer based on user profile.",
        "reference": "Oxford, R. L. (2003). Language learning styles and strategies: An overview. GALA, 1-25.",
        "status": "planned"
    },
    {
        "concept": "Aptitude-Treatment Interaction",
        "service": "pedagogical_policy",
        "implementation_details": "Match teaching strategies to individual aptitudes (e.g., analytical learners benefit from explicit rules). Can be integrated into PolicyEngine with aptitude assessment.",
        "reference": "Robinson, P. (2002). Individual differences and instructed language learning. John Benjamins.",
        "status": "planned"
    }
]

# Check if concept already exists
for concept_data in new_concepts:
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
    else:
        print(f"⏭️  Skipped (exists): {concept_data['concept']}")

# Commit
conn.commit()

# Print summary
cursor.execute("""
    SELECT status, COUNT(*) as count 
    FROM implementation_concepts 
    GROUP BY status 
    ORDER BY count DESC
""")
print("\n=== Updated Summary ===")
for row in cursor.fetchall():
    print(f"  {row[0]}: {row[1]} concepts")

cursor.execute("""
    SELECT service, COUNT(*) as count 
    FROM implementation_concepts 
    GROUP BY service 
    ORDER BY count DESC
""")
print("\nConcepts by Service:")
for row in cursor.fetchall():
    print(f"  {row[0]}: {row[1]} concepts")

print(f"\n✅ Database updated! Total concepts: {cursor.execute('SELECT COUNT(*) FROM implementation_concepts').fetchone()[0]}")

conn.close()

