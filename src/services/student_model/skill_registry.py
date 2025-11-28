"""
Skill Registry - Catálogo de habilidades linguísticas
Define habilidades padrão e seus parâmetros AKT (padrão) e BKT (fallback)
"""

from typing import Dict, List, Optional, Any
from .knowledge_tracer.bkt_tracer import BayesianKnowledgeTracer
from .knowledge_tracer.akt_tracer import AttentiveKnowledgeTracer


# Mapa de Skills para Níveis CEFR
# Baseado nos descritores oficiais do CECR Volume Complémentaire (2021)
SKILL_CEFR_MAP = {
    "A1": [
        # Gramática - Verbos
        "verb_conjugation_present",
        "verb_to_be",
        "verb_ser_estar",
        "pronouns_personal",
        "pronouns_possessive",
        
        # Gramática - Artigos e Preposições
        "articles_definite",
        "articles_indefinite",
        "prepositions_basic",
        "prepositions_time",
        
        # Vocabulário
        "vocabulary_basic",
        "vocabulary_family",
        "vocabulary_numbers",
        "vocabulary_colors",
        "vocabulary_days_week",
        "vocabulary_months",
        "vocabulary_body_parts",
        "vocabulary_food_basic",
        "greeting_formal",
        "greeting_informal",
        "vocabulary_common_objects",
        
        # Pronúncia
        "pronunciation_basic",
        "pronunciation_vowels",
        
        # Interação
        "interaction_simple_questions",
        "interaction_personal_info",
        "interaction_basic_needs",
        
        # Produção
        "production_simple_phrases",
        "production_self_introduction",
        "production_describe_place"
    ],
    "A2": [
        # Gramática - Verbos
        "verb_conjugation_past",
        "verb_past_irregular",
        "verb_past_perfect",
        "verb_past_imperfect",
        "verb_reflexive",
        "verb_imperative",
        
        # Gramática - Estruturas
        "prepositions_location",
        "comparatives",
        "superlatives",
        "adverbs_frequency",
        "adverbs_manner",
        "adjectives_basic",
        "adjectives_comparison",
        
        # Vocabulário
        "vocabulary_intermediate",
        "vocabulary_daily_routine",
        "vocabulary_travel",
        "vocabulary_shopping",
        "vocabulary_weather",
        "vocabulary_time_expressions",
        "vocabulary_activities",
        "vocabulary_places",
        
        # Pronúncia
        "pronunciation_intonation_basic",
        "pronunciation_stress_basic",
        
        # Interação
        "interaction_simple_transactions",
        "interaction_daily_tasks",
        "interaction_social_brief",
        
        # Produção
        "production_describe_family",
        "production_describe_work",
        "production_describe_past_events",
        "production_simple_narrative",
        
        # Compreensão
        "comprehension_simple_announcements",
        "comprehension_basic_instructions"
    ],
    "B1": [
        # Gramática - Verbos
        "verb_conjugation_future",
        "verb_conditional",
        "verb_subjunctive_intro",
        "modal_verbs",
        "verb_gerund",
        "verb_infinitive",
        
        # Gramática - Estruturas
        "discourse_markers",
        "passive_voice",
        "relative_clauses",
        "conditional_sentences",
        "reported_speech",
        "connectors_basic",
        "conjunctions",
        
        # Vocabulário
        "vocabulary_feelings",
        "vocabulary_opinions",
        "vocabulary_abstract_concepts",
        "vocabulary_work_professional",
        "vocabulary_education",
        "vocabulary_health",
        "vocabulary_entertainment",
        "vocabulary_media",
        
        # Pronúncia
        "pronunciation_intonation_intermediate",
        "pronunciation_rhythm",
        
        # Interação
        "interaction_travel_situations",
        "interaction_unprepared_conversation",
        "interaction_exchange_opinions",
        "interaction_clarify_meaning",
        
        # Produção
        "production_narrate_experiences",
        "production_explain_reasons",
        "production_describe_dreams_hopes",
        "production_summarize_story",
        "production_simple_essay",
        
        # Compreensão
        "comprehension_radio_tv_clear",
        "comprehension_work_related_texts",
        "comprehension_personal_letters"
    ],
    "B2": [
        # Gramática - Verbos
        "subjunctive_present",
        "subjunctive_past",
        "verb_compound_tenses",
        "verb_sequence_tenses",
        
        # Gramática - Estruturas
        "complex_sentence_structure",
        "connectors_advanced",
        "discourse_cohesion",
        "nuanced_expressions",
        "hypothetical_structures",
        
        # Vocabulário
        "vocabulary_advanced",
        "idioms_common",
        "vocabulary_formal_register",
        "vocabulary_technical",
        "vocabulary_social_issues",
        "vocabulary_culture",
        "vocabulary_abstract_topics",
        
        # Pronúncia
        "pronunciation_advanced",
        "pronunciation_native_like",
        "pronunciation_intonation_advanced",
        
        # Interação
        "interaction_fluent_conversation",
        "interaction_defend_opinions",
        "interaction_manage_misunderstandings",
        "interaction_multiple_participants",
        
        # Produção
        "production_detailed_descriptions",
        "production_express_advantages_disadvantages",
        "production_clear_arguments",
        "production_essay_report",
        "production_literary_texts",
        
        # Compreensão
        "comprehension_complex_arguments",
        "comprehension_tv_films",
        "comprehension_contemporary_literature",
        "comprehension_technical_discussions"
    ],
    "C1": [
        # Gramática - Estruturas Avançadas
        "complex_grammatical_structures",
        "subtle_grammatical_nuances",
        "stylistic_variation",
        "register_adaptation",
        
        # Vocabulário
        "vocabulary_proficient",
        "idioms_advanced",
        "collocations_advanced",
        "vocabulary_precise",
        "vocabulary_specialized",
        "vocabulary_academic",
        "vocabulary_professional",
        
        # Pronúncia
        "pronunciation_fluent_natural",
        "pronunciation_intonation_precise",
        "pronunciation_stress_advanced",
        
        # Interação
        "interaction_spontaneous_fluent",
        "interaction_flexible_effective",
        "interaction_precise_opinions",
        "interaction_link_contributions",
        "interaction_manage_communication_problems",
        
        # Produção
        "production_complex_topics",
        "production_well_structured",
        "production_appropriate_style",
        "production_different_text_types",
        "production_clear_detailed_descriptions",
        
        # Compreensão
        "comprehension_unstructured_discourse",
        "comprehension_implicit_meanings",
        "comprehension_long_complex_texts",
        "comprehension_specialized_articles",
        "comprehension_style_differences",
        
        # Mediação
        "mediation_texts_complex",
        "mediation_concepts_abstract",
        "mediation_communication_effective"
    ],
    "C2": [
        # Gramática - Maestria
        "grammar_native_like",
        "grammar_subtle_nuances",
        "grammar_stylistic_mastery",
        
        # Vocabulário
        "vocabulary_native",
        "idioms_rare",
        "dialect_variations",
        "literary_expressions",
        "vocabulary_precise_shades",
        "vocabulary_colloquialisms",
        
        # Pronúncia
        "pronunciation_native_equivalent",
        "pronunciation_all_phonological_features",
        "pronunciation_prosody_mastery",
        
        # Interação
        "interaction_effortless",
        "interaction_idiomatic_expressions",
        "interaction_precise_nuances",
        "interaction_repair_seamlessly",
        "interaction_any_context",
        
        # Produção
        "production_clear_fluent_style",
        "production_logical_structure",
        "production_complex_reports",
        "production_critical_reviews",
        "production_precise_nuances",
        
        # Compreensão
        "comprehension_effortless",
        "comprehension_any_register",
        "comprehension_abstract_complex",
        "comprehension_implicit_cultural",
        "comprehension_any_accent",
        
        # Mediação
        "mediation_effective_natural",
        "mediation_sociocultural_differences",
        "mediation_precise_nuances"
    ]
}


def get_skills_by_cefr(level: str) -> List[str]:
    """
    Retorna lista de skills pertencentes a um nível CEFR
    
    Args:
        level: Nível CEFR (A1, A2, etc.)
        
    Returns:
        Lista de skill_ids
    """
    return SKILL_CEFR_MAP.get(level, [])


# Parâmetros BKT por categoria de habilidade
BKT_PARAMS_BY_CATEGORY = {
    "grammar": {
        "verb_present": {
            "p_L0": 0.4,
            "p_T": 0.2,
            "p_F": 0.05,
            "p_G": 0.8,
            "p_S": 0.2
        },
        "verb_past": {
            "p_L0": 0.2,
            "p_T": 0.15,
            "p_F": 0.08,
            "p_G": 0.75,
            "p_S": 0.25
        },
        "verb_future": {
            "p_L0": 0.1,
            "p_T": 0.12,
            "p_F": 0.1,
            "p_G": 0.7,
            "p_S": 0.3
        },
        "articles": {
            "p_L0": 0.3,
            "p_T": 0.18,
            "p_F": 0.06,
            "p_G": 0.78,
            "p_S": 0.22
        },
        "prepositions": {
            "p_L0": 0.25,
            "p_T": 0.16,
            "p_F": 0.07,
            "p_G": 0.72,
            "p_S": 0.28
        }
    },
    "vocabulary": {
        "basic": {
            "p_L0": 0.6,
            "p_T": 0.25,
            "p_F": 0.03,
            "p_G": 0.85,
            "p_S": 0.15
        },
        "intermediate": {
            "p_L0": 0.3,
            "p_T": 0.2,
            "p_F": 0.05,
            "p_G": 0.8,
            "p_S": 0.2
        },
        "advanced": {
            "p_L0": 0.1,
            "p_T": 0.15,
            "p_F": 0.08,
            "p_G": 0.75,
            "p_S": 0.25
        }
    },
    "pronunciation": {
        "basic": {
            "p_L0": 0.5,
            "p_T": 0.22,
            "p_F": 0.04,
            "p_G": 0.82,
            "p_S": 0.18
        },
        "advanced": {
            "p_L0": 0.2,
            "p_T": 0.18,
            "p_F": 0.06,
            "p_G": 0.78,
            "p_S": 0.22
        }
    }
}


# Mapa de dificuldade IRT por nível CEFR (Rasch Model-based)
CEFR_DIFFICULTY_MAP = {
    "A1": 0.2,
    "A2": 0.4,
    "B1": 0.6,
    "B2": 0.75,
    "C1": 0.85,
    "C2": 0.95
}

# Habilidades padrão do sistema
DEFAULT_SKILLS = [
    {
        "skill_id": "verb_conjugation_present",
        "name": "Conjugação de Verbos no Presente",
        "category": "grammar",
        "difficulty": "beginner",
        "description": "Uso correto de verbos no tempo presente"
    },
    {
        "skill_id": "verb_to_be",
        "name": "Verbo Ser/Estar",
        "category": "grammar",
        "difficulty": "beginner",
        "description": "Uso correto dos verbos ser e estar"
    },
    {
        "skill_id": "verb_conjugation_past",
        "name": "Conjugação de Verbos no Passado",
        "category": "grammar",
        "difficulty": "intermediate",
        "description": "Uso correto de verbos no pretérito perfeito e imperfeito"
    },
    {
        "skill_id": "verb_conjugation_future",
        "name": "Conjugação de Verbos no Futuro",
        "category": "grammar",
        "difficulty": "intermediate",
        "description": "Uso correto de verbos no futuro do presente"
    },
    {
        "skill_id": "articles_definite",
        "name": "Artigos Definidos",
        "category": "grammar",
        "difficulty": "beginner",
        "description": "Uso correto de artigos definidos (o, a, os, as)"
    },
    {
        "skill_id": "articles_indefinite",
        "name": "Artigos Indefinidos",
        "category": "grammar",
        "difficulty": "beginner",
        "description": "Uso correto de artigos indefinidos (um, uma, uns, umas)"
    },
    {
        "skill_id": "prepositions_basic",
        "name": "Preposições Básicas",
        "category": "grammar",
        "difficulty": "beginner",
        "description": "Uso de preposições comuns (em, de, para, com)"
    },
    {
        "skill_id": "vocabulary_family",
        "name": "Vocabulário de Família",
        "category": "vocabulary",
        "difficulty": "beginner",
        "description": "Palavras relacionadas a membros da família"
    },
    {
        "skill_id": "vocabulary_numbers",
        "name": "Números",
        "category": "vocabulary",
        "difficulty": "beginner",
        "description": "Números cardinais e ordinais"
    },
    {
        "skill_id": "greeting_formal",
        "name": "Cumprimentos Formais",
        "category": "vocabulary",
        "difficulty": "beginner",
        "description": "Expressões de cumprimento formal"
    },
    {
        "skill_id": "greeting_informal",
        "name": "Cumprimentos Informais",
        "category": "vocabulary",
        "difficulty": "beginner",
        "description": "Expressões de cumprimento informal"
    },
    {
        "skill_id": "vocabulary_basic",
        "name": "Vocabulário Básico",
        "category": "vocabulary",
        "difficulty": "beginner",
        "description": "Palavras e expressões básicas do dia a dia"
    },
    {
        "skill_id": "vocabulary_intermediate",
        "name": "Vocabulário Intermediário",
        "category": "vocabulary",
        "difficulty": "intermediate",
        "description": "Palavras e expressões de nível intermediário"
    },
    {
        "skill_id": "pronunciation_basic",
        "name": "Pronúncia Básica",
        "category": "pronunciation",
        "difficulty": "beginner",
        "description": "Pronúncia correta de sons básicos do português"
    }
]


def get_akt_params(skill_id: str) -> Dict[str, Any]:
    """
    Retorna parâmetros AKT para uma habilidade específica
    
    AKT é o padrão. Usa os mesmos parâmetros base do BKT mas com
    configurações adicionais para attention mechanisms.
    
    Args:
        skill_id: ID da habilidade
        
    Returns:
        Dicionário com parâmetros AKT
    """
    # Obter parâmetros base (BKT)
    bkt_params = get_bkt_params(skill_id)
    
    # Adicionar parâmetros específicos do AKT
    akt_params = bkt_params.copy()
    akt_params.update({
        'temporal_decay': 0.95,  # Decay para interações antigas
        'attention_window': 30,   # Número de interações para considerar (aumentado para melhor context-awareness)
        'adaptation_rate': 0.1,   # Taxa de adaptação de parâmetros
        'folibi_enabled': True,   # FoLiBi forgetting-aware linear bias
        'linear_decay_factor': 0.3  # Fator de decay linear para FoLiBi
    })
    
    return akt_params


def get_bkt_params(skill_id: str) -> Dict[str, float]:
    """
    Retorna parâmetros BKT para uma habilidade específica
    
    Args:
        skill_id: ID da habilidade
        
    Returns:
        Dicionário com parâmetros BKT
    """
    # Tentar encontrar parâmetros específicos
    for category, skills in BKT_PARAMS_BY_CATEGORY.items():
        if skill_id in skills:
            return skills[skill_id]
    
    # Tentar encontrar por categoria
    if "verb" in skill_id.lower():
        if "past" in skill_id.lower():
            return BKT_PARAMS_BY_CATEGORY["grammar"]["verb_past"]
        elif "future" in skill_id.lower():
            return BKT_PARAMS_BY_CATEGORY["grammar"]["verb_future"]
        else:
            return BKT_PARAMS_BY_CATEGORY["grammar"]["verb_present"]
    elif "vocabulary" in skill_id.lower():
        if "intermediate" in skill_id.lower() or "advanced" in skill_id.lower():
            return BKT_PARAMS_BY_CATEGORY["vocabulary"]["intermediate"]
        else:
            return BKT_PARAMS_BY_CATEGORY["vocabulary"]["basic"]
    elif "pronunciation" in skill_id.lower():
        if "advanced" in skill_id.lower():
            return BKT_PARAMS_BY_CATEGORY["pronunciation"]["advanced"]
        else:
            return BKT_PARAMS_BY_CATEGORY["pronunciation"]["basic"]
    elif "article" in skill_id.lower():
        return BKT_PARAMS_BY_CATEGORY["grammar"]["articles"]
    elif "preposition" in skill_id.lower():
        return BKT_PARAMS_BY_CATEGORY["grammar"]["prepositions"]
    
    # Parâmetros padrão se não encontrar
    return BayesianKnowledgeTracer._get_default_params()


def get_default_skills() -> List[Dict[str, str]]:
    """Retorna lista de habilidades padrão"""
    return DEFAULT_SKILLS


def extract_linguistic_features(skill_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extrai features linguísticas de uma skill e contexto
    
    Features extraídas:
    - Para verbos: tense, person, number, mood, aspect
    - Para vocabulário: register (formal/informal), domain (travel/family/work)
    - Para pronúncia: phoneme_category, stress_pattern
    
    Args:
        skill_id: ID da habilidade
        context: Contexto da interação (pode conter user_text, error_type, etc.)
        
    Returns:
        Dicionário com features linguísticas
    """
    features = {}
    
    # Features baseadas no skill_id
    skill_lower = skill_id.lower()
    
    # Extrair tense para verbos
    if "verb" in skill_lower:
        if "present" in skill_lower:
            features["tense"] = "present"
        elif "past" in skill_lower:
            features["tense"] = "past"
        elif "future" in skill_lower:
            features["tense"] = "future"
        elif "conditional" in skill_lower:
            features["tense"] = "conditional"
        elif "subjunctive" in skill_lower:
            features["tense"] = "subjunctive"
            if "past" in skill_lower:
                features["aspect"] = "past"
            else:
                features["aspect"] = "present"
        
        # Person e number podem ser inferidos do contexto se disponível
        if "user_text" in context:
            user_text = context["user_text"].lower()
            # Detectar pessoa (simplificado)
            if any(word in user_text for word in ["eu", "me", "minha"]):
                features["person"] = "1st"
            elif any(word in user_text for word in ["você", "te", "sua", "vocês"]):
                features["person"] = "2nd"
            elif any(word in user_text for word in ["ele", "ela", "eles", "elas", "o", "a", "os", "as"]):
                features["person"] = "3rd"
        
        # Mood (modo verbal)
        if "subjunctive" in skill_lower:
            features["mood"] = "subjunctive"
        elif "conditional" in skill_lower:
            features["mood"] = "conditional"
        else:
            features["mood"] = "indicative"
    
    # Features para vocabulário
    if "vocabulary" in skill_lower:
        if "formal" in skill_lower or "greeting_formal" in skill_lower:
            features["register"] = "formal"
        elif "informal" in skill_lower or "greeting_informal" in skill_lower:
            features["register"] = "informal"
        else:
            features["register"] = "neutral"
        
        # Domain (domínio semântico)
        if "family" in skill_lower:
            features["domain"] = "family"
        elif "travel" in skill_lower:
            features["domain"] = "travel"
        elif "numbers" in skill_lower:
            features["domain"] = "numbers"
        elif "feelings" in skill_lower:
            features["domain"] = "emotions"
        elif "daily_routine" in skill_lower:
            features["domain"] = "daily_life"
        else:
            features["domain"] = "general"
    
    # Features para pronúncia
    if "pronunciation" in skill_lower:
        if "basic" in skill_lower:
            features["phoneme_category"] = "basic"
        elif "advanced" in skill_lower:
            features["phoneme_category"] = "advanced"
    
    # Features para artigos
    if "article" in skill_lower:
        features["grammatical_category"] = "article"
        if "definite" in skill_lower:
            features["article_type"] = "definite"
        elif "indefinite" in skill_lower:
            features["article_type"] = "indefinite"
    
    # Features para preposições
    if "preposition" in skill_lower:
        features["grammatical_category"] = "preposition"
        if "location" in skill_lower:
            features["preposition_type"] = "location"
        else:
            features["preposition_type"] = "basic"
    
    # Number (singular/plural) - pode ser inferido do contexto
    if "user_text" in context:
        user_text = context["user_text"].lower()
        # Detectar plural (simplificado)
        plural_indicators = ["os", "as", "eles", "elas", "nós", "vocês"]
        if any(indicator in user_text for indicator in plural_indicators):
            features["number"] = "plural"
        else:
            features["number"] = "singular"
    
    return features


def get_relevant_skills_for_context(
    user_text: str,
    cefr_level: str,
    context_type: str = "production",
    max_skills: int = 50
) -> List[str]:
    """
    Filtra skills relevantes baseado em contexto e nível CEFR.

    Estratégia:
    - Janela centrada no nível atual (nível-1, nível, nível+1)
    - Hard cap: se nível <= B1, NUNCA inclui C1/C2 (máximo até B2)
    - Quando nível é desconhecido, assume A1–A2
    - Core vs Exploratório:
        - Core = níveis até o nível atual
        - Exploratório = níveis acima (janela superior)
      As skills core vêm primeiro na lista, então têm prioridade
      quando há truncamento por max_skills.
    
    Args:
        user_text: Texto do aluno (para heurística)
        cefr_level: Nível CEFR atual do aluno (A1, A2, B1, B2, C1, C2)
        context_type: Tipo de contexto ("production", "comprehension", "interaction")
        max_skills: Número máximo de skills a retornar
        
    Returns:
        Lista filtrada de skill_ids relevantes (core + exploratórias)
    """
    from typing import List
    
    levels = ["A1", "A2", "B1", "B2", "C1", "C2"]
    core_levels: List[str] = []
    exploratory_levels: List[str] = []

    if cefr_level in levels:
        idx = levels.index(cefr_level)

        # Hard cap: se nível <= B1, não passamos de B2 (ignoramos C1/C2)
        b1_index = levels.index("B1")
        b2_index = levels.index("B2")

        if idx <= b1_index:
            # Janela centrada no nível, mas limitada até B2
            start_idx = max(0, idx - 1)
            end_idx = min(idx + 1, b2_index)
        else:
            # Níveis mais altos podem ir até C2 normalmente
            start_idx = max(0, idx - 1)
            end_idx = min(idx + 1, len(levels) - 1)

        window_levels = levels[start_idx : end_idx + 1]

        # Core: até o nível atual; Exploratório: acima do nível atual dentro da janela
        for level in window_levels:
            if levels.index(level) <= idx:
                core_levels.append(level)
            else:
                exploratory_levels.append(level)
    else:
        # Nível desconhecido → fase inicial / diagnóstico simples: A1–A2 apenas (nenhum avançado)
        core_levels = ["A1", "A2"]
        exploratory_levels = []
    
    # Agregar skills, preservando a ordem: core primeiro, depois exploratório
    skills: List[str] = []
    for level in core_levels:
        skills.extend(SKILL_CEFR_MAP.get(level, []))
    for level in exploratory_levels:
        skills.extend(SKILL_CEFR_MAP.get(level, []))
    
    # 2. Filtrar por tipo de contexto
    if context_type == "production":
        # Apenas skills de produção (gramática, vocabulário, escrita)
        skills = [s for s in skills if any(x in s for x in [
            "verb_", "vocabulary_", "article_", "preposition_", 
            "pronoun_", "adjective_", "adverb_", "production_",
            "greeting_", "comparative", "superlative", "modal_",
            "discourse_", "passive_", "subjunctive_", "conditional"
        ])]
    elif context_type == "comprehension":
        # Skills de compreensão
        skills = [s for s in skills if "comprehension_" in s]
    elif context_type == "interaction":
        # Skills de interação
        skills = [s for s in skills if "interaction_" in s]
    elif context_type == "mediation":
        # Skills de mediação
        skills = [s for s in skills if "mediation_" in s]
    
    # 3. Heurística: detectar features básicas do texto
    text_lower = user_text.lower()
    
    # Detectar tempo verbal
    past_indicators = ["fui", "foi", "foram", "estava", "estavam", "tinha", "tinham", "fez", "fizeram"]
    future_indicators = ["vou", "vai", "vão", "irei", "será", "serão"]
    present_indicators = ["sou", "é", "são", "estou", "está", "estão", "tenho", "tem", "têm"]
    
    if any(word in text_lower for word in past_indicators):
        # Priorizar skills de passado
        past_skills = [s for s in skills if "past" in s or "verb_past" in s]
        other_skills = [s for s in skills if s not in past_skills]
        skills = past_skills + other_skills
    elif any(word in text_lower for word in future_indicators):
        # Priorizar skills de futuro
        future_skills = [s for s in skills if "future" in s or "verb_future" in s]
        other_skills = [s for s in skills if s not in future_skills]
        skills = future_skills + other_skills
    
    # Detectar pronomes
    if any(word in text_lower for word in ["eu", "me", "minha", "meu"]):
        # Priorizar skills de primeira pessoa
        pass  # Já coberto por verbos
    
    # 4. Limitar número de skills
    if len(skills) > max_skills:
        # Priorizar: verbos > vocabulário > outros
        verb_skills = [s for s in skills if "verb_" in s]
        vocab_skills = [s for s in skills if "vocabulary_" in s]
        other_skills = [s for s in skills if s not in verb_skills and s not in vocab_skills]
        
        # Distribuir proporcionalmente
        verb_count = min(len(verb_skills), max_skills // 2)
        vocab_count = min(len(vocab_skills), max_skills // 3)
        other_count = max_skills - verb_count - vocab_count
        
        skills = verb_skills[:verb_count] + vocab_skills[:vocab_count] + other_skills[:other_count]
    
    return skills[:max_skills]


def get_skill_difficulty(skill_id: str) -> float:
    """
    Retorna dificuldade IRT (Rasch Model-based) para uma skill específica
    
    A dificuldade é baseada no nível CEFR da skill:
    - A1: 0.2 (muito fácil)
    - A2: 0.4 (fácil)
    - B1: 0.6 (intermediário)
    - B2: 0.75 (difícil)
    - C1: 0.85 (muito difícil)
    - C2: 0.95 (extremamente difícil)
    
    Args:
        skill_id: ID da habilidade
        
    Returns:
        Dificuldade IRT (0.0 a 1.0)
    """
    # Encontrar nível CEFR da skill
    for level, skills in SKILL_CEFR_MAP.items():
        if skill_id in skills:
            return CEFR_DIFFICULTY_MAP.get(level, 0.5)
    
    # Se não encontrar, inferir baseado no skill_id
    if "basic" in skill_id.lower() or "beginner" in skill_id.lower():
        return CEFR_DIFFICULTY_MAP["A1"]
    elif "intermediate" in skill_id.lower():
        return CEFR_DIFFICULTY_MAP["B1"]
    elif "advanced" in skill_id.lower() or "proficient" in skill_id.lower():
        return CEFR_DIFFICULTY_MAP["B2"]
    elif "native" in skill_id.lower() or "rare" in skill_id.lower():
        return CEFR_DIFFICULTY_MAP["C2"]
    
    # Dificuldade média por padrão
    return 0.5

