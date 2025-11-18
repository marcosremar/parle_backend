#!/usr/bin/env python3
"""
Centralized Conversation Prompts Configuration
Defines all conversation types and optimized prompts based on evaluation metrics
"""

import logging
from enum import Enum
from typing import Dict, Optional, Any
from dataclasses import dataclass
from src.core.exceptions import UltravoxError, wrap_exception

logger = logging.getLogger(__name__)


class ConversationType(Enum):
    """Tipos de conversa disponíveis"""
    INFORMAL = "informal"
    FORMAL = "formal"
    TECHNICAL = "technical"
    CASUAL = "casual"
    SUPPORTIVE = "supportive"


@dataclass
class PromptConfiguration:
    """Configuração de prompt com métricas otimizadas"""
    base_prompt: str
    max_tokens: int
    temperature: float
    engagement_level: str
    naturalness_focus: str
    coherence_rules: str
    relevance_guidelines: str


class ConversationPrompts:
    """
    Gerenciador centralizado de prompts de conversa
    Otimizado baseado em métricas de avaliação:
    - Naturalidade: 0.70 → target 0.85+
    - Engajamento: 0.65 → target 0.80+
    - Coerência: 0.85 → manter
    - Relevância: 0.90 → manter
    """

    # Prompt principal otimizado para naturalidade, engajamento e métricas específicas
    OPTIMIZED_INFORMAL_PROMPT = """You are a friendly, engaging conversational AI assistant. Your responses should feel natural, warm, and genuinely interested in helping.

CORE PERSONALITY:
- Be conversational and authentic, like talking to a good friend
- Show genuine curiosity and interest in the user's questions
- Use natural speech patterns with appropriate contractions (I'll, you're, can't)
- Express enthusiasm when appropriate with phrases like "That's interesting!" or "Great question!"
- Be encouraging and supportive while staying helpful

ENGAGEMENT TECHNIQUES:
- Ask follow-up questions when relevant to keep the conversation flowing
- Use the user's name if provided to personalize responses
- Acknowledge their feelings or situation when appropriate
- Share brief, relevant insights that add value to the conversation
- Use transitional phrases to create smooth conversation flow

NATURALNESS GUIDELINES:
- Speak as humans naturally do - avoid overly formal or robotic language
- Use varied sentence structures and lengths for natural rhythm
- Include appropriate pauses in longer responses with punctuation
- Express uncertainty naturally when you don't know something
- Use contextual responses that build on previous exchanges

USER STIMULUS GUIDELINES (CRITICAL FOR INFORMAL):
- ALWAYS try to stimulate the user to continue the conversation
- End 70% of responses with engaging questions or invitations to share more
- Use phrases like "What do you think about...?", "Tell me more about...", "How did that make you feel?"
- Show genuine interest in the user's experiences and thoughts
- Create conversational hooks that invite further discussion

DIRECTNESS BALANCE (INFORMAL CONVERSATIONS):
- Be direct and clear in your main message (answer the question directly first)
- Then add conversational elements and engagement
- Avoid being overly verbose - get to the point while staying warm
- Structure: Direct answer + brief elaboration + engagement stimulus

RESPONSE STRUCTURE:
- Keep responses conversational but focused (aim for 40-80 tokens)
- Start with direct acknowledgment and answer
- Provide helpful information naturally woven into conversation
- End with user engagement stimulus (question, invitation, or curiosity hook)

Remember: Balance directness with engagement - answer directly, then invite more conversation."""

    TECHNICAL_PROMPT = """You are a knowledgeable technical assistant who explains complex topics in an approachable way.

TECHNICAL APPROACH:
- Break down complex concepts into digestible parts
- Use analogies and examples to clarify difficult ideas
- Maintain technical accuracy while staying accessible
- Ask clarifying questions to understand the user's technical level

ENGAGEMENT IN TECHNICAL CONTEXTS:
- Show enthusiasm for technical topics
- Acknowledge when something is challenging and offer support
- Suggest next steps or related topics that might interest the user
- Use encouraging language when helping with learning

COMMUNICATION STYLE:
- Clear, structured explanations without being overly formal
- Natural transitions between technical concepts
- Appropriate use of technical terminology with explanations when needed
- Conversational tone even when discussing complex topics"""

    SUPPORTIVE_PROMPT = """You are an empathetic, supportive assistant focused on understanding and helping users through challenges.

SUPPORTIVE APPROACH:
- Listen actively and acknowledge emotions or concerns
- Provide gentle guidance and practical suggestions
- Maintain an optimistic but realistic perspective
- Respect boundaries while offering appropriate support

ENGAGEMENT TECHNIQUES:
- Use empathetic language that validates experiences
- Ask thoughtful questions to better understand needs
- Offer multiple options or perspectives when helpful
- Check in on how suggestions feel to the user

TONE AND STYLE:
- Warm, caring, and non-judgmental
- Patient and understanding responses
- Encouraging without being dismissive of challenges
- Natural, human-like expressions of care and support"""

    # Configurações por tipo de conversa
    CONVERSATION_CONFIGS = {
        ConversationType.INFORMAL: PromptConfiguration(
            base_prompt=OPTIMIZED_INFORMAL_PROMPT,
            max_tokens=120,  # Aumentado para evitar cortes
            temperature=0.8,
            engagement_level="high",
            naturalness_focus="conversational_flow",
            coherence_rules="maintain_context_continuity",
            relevance_guidelines="address_user_intent_directly"
        ),

        ConversationType.TECHNICAL: PromptConfiguration(
            base_prompt=TECHNICAL_PROMPT,
            max_tokens=150,  # Maior para explicações técnicas
            temperature=0.6,
            engagement_level="moderate_high",
            naturalness_focus="accessible_explanations",
            coherence_rules="logical_step_by_step",
            relevance_guidelines="technical_accuracy_priority"
        ),

        ConversationType.SUPPORTIVE: PromptConfiguration(
            base_prompt=SUPPORTIVE_PROMPT,
            max_tokens=100,
            temperature=0.7,
            engagement_level="empathetic",
            naturalness_focus="emotional_connection",
            coherence_rules="validate_and_guide",
            relevance_guidelines="address_emotional_needs"
        ),

        ConversationType.CASUAL: PromptConfiguration(
            base_prompt="""You are a relaxed, friendly conversational partner. Keep things light, fun, and engaging while being genuinely helpful.

CASUAL STYLE:
- Use relaxed, everyday language
- Feel free to be a bit playful when appropriate
- Show personality while staying respectful
- Keep the mood positive and upbeat

ENGAGEMENT:
- Use humor when it fits naturally
- Ask interesting questions that spark conversation
- Share relatable insights or observations
- Make the interaction enjoyable and memorable""",
            max_tokens=90,
            temperature=0.9,
            engagement_level="playful",
            naturalness_focus="relaxed_authentic",
            coherence_rules="maintain_positive_flow",
            relevance_guidelines="helpful_but_fun"
        ),

        ConversationType.FORMAL: PromptConfiguration(
            base_prompt="""You are a professional, articulate assistant providing clear and comprehensive responses.

FORMAL APPROACH:
- Use professional language while remaining approachable
- Provide thorough, well-structured responses
- Maintain courtesy and respect in all interactions
- Focus on accuracy and completeness

ENGAGEMENT:
- Show genuine interest in providing quality assistance
- Ask relevant questions to ensure understanding
- Offer additional resources or information when helpful
- Maintain warmth within professional boundaries""",
            max_tokens=140,
            temperature=0.5,
            engagement_level="professional",
            naturalness_focus="polished_but_warm",
            coherence_rules="structured_comprehensive",
            relevance_guidelines="thorough_accurate"
        )
    }

    # Instruções de idioma otimizadas
    LANGUAGE_INSTRUCTIONS = {
        "English": """
LANGUAGE: Respond in natural, fluent English
- Use contractions and natural speech patterns
- Vary sentence structure for engaging flow
- Include conversational markers when appropriate""",

        "Portuguese": """
IDIOMA: Responda em português brasileiro natural e fluente
- Use linguagem coloquial apropriada e contrações naturais
- Varie a estrutura das frases para criar fluidez
- Inclua marcadores conversacionais quando apropriado""",

        "Spanish": """
IDIOMA: Responde en español natural y fluido
- Usa lenguaje coloquial apropiado y contracciones naturales
- Varía la estructura de las oraciones para crear fluidez
- Incluye marcadores conversacionales cuando sea apropiado""",

        "French": """
LANGUE: Répondez en français naturel et fluide
- Utilisez un langage familier approprié et des contractions naturelles
- Variez la structure des phrases pour créer de la fluidité
- Incluez des marqueurs conversationnels quand c'est approprié""",

        "Japanese": """
言語: 自然で流暢な日本語で応答してください
- 適切なカジュアル言語と自然な縮約を使用
- 文の構造を変化させて流暢さを作る
- 適切な場合は会話マーカーを含める""",

        "Chinese": """
语言：用自然流利的中文回应
- 使用适当的口语和自然缩略
- 变化句子结构以创造流畅感
- 在适当时包含对话标记""",

        "Hindi": """
भाषा: प्राकृतिक और धाराप्रवाह हिंदी में जवाब दें
- उपयुक्त बोलचाल की भाषा और प्राकृतिक संकुचन का उपयोग करें
- प्रवाहता बनाने के लिए वाक्य संरचना को बदलें
- उपयुक्त होने पर बातचीत के मार्करों को शामिल करें""",

        "Italian": """
LINGUA: Rispondi in italiano naturale e fluente
- Usa linguaggio colloquiale appropriato e contrazioni naturali
- Varia la struttura delle frasi per creare fluidità
- Includi marcatori conversazionali quando appropriato"""
    }

    @classmethod
    def get_conversation_prompt(cls,
                              conversation_type: ConversationType = ConversationType.INFORMAL,
                              language: str = "English",
                              custom_prompt: str = "") -> str:
        """
        Obter prompt completo otimizado para um tipo de conversa

        Args:
            conversation_type: Tipo de conversa desejado
            language: Idioma para resposta
            custom_prompt: Prompt personalizado adicional

        Returns:
            Prompt completo formatado e otimizado
        """
        config = cls.CONVERSATION_CONFIGS.get(conversation_type,
                                            cls.CONVERSATION_CONFIGS[ConversationType.INFORMAL])

        # Instruções de idioma
        language_instruction = cls.LANGUAGE_INSTRUCTIONS.get(language,
                                                           cls.LANGUAGE_INSTRUCTIONS["English"])

        # Montar prompt completo
        full_prompt = f"""{config.base_prompt}

{language_instruction}

RESPONSE OPTIMIZATION:
- Maximum tokens: {config.max_tokens} (to prevent truncation)
- Engagement level: {config.engagement_level}
- Naturalness focus: {config.naturalness_focus}
- Coherence rule: {config.coherence_rules}
- Relevance guideline: {config.relevance_guidelines}"""

        # Adicionar prompt customizado se fornecido
        if custom_prompt.strip():
            full_prompt += f"\n\nCUSTOM INSTRUCTIONS: {custom_prompt.strip()}"

        return full_prompt

    @classmethod
    def get_language_from_voice_id(cls, voice_id: str) -> str:
        """
        Detectar idioma baseado no ID da voz

        Args:
            voice_id: ID da voz selecionada

        Returns:
            Nome do idioma
        """
        if not voice_id:
            return "English"

        # Mapear prefixo para idioma
        language_map = {
            'af_': 'English',      # American Female
            'am_': 'English',      # American Male
            'bf_': 'English',      # British Female
            'bm_': 'English',      # British Male
            'pf_': 'Portuguese',   # Portuguese Female
            'pm_': 'Portuguese',   # Portuguese Male
            'ef_': 'Spanish',      # Spanish Female
            'em_': 'Spanish',      # Spanish Male
            'ff_': 'French',       # French Female
            'fm_': 'French',       # French Male
            'jf_': 'Japanese',     # Japanese Female
            'jm_': 'Japanese',     # Japanese Male
            'zf_': 'Chinese',      # Chinese Female
            'zm_': 'Chinese',      # Chinese Male
            'hf_': 'Hindi',        # Hindi Female
            'hm_': 'Hindi',        # Hindi Male
            'if_': 'Italian',      # Italian Female
            'im_': 'Italian',      # Italian Male
        }

        # Extrair prefixo (primeiros 3 caracteres)
        prefix = voice_id[:3] if len(voice_id) >= 3 else ''
        return language_map.get(prefix, 'English')

    @classmethod
    def get_token_limit(cls, conversation_type: ConversationType = ConversationType.INFORMAL) -> int:
        """
        Obter limite de tokens para tipo de conversa

        Args:
            conversation_type: Tipo de conversa

        Returns:
            Limite de tokens recomendado
        """
        config = cls.CONVERSATION_CONFIGS.get(conversation_type,
                                            cls.CONVERSATION_CONFIGS[ConversationType.INFORMAL])
        return config.max_tokens

    @classmethod
    def get_temperature(cls, conversation_type: ConversationType = ConversationType.INFORMAL) -> float:
        """
        Obter temperatura recomendada para tipo de conversa

        Args:
            conversation_type: Tipo de conversa

        Returns:
            Temperatura recomendada
        """
        config = cls.CONVERSATION_CONFIGS.get(conversation_type,
                                            cls.CONVERSATION_CONFIGS[ConversationType.INFORMAL])
        return config.temperature

    @classmethod
    def validate_response_length(cls, response: str, max_tokens: int) -> bool:
        """
        Validar se resposta não foi truncada

        Args:
            response: Resposta gerada
            max_tokens: Limite máximo de tokens

        Returns:
            True se resposta parece completa, False se possivelmente truncada
        """
        # Verificações simples para detectar truncamento
        if not response.strip():
            return False

        # Verificar se termina abruptamente (sem pontuação)
        last_char = response.strip()[-1]
        if last_char.isalnum():  # Termina com letra ou número
            logger.warning(f"⚠️ Resposta possivelmente truncada: '{response[-20:]}'")
            return False

        # Verificar se está próximo do limite de tokens (estimativa rough)
        estimated_tokens = len(response.split()) * 1.3  # Estimativa aproximada
        if estimated_tokens > max_tokens * 0.95:
            logger.warning(f"⚠️ Resposta próxima do limite de tokens: {estimated_tokens:.0f}/{max_tokens}")
            return False

        return True

    @classmethod
    def advanced_response_validation(cls, response: str, max_tokens: int, conversation_type: ConversationType = ConversationType.INFORMAL, voice_id: str = None) -> dict:
        """
        Validação avançada de resposta usando análise por LLM
        Mais flexível e funciona melhor com diferentes idiomas

        Args:
            response: Resposta gerada
            max_tokens: Limite máximo de tokens
            conversation_type: Tipo de conversa para validação específica
            voice_id: ID da voz para detectar idioma esperado

        Returns:
            Dicionário detalhado com resultados da validação
        """
        if not response or not response.strip():
            return {
                'valid': False,
                'complete': False,
                'warnings': ['Resposta vazia'],
                'truncation_risk': 'high',
                'sentence_complete': False,
                'user_stimulus_present': False,
                'conversational_directness_score': 0.0,
                'language_consistency_score': 0.0,
                'language_is_consistent': False,
                'primary_language': 'indefinido',
                'engagement_indicators': [],
                'estimated_tokens': 0,
                'max_tokens': max_tokens,
                'length_ok': False,
                'llm_analysis': 'N/A - resposta vazia',
                'analysis_method': 'empty_response'
            }

        response_clean = response.strip()
        warnings = []

        # 1. Detectar truncamento de frases (mantém análise básica)
        sentence_complete = cls._check_sentence_completeness(response_clean)
        if not sentence_complete:
            warnings.append('Frase possivelmente cortada')

        # 2. Verificar proximidade do limite de tokens
        estimated_tokens = len(response_clean.split()) * 1.3
        truncation_risk = cls._assess_truncation_risk(estimated_tokens, max_tokens)
        if truncation_risk == 'high':
            warnings.append('Alto risco de truncamento')
        elif truncation_risk == 'medium':
            warnings.append('Médio risco de truncamento')

        # 3. Detectar idioma esperado baseado no voice_id
        expected_language = "português"  # Padrão
        if voice_id:
            detected_lang = cls.get_language_from_voice_id(voice_id)
            # Mapear idiomas para versões do prompt LLM (fallback para português)
            language_mapping = {
                "Portuguese": "português",
                "English": "english",
                "Spanish": "español",
                "French": "français"
            }
            expected_language = language_mapping.get(detected_lang, "português")
            logger.info(f"🌍 Voice ID: {voice_id} → Idioma detectado: {detected_lang} → Prompt: {expected_language}")

        # 4. Para conversas informais: usar análise LLM para tudo
        user_stimulus_present = False
        conversational_directness_score = 0.0
        language_consistency_score = 1.0
        language_is_consistent = True
        primary_language = expected_language
        llm_analysis = "N/A"

        if conversation_type == ConversationType.INFORMAL:
            # Usar análise LLM em vez de regras fixas, com idioma detectado
            try:
                llm_analysis_result = cls._analyze_response_with_llm(response_clean, conversation_type, expected_language)

                user_stimulus_present = llm_analysis_result.get('user_stimulus_present', False)
                conversational_directness_score = llm_analysis_result.get('conversational_directness_score', 0.0)
                language_consistency_score = llm_analysis_result.get('language_consistency_score', 1.0)
                language_is_consistent = llm_analysis_result.get('language_is_consistent', True)
                primary_language = llm_analysis_result.get('primary_language', 'português')
                llm_analysis = llm_analysis_result.get('analysis_text', "Análise LLM não disponível")

                # Avisos baseados na análise LLM - critérios mais realistas
                # Só exigir estímulo para conversas informais, não técnicas
                if conversation_type == ConversationType.INFORMAL and not user_stimulus_present:
                    warnings.append('Falta estímulo para continuar conversa')

                # Ser menos rigoroso com diretividade (0.4 em vez de 0.6)
                if conversational_directness_score < 0.4:
                    warnings.append('Conversa não é direta e estimulante o suficiente')

                # Só avisar sobre consistência se for realmente baixa (0.5 em vez de 0.7)
                if language_consistency_score < 0.5:
                    warnings.append('Problemas na consistência do idioma')

            except Exception as e:
                # Fallback para análise básica se LLM falhar
                user_stimulus_present, _ = cls._check_user_stimulus(response_clean)
                conversational_directness_score = 0.5  # Score neutro se LLM falhar
                language_consistency_score = 0.8  # Assumir OK no fallback
                language_is_consistent = True
                primary_language = "português"
                llm_analysis = f"Fallback usado - erro LLM: {str(e)}"

                if not user_stimulus_present:
                    warnings.append('Falta estímulo para continuar conversa')

        # 4. Validação geral de qualidade
        length_ok = 10 <= len(response_clean) <= max_tokens * 4

        return {
            'valid': len(warnings) == 0 and sentence_complete and length_ok,
            'complete': sentence_complete,
            'warnings': warnings,
            'truncation_risk': truncation_risk,
            'sentence_complete': sentence_complete,
            'user_stimulus_present': user_stimulus_present,
            'conversational_directness_score': conversational_directness_score,
            'language_consistency_score': language_consistency_score,
            'language_is_consistent': language_is_consistent,
            'primary_language': primary_language,
            'engagement_indicators': [],  # Agora vem da análise LLM
            'estimated_tokens': estimated_tokens,
            'max_tokens': max_tokens,
            'length_ok': length_ok,
            'llm_analysis': llm_analysis,
            'analysis_method': 'llm_based'
        }

    @classmethod
    def _check_sentence_completeness(cls, response: str) -> bool:
        """Verificar se a frase parece completa"""
        if not response:
            return False

        response = response.strip()

        # Verificar terminações válidas
        valid_endings = ['.', '!', '?', ':', ';', '"', "'", ')', ']', '}']
        last_char = response[-1]

        # Se termina com pontuação válida, provavelmente está completa
        if last_char in valid_endings:
            return True

        # Se termina com letra/número, verificar se parece truncado
        if last_char.isalnum():
            words = response.split()
            if not words:
                return False

            last_word = words[-1].lower()

            # Verificar padrões de truncamento comum
            truncation_patterns = [
                'and', 'or', 'but', 'the', 'a', 'an', 'to', 'of', 'in', 'for', 'with', 'on', 'at',
                'com', 'para', 'que', 'de', 'da', 'do', 'na', 'no', 'em', 'por', 'se', 'é', 'mas', 'ou'
            ]

            # Palavras que indicam truncamento
            if last_word in truncation_patterns:
                return False

            # Verbos auxiliares que indicam truncamento
            auxiliary_verbs = ['está', 'sendo', 'tendo', 'fazendo', 'going', 'having', 'doing', 'being']
            if last_word in auxiliary_verbs:
                return False

            # Respostas muito curtas mas válidas (Ok, Sim, Não, etc.)
            short_complete_words = ['ok', 'sim', 'não', 'yes', 'no', 'maybe', 'talvez', 'claro', 'sure']
            if len(words) <= 2 and last_word in short_complete_words:
                return True

            # Se a última palavra é muito curta (provavelmente truncada)
            if len(last_word) <= 2 and last_word not in short_complete_words:
                return False

        return True

    @classmethod
    def _assess_truncation_risk(cls, estimated_tokens: float, max_tokens: int) -> str:
        """Avaliar risco de truncamento baseado em tokens"""
        usage_ratio = estimated_tokens / max_tokens

        if usage_ratio >= 0.95:
            return 'high'
        elif usage_ratio >= 0.85:
            return 'medium'
        else:
            return 'low'

    @classmethod
    def _check_user_stimulus(cls, response: str) -> tuple:
        """Verificar se há estímulo para o usuário continuar falando"""
        response_lower = response.lower()

        # Indicadores de engajamento
        question_indicators = ['?', 'what', 'how', 'why', 'when', 'where', 'which', 'who']
        invitation_phrases = [
            'tell me', 'share', 'what do you think', 'how do you feel',
            'what about', 'have you', 'would you', 'could you',
            'me conte', 'compartilhe', 'o que você acha', 'como você se sente',
            'que tal', 'você já', 'você gostaria', 'você poderia'
        ]

        engagement_indicators = []

        # Verificar perguntas
        if '?' in response:
            engagement_indicators.append('question_mark')

        # Verificar palavras de pergunta
        for indicator in question_indicators:
            if indicator in response_lower:
                engagement_indicators.append(f'question_word_{indicator}')

        # Verificar frases de convite
        for phrase in invitation_phrases:
            if phrase in response_lower:
                engagement_indicators.append(f'invitation_{phrase.replace(" ", "_")}')

        # Considerar estímulo presente se há pelo menos um indicador
        user_stimulus_present = len(engagement_indicators) > 0

        return user_stimulus_present, engagement_indicators

    @classmethod
    def _evaluate_conversational_directness(cls, response: str) -> float:
        """
        Avaliar se a conversa é direta e estimulante ao mesmo tempo
        Ideal para conversas informais: ser claro/direto MAS também engajador

        Score: 0.0 = conversa indireta/confusa, 1.0 = conversa direta e estimulante
        """
        if not response:
            return 0.0

        response_lower = response.lower()
        words = response.split()

        # ANÁLISE PRIMÁRIA: DEVE TER ESTÍMULO PARA SER ESTIMULANTE
        stimulus_present, _ = cls._check_user_stimulus(response)

        # Se não tem estímulo, não pode ser considerada estimulante
        if not stimulus_present:
            # Ainda pode ter score médio/baixo se for direta, mas não pode ser alta
            max_score_without_stimulus = 0.6
        else:
            max_score_without_stimulus = 1.0

        # ANÁLISE DE QUALIFICADORES PARA DEFINIR LIMITE MÁXIMO
        strong_qualifiers = [
            'maybe', 'perhaps', 'i think', 'possibly', 'probably', 'sort of', 'kind of',
            'talvez', 'quem sabe', 'acho que', 'possivelmente', 'provavelmente', 'meio que'
        ]
        qualifier_count = sum(1 for q in strong_qualifiers if q in response_lower)

        # Ajustar limite máximo baseado em qualificadores
        if qualifier_count >= 2:
            max_score_without_stimulus = min(max_score_without_stimulus, 0.7)  # Com 2+ qualificadores, max 0.7
        elif qualifier_count == 1:
            max_score_without_stimulus = min(max_score_without_stimulus, 0.8)  # Com 1 qualificador, max 0.8

        # Começar com score base
        score = 0.5

        # Penalização mais severa por qualificadores (usando a contagem já feita)
        if qualifier_count == 1:
            score -= 0.2   # Um qualificador já reduz significativamente
        elif qualifier_count == 2:
            score -= 0.4   # Dois qualificadores reduzem muito
        elif qualifier_count >= 3:
            score -= 0.6   # Três ou mais qualificadores são inaceitáveis

        # ANÁLISE DE COMEÇO HESITANTE (muito severa)
        hesitant_starters = ['well', 'so', 'actually', 'hmm', 'uh', 'bem', 'então', 'na verdade']
        if words and words[0].lower() in hesitant_starters:
            score -= 0.5  # Começo hesitante é muito prejudicial

        # ANÁLISE DE DESVIOS DO ASSUNTO
        diverting_phrases = [
            'by the way', 'speaking of', 'a propósito', 'falando nisso', 'aliás'
        ]
        diversion_count = sum(1 for phrase in diverting_phrases if phrase in response_lower)
        score -= diversion_count * 0.3

        # ANÁLISE DE COMPRIMENTO
        word_count = len(words)
        if word_count > 50:
            score -= 0.25  # Respostas muito longas são menos diretas
        elif word_count < 3:
            score -= 0.2   # Respostas muito curtas podem não ser estimulantes

        # BONIFICAÇÕES POR BOA CONVERSAÇÃO DIRETA E ESTIMULANTE

        # 1. Começou direto ao ponto (+0.3)
        direct_starters = [
            'sim', 'não', 'claro', 'certo', 'perfeito', 'entendi', 'legal', 'interessante',
            'yes', 'no', 'sure', 'right', 'perfect', 'got it', 'cool', 'interesting'
        ]
        if words and words[0].lower() in direct_starters:
            score += 0.3

        # 2. Tem estímulo ao usuário (+0.4 - crítico)
        if stimulus_present:
            score += 0.4

        # 3. Tamanho ideal para conversa estimulante (+0.2)
        if 5 <= word_count <= 25:  # Tamanho perfeito para conversa informal
            score += 0.2
        elif 26 <= word_count <= 35:  # Ainda ok, mas menos ideal
            score += 0.1

        # 4. Estrutura conversacional ideal: afirmação + pergunta (+0.25)
        has_question = '?' in response
        has_statement = any(end in response for end in ['.', '!'])
        if has_question and has_statement:
            score += 0.25

        # 5. Bonus por ser direto sem qualificadores (+0.15)
        if qualifier_count == 0 and words and words[0].lower() not in hesitant_starters:
            score += 0.15

        # 6. Bonus por tom positivo/engajador (+0.1)
        positive_words = ['legal', 'interessante', 'ótimo', 'perfeito', 'cool', 'great', 'awesome', 'nice']
        if any(word in response_lower for word in positive_words):
            score += 0.1

        # APLICAR LIMITE BASEADO EM ESTÍMULO
        final_score = min(score, max_score_without_stimulus)

        # Garantir que score fica entre 0.0 e 1.0
        return max(0.0, min(1.0, final_score))

    @classmethod
    def _get_analysis_prompt_for_language(cls, response: str, target_language: str = "português") -> str:
        """
        Obter prompt de análise específico para o idioma
        Estrutura preparada para múltiplos idiomas
        """

        # Prompts específicos por idioma (começando com português)
        language_prompts = {
            "português": f"""
Você é o Llama 3.3, um modelo avançado de análise de qualidade conversacional.

Analise esta resposta de conversa informal em PORTUGUÊS BRASILEIRO:

RESPOSTA: "{response}"

Use sua capacidade de compreensão linguística avançada para avaliar (escala 0.0 a 1.0):

1. ESTÍMULO AO USUÁRIO: A resposta encoraja o usuário a continuar a conversa?
   - Analise o tom, intenção e abertura para diálogo
   - NOTA: Nem toda resposta precisa de pergunta direta - confirmações e respostas técnicas podem ser válidas
   - Use sua compreensão natural da linguagem

2. CONVERSAÇÃO DIRETA E ESTIMULANTE: A resposta é clara E envolvente?
   - Avalie se é objetiva sem perder interesse
   - Considere naturalidade e fluidez
   - SEJA EQUILIBRADO: respostas claras e diretas são boas mesmo sem ser super estimulantes

3. CONSISTÊNCIA DE IDIOMA: A resposta mantém consistência em português brasileiro?
   - Detecte qualquer mistura com outras línguas
   - Palavras emprestadas ocasionais (OK, legal, show) são aceitáveis
   - Use sua capacidade de identificar padrões linguísticos

IMPORTANTE: NÃO use listas de palavras pré-definidas. Use sua compreensão linguística natural.

RESPONDA EXATAMENTE neste formato:
STIMULUS_SCORE: [0.0-1.0]
DIRECTNESS_SCORE: [0.0-1.0]
LANGUAGE_CONSISTENCY_SCORE: [0.0-1.0]
PRIMARY_LANGUAGE: português
ANALYSIS: [Análise detalhada de 2-3 frases sobre cada aspecto]
""",

            "english": f"""
You are Llama 3.3, an advanced conversational quality analysis model.

Analyze this informal conversation response in ENGLISH:

RESPONSE: "{response}"

Use your advanced linguistic understanding to evaluate (scale 0.0 to 1.0):

1. USER STIMULUS: Does the response encourage the user to continue the conversation?
   - Analyze tone, intent, and openness for dialogue
   - NOTE: Not every response needs a direct question - confirmations and technical answers can be valid
   - Use your natural language understanding

2. DIRECT AND ENGAGING CONVERSATION: Is the response clear AND engaging?
   - Evaluate if it's objective without losing interest
   - Consider naturalness and flow
   - BE BALANCED: clear and direct responses are good even without being super stimulating

3. LANGUAGE CONSISTENCY: Does the response maintain consistency in English?
   - Detect any mixing with other languages
   - Occasional borrowed words (okay, cool) are acceptable
   - Use your capability to identify linguistic patterns

IMPORTANT: DO NOT use pre-defined word lists. Use your natural linguistic comprehension.

RESPOND EXACTLY in this format:
STIMULUS_SCORE: [0.0-1.0]
DIRECTNESS_SCORE: [0.0-1.0]
LANGUAGE_CONSISTENCY_SCORE: [0.0-1.0]
PRIMARY_LANGUAGE: english
ANALYSIS: [Detailed analysis of 2-3 sentences about each aspect]
""",

            "español": f"""
Eres Llama 3.3, un modelo avanzado de análisis de calidad conversacional.

Analiza esta respuesta de conversación informal en ESPAÑOL:

RESPUESTA: "{response}"

Usa tu comprensión lingüística avanzada para evaluar (escala 0.0 a 1.0):

1. ESTÍMULO AL USUARIO: ¿La respuesta anima al usuario a continuar la conversación?
   - Analiza el tono, intención y apertura para el diálogo
   - Usa tu comprensión natural del lenguaje

2. CONVERSACIÓN DIRECTA Y ESTIMULANTE: ¿La respuesta es clara Y atractiva?
   - Evalúa si es objetiva sin perder interés
   - Considera la naturalidad y fluidez

3. CONSISTENCIA DE IDIOMA: ¿La respuesta mantiene consistencia en español?
   - Detecta cualquier mezcla con otros idiomas
   - Usa tu capacidad para identificar patrones lingüísticos

IMPORTANTE: NO uses listas de palabras predefinidas. Usa tu comprensión lingüística natural.

RESPONDE EXACTAMENTE en este formato:
STIMULUS_SCORE: [0.0-1.0]
DIRECTNESS_SCORE: [0.0-1.0]
LANGUAGE_CONSISTENCY_SCORE: [0.0-1.0]
PRIMARY_LANGUAGE: español
ANALYSIS: [Análisis detallado de 2-3 frases sobre cada aspecto]
""",

            "français": f"""
Tu es Llama 3.3, un modèle avancé d'analyse de qualité conversationnelle.

Analyse cette réponse de conversation informelle en FRANÇAIS:

RÉPONSE: "{response}"

Utilise ta compréhension linguistique avancée pour évaluer (échelle 0.0 à 1.0):

1. STIMULATION DE L'UTILISATEUR: La réponse encourage-t-elle l'utilisateur à continuer la conversation?
   - Analyse le ton, l'intention et l'ouverture au dialogue
   - Utilise ta compréhension naturelle du langage

2. CONVERSATION DIRECTE ET STIMULANTE: La réponse est-elle claire ET engageante?
   - Évalue si elle est objective sans perdre l'intérêt
   - Considère la naturalité et la fluidité

3. COHÉRENCE LINGUISTIQUE: La réponse maintient-elle la cohérence en français?
   - Détecte tout mélange avec d'autres langues
   - Utilise ta capacité à identifier les patterns linguistiques

IMPORTANT: N'utilise PAS de listes de mots prédéfinies. Utilise ta compréhension linguistique naturelle.

RÉPONDS EXACTEMENT dans ce format:
STIMULUS_SCORE: [0.0-1.0]
DIRECTNESS_SCORE: [0.0-1.0]
LANGUAGE_CONSISTENCY_SCORE: [0.0-1.0]
PRIMARY_LANGUAGE: français
ANALYSIS: [Analyse détaillée de 2-3 phrases sur chaque aspect]
"""
        }

        # Usar prompt específico ou fallback para português
        return language_prompts.get(target_language.lower(), language_prompts["português"])

    @classmethod
    def _analyze_response_with_llm(cls, response: str, conversation_type: ConversationType, target_language: str = "português") -> dict:
        """
        Analisar resposta usando LLM com prompts específicos por idioma

        Args:
            response: Resposta para analisar
            conversation_type: Tipo de conversa
            target_language: Idioma esperado da resposta

        Returns:
            Dicionário com resultados da análise LLM
        """
        try:
            # Importar o evaluador Groq
            from src.services.llm.evaluators.groq_llama import GroqLlamaEvaluator
            import asyncio

            # Obter prompt específico para o idioma
            analysis_prompt = cls._get_analysis_prompt_for_language(response, target_language)

            # Criar instância do evaluador com Llama 3.3
            evaluator = GroqLlamaEvaluator(model="llama-3.3-70b-versatile")

            # Inicializar o cliente
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Inicializar de forma síncrona
            initialized = loop.run_until_complete(evaluator.initialize())

            if not initialized:
                raise Exception("Falha ao inicializar Groq client")

            # Fazer a análise usando o LLM
            result = evaluator.client.chat.completions.create(
                messages=[{"role": "user", "content": analysis_prompt}],
                model=evaluator.model,
                max_tokens=300,
                temperature=0.1
            )

            # Parsear resultado do LLM
            analysis_text = result.choices[0].message.content if hasattr(result, 'choices') and result.choices else str(result)

            # Extrair scores da resposta do LLM
            stimulus_score = 0.5
            directness_score = 0.5
            language_consistency_score = 0.5
            primary_language = "desconhecido"

            lines = analysis_text.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('STIMULUS_SCORE:'):
                    try:
                        stimulus_score = float(line.split(':')[1].strip())
                    except Exception as e:
                        pass
                elif line.startswith('DIRECTNESS_SCORE:'):
                    try:
                        directness_score = float(line.split(':')[1].strip())
                    except Exception as e:
                        pass
                elif line.startswith('LANGUAGE_CONSISTENCY_SCORE:'):
                    try:
                        language_consistency_score = float(line.split(':')[1].strip())
                    except Exception as e:
                        pass
                elif line.startswith('PRIMARY_LANGUAGE:'):
                    try:
                        primary_language = line.split(':')[1].strip().lower()
                    except Exception as e:
                        pass

            # Converter para formato esperado
            return {
                'user_stimulus_present': stimulus_score > 0.4,
                'conversational_directness_score': directness_score,
                'language_consistency_score': language_consistency_score,
                'language_is_consistent': language_consistency_score > 0.5,
                'primary_language': primary_language,
                'analysis_text': analysis_text,
                'llm_stimulus_score': stimulus_score,
                'llm_directness_score': directness_score,
                'llm_language_score': language_consistency_score,
                'method': 'groq_llama_analysis'
            }

        except Exception as e:
            logger.warning(f"⚠️ Análise LLM falhou: {e}")

            # Fallback simples quando LLM não está disponível
            return {
                'user_stimulus_present': False,
                'conversational_directness_score': 0.5,
                'language_consistency_score': 0.8,
                'language_is_consistent': True,
                'primary_language': target_language,
                'analysis_text': f"Fallback: LLM indisponível ({str(e)})",
                'llm_stimulus_score': 0.5,
                'llm_directness_score': 0.5,
                'llm_language_score': 0.8,
                'method': 'fallback_simple'
            }



# Instância global para fácil acesso
conversation_prompts = ConversationPrompts()