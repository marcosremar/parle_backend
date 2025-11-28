"""
LLM Client for Diagnostic Analysis
Cliente para chamar LLM para análise de erros e complexidade
"""

import aiohttp
import os
import json
import asyncio
from typing import Dict, Any, Optional, List
from loguru import logger


class DiagnosticLLMClient:
    """Cliente para análise usando LLM"""
    
    # Constantes
    MAX_PROMPT_LENGTH = 8000  # Ajustar conforme modelo LLM
    MAX_RETRIES = 3
    RETRY_BASE_DELAY = 0.5  # segundos
    
    def __init__(self):
        self.base_url = os.getenv("LLM_SERVICE_URL", "http://localhost:8110")
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Métricas
        self.metrics = {
            "extract_skills_calls": 0,
            "extract_skills_success": 0,
            "extract_skills_failures": 0,
            "avg_skills_detected": 0.0,
            "analyze_grammar_calls": 0,
            "analyze_grammar_success": 0,
            "analyze_grammar_failures": 0
        }
    
    async def initialize(self, session: aiohttp.ClientSession):
        """Inicializar com sessão HTTP"""
        self.session = session
    
    def _extract_content_from_llm_response(self, result: Dict[str, Any]) -> str:
        """
        Extract content from LLM response (handles multiple formats)
        
        Handles:
        - OpenAI format: {"choices": [{"message": {"content": "..."}}]}
        - Groq format: {"text": "..."}
        - Other formats: tries common field names
        
        Args:
            result: Raw response from LLM service
            
        Returns:
            Content string (empty if not found)
        """
        # Format 1: OpenAI-style with "choices"
        if "choices" in result and result["choices"]:
            message = result["choices"][0].get("message", {})
            return message.get("content", "")
        
        # Format 2: Direct "text" field (Groq/other providers)
        if "text" in result:
            return result.get("text", "")
        
        # Fallback: try common field names
        for field in ["content", "response", "output", "message"]:
            if field in result:
                return str(result[field])
        
        return ""
    
    def _validate_and_normalize_confidence(self, confidence_raw: Any, skill_id: str = "unknown") -> float:
        """
        Valida e normaliza confidence score
        
        Args:
            confidence_raw: Valor raw de confidence (pode ser int, float, str, None)
            skill_id: ID da skill (para logging)
            
        Returns:
            Confidence normalizado entre 0.0 e 1.0
        """
        try:
            if confidence_raw is None:
                return 0.5
            elif isinstance(confidence_raw, str):
                confidence = float(confidence_raw)
            elif isinstance(confidence_raw, (int, float)):
                confidence = float(confidence_raw)
            else:
                logger.warning(f"Invalid confidence type for skill {skill_id}: {type(confidence_raw)}, using 0.5")
                return 0.5
            
            return max(0.0, min(1.0, confidence))
        except (ValueError, TypeError) as e:
            logger.warning(f"Invalid confidence for skill {skill_id}: {confidence_raw}, using 0.5. Error: {e}")
            return 0.5
    
    def _build_skills_prompt(
        self,
        user_text: str,
        valid_skills: List[str],
        ai_text: Optional[str] = None,
        prompt_type: str = "extract"
    ) -> str:
        """
        Build prompt with size validation
        
        Args:
            user_text: Texto do aluno
            valid_skills: Lista de skills válidas
            ai_text: Contexto do AI (opcional)
            prompt_type: "extract" ou "analyze"
            
        Returns:
            Prompt validado (skills podem ser truncadas se necessário)
        """
        # Construir lista de skills
        skills_list = "\n".join([f"- {skill_id}" for skill_id in valid_skills])
        
        # Template base do prompt
        if prompt_type == "extract":
            base_prompt = f"""Você é um especialista em identificação de habilidades linguísticas. Analise o seguinte texto e identifique TODAS as skills que o aluno está usando.

Texto do aluno: "{user_text}"
{"Resposta do professor (contexto): " + ai_text if ai_text else ""}

Skills válidas disponíveis (você DEVE usar APENAS estas):
{skills_list}

IMPORTANTE:
- Você DEVE identificar skills que o aluno está usando no texto (corretas ou não)
- NÃO analise erros (isso é feito por outro módulo)
- Para cada skill identificada, forneça:
  1. skill_id (deve estar na lista de skills válidas acima)
  2. confidence (0.0 a 1.0) - quão confiante você está que essa skill foi usada
  3. linguistic_features específicas dessa skill (tense, person, number, register, domain, etc.)

EXTRAÇÃO OBRIGATÓRIA DE FEATURES LINGUÍSTICAS:
Para cada skill identificada, extraia features linguísticas:
- Para verbos: tense (present/past/future), person (1st/2nd/3rd), number (singular/plural), mood (indicative/subjunctive/conditional)
- Para vocabulário: register (formal/informal/neutral), domain (family/travel/emotions/daily_life/general)
- Para artigos: article_type (definite/indefinite)
- Para preposições: preposition_type (location/basic)

Além disso, extraia features linguísticas gerais do texto inteiro (overall_linguistic_features).

Exemplos:
- Texto "Eu fui ao banco ontem" → skill: verb_conjugation_past com features {{"tense": "past", "person": "1st", "number": "singular"}}
- Texto "Eles compraram comida" → skill: verb_conjugation_past com features {{"tense": "past", "person": "3rd", "number": "plural"}}

Retorne SEMPRE em formato JSON válido:
{{
    "skills": [
        {{
            "skill_id": "verb_conjugation_past",
            "confidence": 0.85,
            "linguistic_features": {{
                "tense": "past",
                "person": "1st",
                "number": "singular",
                "mood": "indicative"
            }}
        }},
        {{
            "skill_id": "prepositions_basic",
            "confidence": 0.70,
            "linguistic_features": {{
                "preposition_type": "location"
            }}
        }}
    ],
    "overall_linguistic_features": {{
        "tense": "past",
        "register": "informal",
        "domain": "daily_life"
    }},
    "summary": "Texto usa principalmente verbos no passado e preposições de localização."
}}

Lembre-se:
- Use APENAS skill_ids da lista fornecida
- Retorne confidence entre 0.0 e 1.0
- SEMPRE retorne linguistic_features, mesmo que vazio (objeto vazio)
- NUNCA retorne null"""
        else:
            # analyze_grammar prompt (mantém o existente, mas com validação de tamanho)
            # Extrair strings com \n para variáveis (evita problema com f-strings)
            ai_context = f"Resposta do professor (contexto): {ai_text}" if ai_text else ""
            skills_context = ""
            if valid_skills:
                skills_context = f"\n\nConsidere as seguintes skill_ids válidas para tagging: {', '.join(valid_skills[:20])}..."
            
            base_prompt = f"""Você é um especialista em análise linguística de português. Analise o seguinte texto e identifique TODOS os erros gramaticais e as habilidades linguísticas utilizadas.

Texto do aluno: "{user_text}"
{ai_context}{skills_context}

IMPORTANTE: 
- Você DEVE identificar erros gramaticais mesmo que sutis
- Você DEVE extrair features linguísticas SEMPRE, mesmo quando não há erros
- Para o texto "Eu ir na praia ontem", o erro é: "ir" deveria ser "fui" (verb_conjugation_past)
- Para o texto "Ela foi ao banco", se estiver correto, não há erro, mas extraia as features linguísticas

Para cada erro encontrado, forneça:
1. Tipo de erro (grammar, vocabulary, syntax)
2. Categoria específica (ex: verb_conjugation_past, article_definite)
3. Texto original com erro
4. Texto corrigido
5. Explicação breve
6. Severidade (low, medium, high)
7. skill_id relacionado (ex: verb_conjugation_past)
8. linguistic_features do erro (tense, person, number, etc.)

Além dos erros, identifique as skill_ids que o aluno utilizou corretamente no texto.

EXTRAÇÃO OBRIGATÓRIA DE FEATURES LINGUÍSTICAS:
Você DEVE sempre extrair features linguísticas do texto, mesmo quando não há erros:
- Para verbos: tense (present/past/future), person (1st/2nd/3rd), number (singular/plural), mood (indicative/subjunctive/conditional)
- Para vocabulário: register (formal/informal/neutral), domain (family/travel/emotions/daily_life/general)
- Para artigos: article_type (definite/indefinite)
- Para preposições: preposition_type (location/basic)

Exemplos de features:
- "Eu fui" → {{"tense": "past", "person": "1st", "number": "singular"}}
- "Eles foram" → {{"tense": "past", "person": "3rd", "number": "plural"}}
- "Nós vamos" → {{"tense": "future", "person": "1st", "number": "plural"}}

Retorne SEMPRE em formato JSON válido:
{{
    "errors": [
        {{
            "error_type": "grammar",
            "category": "verb_conjugation_past",
            "original_text": "ir",
            "corrected_text": "fui",
            "explanation": "Verbo 'ir' no passado deve ser conjugado como 'fui' na primeira pessoa",
            "severity": "medium",
            "skill_id": "verb_conjugation_past",
            "linguistic_features": {{
                "tense": "past",
                "person": "1st",
                "number": "singular",
                "mood": "indicative"
            }}
        }}
    ],
    "correct_skills": ["skill_id_1", "skill_id_2"],
    "linguistic_features": {{
        "tense": "past",
        "person": "1st",
        "number": "singular",
        "register": "informal"
    }},
    "summary": "Análise geral do texto."
}}

Lembre-se: SEMPRE retorne linguistic_features, mesmo que vazio (objeto vazio). NUNCA retorne null."""
        
        # Validar tamanho do prompt
        if len(base_prompt) > self.MAX_PROMPT_LENGTH:
            logger.warning(f"Prompt too long ({len(base_prompt)} chars), truncating skills list")
            
            # Calcular espaço disponível para skills
            base_without_skills = base_prompt.replace(skills_list, "")
            available_space = self.MAX_PROMPT_LENGTH - len(base_without_skills) - 100  # 100 chars de margem
            
            # Priorizar skills mais relevantes
            priority_skills = [
                s for s in valid_skills 
                if any(x in s for x in ["verb_", "vocabulary_basic", "article_", "preposition_", "pronoun_"])
            ]
            remaining_skills = [s for s in valid_skills if s not in priority_skills]
            
            # Calcular quantas skills cabem
            avg_chars_per_skill = 25  # "- skill_id\n"
            max_skills = max(10, available_space // avg_chars_per_skill)  # Mínimo 10 skills
            
            # Combinar prioridade + restantes até limite
            selected_skills = priority_skills[:max_skills//2] + remaining_skills[:max_skills//2]
            skills_list = "\n".join([f"- {skill_id}" for skill_id in selected_skills])
            
            # Reconstruir prompt
            base_prompt = base_prompt.replace(
                "\n".join([f"- {skill_id}" for skill_id in valid_skills]),
                skills_list
            )
            
            logger.info(f"Reduced skills list from {len(valid_skills)} to {len(selected_skills)}")
        
        return base_prompt
    
    def _calculate_timeout(self, valid_skills_count: int) -> int:
        """Calcula timeout baseado no número de skills"""
        base_timeout = 15  # 15s base
        per_skill_timeout = 0.1  # 100ms por skill
        return int(base_timeout + (valid_skills_count * per_skill_timeout))
    
    async def analyze_grammar(
        self,
        user_text: str,
        ai_text: Optional[str] = None,
        valid_skills: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analisa erros gramaticais usando LLM e extrai features linguísticas
        
        Args:
            user_text: Texto do usuário
            ai_text: Resposta do AI (para contexto)
            valid_skills: Lista de skill_ids válidas para skill tagging (SINKT)
            
        Returns:
            Dicionário com análise de erros, features linguísticas e correct_skills
        """
        skill_list_instruction = ""
        if valid_skills:
            skill_list_instruction = f"\n\nConsidere as seguintes skill_ids válidas para tagging: {', '.join(valid_skills)}." \
                                     "\nPara cada skill_id relevante no texto do aluno (mesmo que correta), inclua-a na lista 'correct_skills'."
        
        prompt = f"""Você é um especialista em análise linguística de português. Analise o seguinte texto e identifique TODOS os erros gramaticais e as habilidades linguísticas utilizadas.

Texto do aluno: "{user_text}"
{"Resposta do professor (contexto): " + ai_text if ai_text else ""}
{skill_list_instruction}

IMPORTANTE: 
- Você DEVE identificar erros gramaticais mesmo que sutis
- Você DEVE extrair features linguísticas SEMPRE, mesmo quando não há erros
- Para o texto "Eu ir na praia ontem", o erro é: "ir" deveria ser "fui" (verb_conjugation_past)
- Para o texto "Ela foi ao banco", se estiver correto, não há erro, mas extraia as features linguísticas

Para cada erro encontrado, forneça:
1. Tipo de erro (grammar, vocabulary, syntax)
2. Categoria específica (ex: verb_conjugation_past, article_definite)
3. Texto original com erro
4. Texto corrigido
5. Explicação breve
6. Severidade (low, medium, high)
7. skill_id relacionado (ex: verb_conjugation_past)
8. linguistic_features do erro (tense, person, number, etc.)

Além dos erros, identifique as skill_ids que o aluno utilizou corretamente no texto.

EXTRAÇÃO OBRIGATÓRIA DE FEATURES LINGUÍSTICAS:
Você DEVE sempre extrair features linguísticas do texto, mesmo quando não há erros:
- Para verbos: tense (present/past/future), person (1st/2nd/3rd), number (singular/plural), mood (indicative/subjunctive/conditional)
- Para vocabulário: register (formal/informal/neutral), domain (family/travel/emotions/daily_life/general)
- Para artigos: article_type (definite/indefinite)
- Para preposições: preposition_type (location/basic)

Exemplos de features:
- "Eu fui" → {{"tense": "past", "person": "1st", "number": "singular"}}
- "Eles foram" → {{"tense": "past", "person": "3rd", "number": "plural"}}
- "Nós vamos" → {{"tense": "future", "person": "1st", "number": "plural"}}

Retorne SEMPRE em formato JSON válido:
{{
    "errors": [
        {{
            "error_type": "grammar",
            "category": "verb_conjugation_past",
            "original_text": "ir",
            "corrected_text": "fui",
            "explanation": "Verbo 'ir' no passado deve ser conjugado como 'fui' na primeira pessoa",
            "severity": "medium",
            "skill_id": "verb_conjugation_past",
            "linguistic_features": {{
                "tense": "past",
                "person": "1st",
                "number": "singular",
                "mood": "indicative"
            }}
        }}
    ],
    "correct_skills": ["skill_id_1", "skill_id_2"],
    "linguistic_features": {{
        "tense": "past",
        "person": "1st",
        "number": "singular",
        "register": "informal"
    }},
    "summary": "Análise geral do texto."
}}

Lembre-se: SEMPRE retorne linguistic_features, mesmo que vazio (objeto vazio). NUNCA retorne null."""
        
        self.metrics["analyze_grammar_calls"] += 1
        
        # Retry logic
        for attempt in range(self.MAX_RETRIES):
            try:
                # Chamar LLM service
                timeout = self._calculate_timeout(len(valid_skills) if valid_skills else 0)
                async with self.session.post(
                    f"{self.base_url}/chat",
                    json={
                        "messages": [
                            {"role": "system", "content": "Você é um especialista em análise linguística."},
                            {"role": "user", "content": prompt}
                        ],
                        "model": "groq/llama-3.1-8b-instant"
                    },
                    timeout=aiohttp.ClientTimeout(total=timeout)
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        
                        # Extract content using helper method
                        content = self._extract_content_from_llm_response(result)
                        
                        if not content:
                            logger.warning(f"analyze_grammar - Empty content from LLM (attempt {attempt+1})")
                            if attempt < self.MAX_RETRIES - 1:
                                await asyncio.sleep(self.RETRY_BASE_DELAY * (attempt + 1))
                                continue
                            return {"errors": [], "correct_skills": [], "linguistic_features": {}, "summary": "LLM retornou conteúdo vazio"}
                        
                        # Tentar parsear JSON da resposta
                        try:
                            # Remover markdown code blocks se presentes
                            if "```json" in content:
                                content = content.split("```json")[1].split("```")[0].strip()
                            elif "```" in content:
                                content = content.split("```")[1].split("```")[0].strip()
                            
                            parsed = json.loads(content)
                            # Garantir que todos os campos esperados existam
                            parsed.setdefault("errors", [])
                            parsed.setdefault("correct_skills", [])
                            parsed.setdefault("linguistic_features", {})
                            parsed.setdefault("summary", "Análise realizada")
                            
                            self.metrics["analyze_grammar_success"] += 1
                            return parsed
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse LLM response as JSON (attempt {attempt+1}): {e}, content: {content[:200]}")
                            if attempt < self.MAX_RETRIES - 1:
                                await asyncio.sleep(self.RETRY_BASE_DELAY * (attempt + 1))
                                continue
                            return {"errors": [], "correct_skills": [], "linguistic_features": {}, "summary": "Análise parcial"}
                    else:
                        error_text = await resp.text()
                        logger.error(f"LLM analysis failed (attempt {attempt+1}): {resp.status}, {error_text[:200]}")
                        if attempt < self.MAX_RETRIES - 1:
                            await asyncio.sleep(self.RETRY_BASE_DELAY * (attempt + 1))
                            continue
                        return {"errors": [], "correct_skills": [], "linguistic_features": {}, "summary": "Análise falhou"}
            except Exception as e:
                logger.error(f"Error calling LLM for grammar analysis (attempt {attempt+1}): {e}")
                import traceback
                logger.debug(f"Traceback: {traceback.format_exc()}")
                if attempt < self.MAX_RETRIES - 1:
                    await asyncio.sleep(self.RETRY_BASE_DELAY * (attempt + 1))
                    continue
        
        self.metrics["analyze_grammar_failures"] += 1
        return {"errors": [], "correct_skills": [], "linguistic_features": {}, "summary": "Erro na análise após retries"}
    
    async def semantic_skill_tagging(
        self,
        user_text: str,
        valid_skills: List[str],
        skill_descriptions: Optional[Dict[str, str]] = None
    ) -> Dict[str, float]:
        """
        Mapeia semanticamente o texto do aluno para skills relevantes (SINKT)
        
        Usa LLM para gerar embedding semântico do texto e comparar com descrições
        das skills, retornando top-3 skills mais relevantes com confidence scores.
        
        Args:
            user_text: Texto do aluno
            valid_skills: Lista de skill_ids válidas
            skill_descriptions: Dicionário opcional com descrições das skills
            
        Returns:
            Dicionário mapeando skill_id -> confidence score (0.0 a 1.0)
        """
        if not valid_skills:
            return {}
        
        # Construir lista de skills com descrições
        skills_list = []
        for skill_id in valid_skills:
            if skill_descriptions and skill_id in skill_descriptions:
                skills_list.append(f"- {skill_id}: {skill_descriptions[skill_id]}")
            else:
                skills_list.append(f"- {skill_id}")
        
        prompt = f"""Mapeie semanticamente o seguinte texto do aluno para as skills mais relevantes.

Texto do aluno: "{user_text}"

Skills disponíveis:
{chr(10).join(skills_list)}

Para cada skill, determine o quão relevante ela é para o texto do aluno baseado em:
- Similaridade semântica
- Uso de estruturas gramaticais relacionadas
- Vocabulário relacionado
- Contexto da conversa

Retorne em formato JSON com scores de 0.0 a 1.0 (1.0 = muito relevante, 0.0 = não relevante):
{{
    "semantic_skill_mapping": {{
        "skill_id_1": 0.85,
        "skill_id_2": 0.65,
        "skill_id_3": 0.45
    }},
    "top_skills": ["skill_id_1", "skill_id_2", "skill_id_3"],
    "reasoning": "Explicação breve do mapeamento"
}}"""
        
        try:
            async with self.session.post(
                f"{self.base_url}/chat",
                json={
                    "messages": [
                        {"role": "system", "content": "Você é um especialista em análise semântica e mapeamento de habilidades linguísticas."},
                        {"role": "user", "content": prompt}
                    ],
                    "model": "groq/llama-3.1-8b-instant"
                }
            ) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    content = self._extract_content_from_llm_response(result)
                    
                    if not content:
                        logger.warning("semantic_skill_tagging - Empty content from LLM")
                        return {}
                    
                    try:
                        # Remover markdown code blocks se presentes
                        if "```json" in content:
                            content = content.split("```json")[1].split("```")[0].strip()
                        elif "```" in content:
                            content = content.split("```")[1].split("```")[0].strip()
                        
                        parsed = json.loads(content)
                        semantic_mapping = parsed.get("semantic_skill_mapping", {})
                        
                        # Filtrar apenas skills válidas e normalizar scores usando helper
                        filtered_mapping = {}
                        for skill_id, score in semantic_mapping.items():
                            if skill_id in valid_skills:
                                filtered_mapping[skill_id] = self._validate_and_normalize_confidence(score, skill_id)
                        
                        # Ordenar por score e retornar top-3
                        sorted_skills = sorted(filtered_mapping.items(), key=lambda x: x[1], reverse=True)
                        return dict(sorted_skills[:3])
                    except (json.JSONDecodeError, ValueError) as e:
                        logger.warning(f"Failed to parse SINKT response: {e}")
                        import traceback
                        logger.debug(f"Traceback: {traceback.format_exc()}")
                        return {}
                else:
                    error_text = await resp.text()
                    logger.error(f"SINKT semantic tagging failed: {resp.status}, {error_text[:200]}")
                    return {}
        except Exception as e:
            logger.error(f"Error calling LLM for semantic skill tagging: {e}")
            return {}
    
    async def extract_skills(
        self,
        user_text: str,
        valid_skills: List[str],
        ai_text: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extrai skills usadas no texto do aluno usando LLM
        
        Foco exclusivo em identificar skills (sem análise de erros).
        Retorna skill_ids com confidence scores e features linguísticas.
        
        Args:
            user_text: Texto do aluno
            valid_skills: Lista obrigatória de skill_ids válidas
            ai_text: Resposta do AI (opcional, para contexto)
            
        Returns:
            Dicionário com:
            - skills: Lista de dicts com skill_id, confidence, linguistic_features
            - overall_linguistic_features: Features gerais do texto
            - summary: Resumo da extração
        """
        if not valid_skills:
            logger.warning("extract_skills called with empty valid_skills list")
            return {
                "skills": [],
                "overall_linguistic_features": {},
                "summary": "Nenhuma skill válida fornecida"
            }
        
        # Construir lista de skills para o prompt
        skills_list = "\n".join([f"- {skill_id}" for skill_id in valid_skills])
        
        prompt = f"""Você é um especialista em identificação de habilidades linguísticas. Analise o seguinte texto e identifique TODAS as skills que o aluno está usando.

Texto do aluno: "{user_text}"
{"Resposta do professor (contexto): " + ai_text if ai_text else ""}

Skills válidas disponíveis (você DEVE usar APENAS estas):
{skills_list}

IMPORTANTE:
- Você DEVE identificar skills que o aluno está usando no texto (corretas ou não)
- NÃO analise erros (isso é feito por outro módulo)
- Para cada skill identificada, forneça:
  1. skill_id (deve estar na lista de skills válidas acima)
  2. confidence (0.0 a 1.0) - quão confiante você está que essa skill foi usada
  3. linguistic_features específicas dessa skill (tense, person, number, register, domain, etc.)

EXTRAÇÃO OBRIGATÓRIA DE FEATURES LINGUÍSTICAS:
Para cada skill identificada, extraia features linguísticas:
- Para verbos: tense (present/past/future), person (1st/2nd/3rd), number (singular/plural), mood (indicative/subjunctive/conditional)
- Para vocabulário: register (formal/informal/neutral), domain (family/travel/emotions/daily_life/general)
- Para artigos: article_type (definite/indefinite)
- Para preposições: preposition_type (location/basic)

Além disso, extraia features linguísticas gerais do texto inteiro (overall_linguistic_features).

Exemplos:
- Texto "Eu fui ao banco ontem" → skill: verb_conjugation_past com features {{"tense": "past", "person": "1st", "number": "singular"}}
- Texto "Eles compraram comida" → skill: verb_conjugation_past com features {{"tense": "past", "person": "3rd", "number": "plural"}}

Retorne SEMPRE em formato JSON válido:
{{
    "skills": [
        {{
            "skill_id": "verb_conjugation_past",
            "confidence": 0.85,
            "linguistic_features": {{
                "tense": "past",
                "person": "1st",
                "number": "singular",
                "mood": "indicative"
            }}
        }},
        {{
            "skill_id": "prepositions_basic",
            "confidence": 0.70,
            "linguistic_features": {{
                "preposition_type": "location"
            }}
        }}
    ],
    "overall_linguistic_features": {{
        "tense": "past",
        "register": "informal",
        "domain": "daily_life"
    }},
    "summary": "Texto usa principalmente verbos no passado e preposições de localização."
}}

Lembre-se:
- Use APENAS skill_ids da lista fornecida
- Retorne confidence entre 0.0 e 1.0
- SEMPRE retorne linguistic_features, mesmo que vazio (objeto vazio)
- NUNCA retorne null"""
        
        self.metrics["extract_skills_calls"] += 1
        
        # Build prompt with size validation
        prompt = self._build_skills_prompt(user_text, valid_skills, ai_text, prompt_type="extract")
        
        # Retry logic
        for attempt in range(self.MAX_RETRIES):
            try:
                # Log the request being sent
                logger.debug(f"🔍 extract_skills - Sending request to LLM (attempt {attempt+1})")
                logger.debug(f"   User text: {user_text[:100]}")
                logger.debug(f"   Valid skills count: {len(valid_skills)}")
                logger.debug(f"   First 5 skills: {valid_skills[:5]}")
                logger.debug(f"   Prompt length: {len(prompt)} chars")
                
                timeout = self._calculate_timeout(len(valid_skills))
                async with self.session.post(
                    f"{self.base_url}/chat",
                    json={
                        "messages": [
                            {"role": "system", "content": "Você é um especialista em identificação de habilidades linguísticas. Foque apenas em identificar skills usadas, não analise erros."},
                            {"role": "user", "content": prompt}
                        ],
                        "model": "groq/llama-3.1-8b-instant"
                    },
                    timeout=aiohttp.ClientTimeout(total=timeout)
                ) as resp:
                    logger.debug(f"🔍 extract_skills - LLM response status: {resp.status}")
                    
                    if resp.status == 200:
                        result = await resp.json()
                        logger.debug(f"🔍 extract_skills - Raw LLM response structure: {list(result.keys())}")
                        
                        # Extract content using helper method
                        content = self._extract_content_from_llm_response(result)
                        
                        # LOG RAW CONTENT BEFORE PARSING
                        logger.info(f"🔍 extract_skills - RAW LLM CONTENT (before parsing, attempt {attempt+1}):")
                        logger.info(f"   Length: {len(content)} chars")
                        logger.info(f"   First 500 chars: {content[:500]}")
                        logger.info(f"   Last 200 chars: {content[-200:] if len(content) > 200 else content}")
                        logger.debug(f"   Full content: {content}")
                        
                        if not content or content.strip() == "":
                            logger.warning(f"🔍 extract_skills - LLM returned empty content (attempt {attempt+1})")
                            if attempt < self.MAX_RETRIES - 1:
                                await asyncio.sleep(self.RETRY_BASE_DELAY * (attempt + 1))
                                continue
                            return {
                                "skills": [],
                                "overall_linguistic_features": {},
                                "summary": "LLM retornou conteúdo vazio"
                            }
                        
                        try:
                            # Remover markdown code blocks se presentes
                            original_content = content
                            if "```json" in content:
                                logger.debug("🔍 extract_skills - Found ```json block, extracting...")
                                content = content.split("```json")[1].split("```")[0].strip()
                            elif "```" in content:
                                logger.debug("🔍 extract_skills - Found ``` block, extracting...")
                                content = content.split("```")[1].split("```")[0].strip()
                            
                            logger.debug(f"🔍 extract_skills - Content after markdown removal: {content[:300]}")
                            
                            parsed = json.loads(content)
                            logger.info(f"🔍 extract_skills - Successfully parsed JSON")
                            logger.debug(f"   Parsed keys: {list(parsed.keys())}")
                            
                            # Validar e filtrar skills
                            skills = parsed.get("skills", [])
                            logger.info(f"🔍 extract_skills - Found {len(skills)} skills in parsed JSON")
                            
                            validated_skills = []
                            for i, skill in enumerate(skills):
                                skill_id = skill.get("skill_id")
                                
                                if skill_id and skill_id in valid_skills:
                                    # Usar helper method para validar confidence
                                    confidence = self._validate_and_normalize_confidence(
                                        skill.get("confidence", 0.5),
                                        skill_id
                                    )
                                    validated_skills.append({
                                        "skill_id": skill_id,
                                        "confidence": confidence,
                                        "linguistic_features": skill.get("linguistic_features", {})
                                    })
                                    logger.debug(f"   Skill {i}: {skill_id} (confidence: {confidence:.2%})")
                                else:
                                    logger.debug(f"   Skill {i} '{skill_id}' not in valid_skills list or invalid")
                            
                            logger.info(f"🔍 extract_skills - Validated {len(validated_skills)} skills after filtering")
                            
                            # Update metrics
                            self.metrics["extract_skills_success"] += 1
                            if validated_skills:
                                # Update average skills detected
                                prev_avg = self.metrics["avg_skills_detected"]
                                prev_count = self.metrics["extract_skills_success"] - 1
                                self.metrics["avg_skills_detected"] = (
                                    (prev_avg * prev_count + len(validated_skills)) / 
                                    self.metrics["extract_skills_success"]
                                )
                            
                            return {
                                "skills": validated_skills,
                                "overall_linguistic_features": parsed.get("overall_linguistic_features", {}),
                                "summary": parsed.get("summary", "Extração de skills realizada")
                            }
                        except json.JSONDecodeError as e:
                            logger.error(f"🔍 extract_skills - JSON PARSE ERROR (attempt {attempt+1}):")
                            logger.error(f"   Error: {e}")
                            logger.error(f"   Content length: {len(content)}")
                            logger.error(f"   Content (first 500): {content[:500]}")
                            logger.error(f"   Content (last 500): {content[-500:] if len(content) > 500 else content}")
                            if attempt < self.MAX_RETRIES - 1:
                                await asyncio.sleep(self.RETRY_BASE_DELAY * (attempt + 1))
                                continue
                            return {
                                "skills": [],
                                "overall_linguistic_features": {},
                                "summary": f"Falha ao processar resposta do LLM: {str(e)}"
                            }
                    else:
                        error_text = await resp.text()
                        logger.error(f"🔍 extract_skills - LLM HTTP ERROR (attempt {attempt+1}):")
                        logger.error(f"   Status: {resp.status}")
                        logger.error(f"   Response: {error_text[:500]}")
                        if attempt < self.MAX_RETRIES - 1:
                            await asyncio.sleep(self.RETRY_BASE_DELAY * (attempt + 1))
                            continue
                        return {
                            "skills": [],
                            "overall_linguistic_features": {},
                            "summary": f"Falha na chamada ao LLM (status {resp.status})"
                        }
            except Exception as e:
                logger.error(f"🔍 extract_skills - EXCEPTION (attempt {attempt+1}):")
                logger.error(f"   Error type: {type(e).__name__}")
                logger.error(f"   Error message: {str(e)}")
                import traceback
                logger.error(f"   Traceback: {traceback.format_exc()}")
                if attempt < self.MAX_RETRIES - 1:
                    await asyncio.sleep(self.RETRY_BASE_DELAY * (attempt + 1))
                    continue
        
        self.metrics["extract_skills_failures"] += 1
        return {
            "skills": [],
            "overall_linguistic_features": {},
            "summary": "Erro na extração após retries"
        }
    
    async def estimate_complexity(self, text: str) -> Dict[str, Any]:
        """
        Estima nível CEFR baseado na complexidade do texto
        
        Args:
            text: Texto para análise
            
        Returns:
            Dicionário com estimativa de nível
        """
        prompt = f"""Estime o nível CEFR (A1, A2, B1, B2, C1, C2) do seguinte texto em português.

Texto: "{text}"

Considere:
- Complexidade do vocabulário
- Estruturas gramaticais usadas
- Tamanho e complexidade das frases
- Uso de expressões idiomáticas
- Fluência geral

Retorne em formato JSON:
{{
    "cefr_level": "B1",
    "confidence": 0.85,
    "vocabulary_complexity": "intermediate",
    "grammar_complexity": "intermediate",
    "sentence_length_avg": 12.5,
    "indicators": ["usa tempos verbais variados", "vocabulário intermediário"],
    "reasoning": "O texto demonstra..."
}}"""
        
        try:
            async with self.session.post(
                f"{self.base_url}/chat",
                json={
                    "messages": [
                        {"role": "system", "content": "Você é um especialista em avaliação de proficiência linguística."},
                        {"role": "user", "content": prompt}
                    ],
                    "model": "groq/llama-3.1-8b-instant"
                }
            ) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    
                    # Extract content - handle both formats
                    content = ""
                    if "choices" in result and result["choices"]:
                        choices = result.get("choices", [])
                        message = choices[0].get("message", {})
                        content = message.get("content", "{}")
                    elif "text" in result:
                        content = result.get("text", "{}")
                    else:
                        for field in ["content", "response", "output", "message"]:
                            if field in result:
                                content = str(result[field])
                                break
                        if not content:
                            content = "{}"
                    
                    # TODO: Parse JSON da resposta
                    return {
                        "cefr_level": "A2",
                        "confidence": 0.7,
                        "indicators": [],
                        "reasoning": "Análise automática"
                    }
                else:
                    logger.error(f"LLM complexity estimation failed: {resp.status}")
                    return {"cefr_level": "A1", "confidence": 0.5, "indicators": [], "reasoning": "Fallback"}
        except Exception as e:
            logger.error(f"Error calling LLM for complexity estimation: {e}")
            return {"cefr_level": "A1", "confidence": 0.5, "indicators": [], "reasoning": "Error"}

