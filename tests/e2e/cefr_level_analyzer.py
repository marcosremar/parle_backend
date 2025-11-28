"""
Analisador de Nível CEFR baseado em LLM (Sonnet 4.5)
Usa prompts estruturados baseados em papers acadêmicos para classificação precisa.
Elimina heurísticas em favor de análise linguística avançada via LLM.
"""

import os
import json
import asyncio
import aiohttp
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
project_root = Path(__file__).parent.parent.parent
load_dotenv(project_root / ".env")

# Import Phase 2, 3, 4 & 5 metrics
try:
    from lexical_diversity import LexicalDiversityCalculator
    from syntactic_complexity import SyntacticComplexityCalculator
    from speech_features import SpeechFeaturesAnalyzer
    from complexity_contours import ComplexityContoursCalculator
    from pairwise_classifier import PairwiseClassifier
    from semantic_cohesion import SemanticCohesionAnalyzer
    from discourse_markers import DiscourseMarkersAnalyzer
    from referential_cohesion import ReferentialCohesionAnalyzer
    from psycholinguistic_metrics import PsycholinguisticMetricsCalculator
except ImportError:
    # Fallback if modules not found
    LexicalDiversityCalculator = None
    SyntacticComplexityCalculator = None
    SpeechFeaturesAnalyzer = None
    ComplexityContoursCalculator = None
    PairwiseClassifier = None
    SemanticCohesionAnalyzer = None
    DiscourseMarkersAnalyzer = None
    ReferentialCohesionAnalyzer = None
    PsycholinguisticMetricsCalculator = None


# LLM Service Configuration
LLM_URL = os.getenv("LLM_URL", "http://localhost:8006")

# AKT Integration Configuration
STUDENT_MODEL_URL = os.getenv("STUDENT_MODEL_URL", "http://localhost:8900")

# Model selection: use Gemini Flash 2.5 via OpenRouter
if os.getenv("OPENROUTER_API_KEY"):
    SONNET_MODEL = "openrouter/google/gemini-2.5-flash"  # Gemini Flash 2.5
else:
    SONNET_MODEL = "openrouter/google/gemini-2.5-flash"  # Fallback to same model


# Critérios CEFR baseados nos papers acadêmicos (usados no prompt para o LLM)
CEFR_CRITERIA_PROMPT = """
# Critérios CEFR para Português (baseados em papers acadêmicos)

Você é um especialista em linguística e classificação CEFR para português. Sua tarefa é analisar um texto e identificar seu nível CEFR com base nos seguintes critérios extraídos de papers acadêmicos:

## Fontes dos Critérios:
1. **NILC-Metrix** (Leal et al., 2022): Métricas de complexidade sintática e lexical para português brasileiro
2. **Vajjala & Rama** (2021): Classificação automática de níveis CEFR usando contornos de complexidade e RNNs
3. **Arnold et al.** (2018): Predição de níveis CEFR baseada em métricas e textos completos
4. **Ribeiro et al.** (2024): Avaliação automática de complexidade textual em português europeu

## Critérios por Nível CEFR:

### **A1 (Iniciante)**
**Sintático:**
- Comprimento médio: 3-8 palavras por frase (frases muito curtas)
- Estruturas: APENAS Sujeito-Verbo-Objeto (SVO) simples
- Tempos verbais: APENAS presente do indicativo e imperativo básico
- Subordinação: AUSENTE ou quase ausente
- Voz passiva: AUSENTE
- Subjuntivo: AUSENTE
- Orações relativas: AUSENTES

**Lexical:**
- Vocabulário: ~500-1000 palavras mais frequentes (família, números, cores, comida básica)
- Palavras curtas: média 4-5 caracteres
- Repetição alta: type-token ratio muito baixo (muitas repetições de palavras)
- Expressões idiomáticas: AUSENTES
- Vocabulário técnico/abstrato: AUSENTE

**Discursivo:**
- Conectores: APENAS "e", "ou", "mas" (básicos)
- Coesão: Muito limitada, frases justapostas
- Marcadores discursivos: AUSENTES

---

### **A2 (Elementar)**
**Sintático:**
- Comprimento médio: 6-10 palavras por frase (MÍNIMO 6 palavras, não 3-5 como A1)
- Estruturas: SVO + algumas coordenações simples
- Tempos verbais: Presente, pretérito perfeito, futuro simples
- **OBRIGATÓRIO: A2 DEVE ter pelo menos 1 frase com pretérito perfeito (fui, visitei, comprei, fiz, vi, falei, gostei)**
- Subordinação: Muito limitada ("porque", "quando" básicos)
- **OBRIGATÓRIO: A2 DEVE ter pelo menos 1 conector "porque" ou "quando"**
- Voz passiva: AUSENTE ou muito rara
- Subjuntivo: AUSENTE
- Orações relativas: APENAS "que" em contextos muito simples

**Lexical:**
- Vocabulário: ~1000-2000 palavras (rotina diária, viagem, compras)
- **OBRIGATÓRIO: A2 DEVE ter vocabulário de rotina (ontem, amanhã, trabalho, fim de semana, mercado, compras)**
- Palavras curtas/médias: média 5-6 caracteres
- Repetição moderada: type-token ratio baixo-médio
- Expressões idiomáticas: Muito raras e apenas as mais comuns
- Vocabulário técnico: Mínimo

**Discursivo:**
- Conectores: "porque", "mas", "então", "e depois"
- Coesão: Básica, com conectores simples
- Marcadores discursivos: Mínimos ("por exemplo" raramente)

---

### **B1 (Intermediário)**
**Sintático:**
- Comprimento médio: 10-15 palavras por frase (MÍNIMO 10 palavras, não 6-8 como A2)
- Estruturas: Coordenações e subordinações simples
- Tempos verbais: Presente, pretéritos (perfeito/imperfeito), futuro, condicional simples
- **OBRIGATÓRIO: B1 DEVE ter diferentes tempos verbais (presente + passado OU futuro)**
- Subordinação: Presente ("porque", "quando", "se", "que")
- **OBRIGATÓRIO: B1 DEVE ter subordinação em pelo menos 2 frases**
- Voz passiva: Começa a aparecer (limitada)
- **OBRIGATÓRIO: B1 NÃO deve ter voz passiva complexa (apenas formas simples como "é feito", "foi construído")**
- Subjuntivo: Muito raro (apenas formas mais comuns como "seja")
- **OBRIGATÓRIO: B1 NÃO deve ter subjuntivo ativo (apenas "seja" em expressões fixas como "seja como for")**
- Orações relativas: "que", "quem", "onde" (uso básico)
- **OBRIGATÓRIO: B1 DEVE ter pelo menos 1 oração relativa ("que", "quem", "onde")**

**Lexical:**
- Vocabulário: ~2000-3500 palavras (opinião, trabalho, educação básica)
- Palavras médias: média 6-7 caracteres
- Repetição moderada: type-token ratio médio
- Expressões idiomáticas: Algumas expressões comuns
- Vocabulário técnico: Limitado, apenas áreas familiares

**Discursivo:**
- Conectores: Repertório ampliado ("portanto", "além disso", "por outro lado")
- Coesão: Boa, com progressão lógica
- Marcadores discursivos: Alguns ("por exemplo", "ou seja")

---

### **B2 (Intermediário Superior)**
**Sintático:**
- Comprimento médio: 12-20 palavras por frase
- Estruturas: Subordinações variadas, coordenações complexas
- Tempos verbais: Todos os tempos, incluindo compostos
- Subordinação: Variada e natural
- Voz passiva: Presente e natural
- Subjuntivo: Presente (ainda com alguns erros possíveis)
- Orações relativas: Todos os pronomes relativos básicos ("cujo" começa a aparecer)

**Lexical:**
- Vocabulário: ~3500-5000 palavras (temas abstratos, opinião complexa)
- Palavras médias/longas: média 7-8 caracteres
- Repetição baixa: type-token ratio médio-alto
- Expressões idiomáticas: Moderadas, uso apropriado
- Vocabulário técnico: Moderado, em áreas de interesse

**Discursivo:**
- Conectores: Repertório amplo e variado
- Coesão: Boa, com uso de referências anafóricas
- Marcadores discursivos: Variados ("isto é", "em outras palavras", "dessa forma")

---

### **C1 (Avançado)**
**Sintático:**
- Comprimento médio: 15-25 palavras por frase
- Estruturas: Subordinações múltiplas, encadeamento complexo
- Tempos verbais: Todos os tempos, uso sofisticado e preciso
- Subordinação: Avançada, múltiplos níveis
- Voz passiva: Natural e variada
- Subjuntivo: Natural e correto em todos os contextos
- Orações relativas: Todos os tipos, incluindo complexos

**Lexical:**
- Vocabulário: ~5000-8000 palavras (abstrato, técnico, nuances)
- Palavras longas: média 8-9 caracteres
- Repetição muito baixa: type-token ratio alto
- Expressões idiomáticas: Naturais e variadas
- Vocabulário técnico: Amplo, várias áreas

**Discursivo:**
- Conectores: Completos e sofisticados
- Coesão: Muito boa, texto fluente e natural
- Marcadores discursivos: Sofisticados e variados

---

### **C2 (Proficiência)**
**Sintático:**
- Comprimento médio: 20-30+ palavras por frase (MÍNIMO 20 palavras)
- Estruturas: Encadeamento muito complexo, estruturas raras/literárias
- Tempos verbais: Uso nativo, incluindo formas raras
- Subordinação: Muito avançada, encadeamentos longos
- **OBRIGATÓRIO: C2 DEVE ter múltiplas e encadeadas subordinações (3 ou mais)**
- Voz passiva: Sofisticada, incluindo formas raras
- **OBRIGATÓRIO: C2 DEVE ter voz passiva sofisticada ou formas raras**
- Subjuntivo: Nativo, incluindo formas compostas e raras
- **OBRIGATÓRIO: C2 DEVE ter subjuntivo em formas compostas ou raras**
- Orações relativas: Todas, incluindo as mais raras e complexas

**Lexical:**
- Vocabulário: 8000+ palavras (nativo, criativo, especializado)
- **OBRIGATÓRIO: C2 DEVE ter vocabulário acadêmico/formal específico (depreende-se, transcende, meticulosa, inerente, contemple, entrelaçando-se, logrei, cuja obra)**
- Palavras muito longas: média 9+ caracteres
- Repetição mínima: type-token ratio muito alto
- Expressões idiomáticas: Sofisticadas, criativas, até neologismos
- Vocabulário técnico: Muito amplo e especializado

**Discursivo:**
- Conectores: Completos, criativos, sofisticados
- Coesão: Perfeita, nível nativo
- Marcadores discursivos: Sofisticados, criativos

---

## IMPORTANTE:
- **Textos curtos (1-3 frases):** Considere que o comprimento médio pode ser enganoso. Foque mais em estruturas sintáticas, vocabulário e presença/ausência de elementos complexos.
- **Type-token ratio:** Ignore ou dê peso mínimo em textos com menos de 30 palavras.
- **Priorize sintaxe e gramática:** Para português, a presença de subjuntivo, voz passiva, orações relativas e subordinação múltipla são os melhores indicadores de nível alto.
- **Contexto de conversação:** Se o texto é uma resposta conversacional curta, ajuste as expectativas de comprimento de frase para baixo (conversas naturais têm frases mais curtas que textos escritos).
"""


async def identify_cefr_level_llm(
    text: str,
    session: aiohttp.ClientSession,
    expected_level: Optional[str] = None
) -> Dict[str, Any]:
    """
    Identifica o nível CEFR de um texto usando o LLM (Sonnet 4.5) com prompt especializado.
    
    Args:
        text: O texto a ser analisado
        session: Sessão aiohttp para fazer chamadas HTTP
        expected_level: Nível esperado (opcional, para contexto)
    
    Returns:
        Dicionário com: identified_level, confidence, explanation, syntactic_analysis, lexical_analysis, discursive_analysis
    """
    
    # Construir prompt para o LLM
    analysis_prompt = f"""{CEFR_CRITERIA_PROMPT}

## TAREFA:
Analise o seguinte texto em português e identifique seu nível CEFR (A1, A2, B1, B2, C1, ou C2).

**Texto a analisar:**
"{text}"

{f'**Nota:** Este texto foi gerado para o nível {expected_level}. Verifique se está adequado.' if expected_level else ''}

## INSTRUÇÕES:
1. Analise o texto cuidadosamente considerando:
   - Comprimento das frases e palavras
   - Estruturas sintáticas (subordinação, voz passiva, subjuntivo, orações relativas)
   - Vocabulário (complexidade, variedade, abstração)
   - Coesão e marcadores discursivos
   
2. Compare com os critérios acima para cada nível CEFR

3. Retorne um JSON válido no seguinte formato:
{{
  "identified_level": "A1|A2|B1|B2|C1|C2",
  "confidence": 0.0-1.0,
  "explanation": "Explicação clara de por que o texto pertence a este nível, mencionando os critérios que o justificam",
  "syntactic_analysis": {{
    "avg_words_per_sentence": <número>,
    "has_subordination": true|false,
    "subordination_level": "absent|limited|moderate|advanced|very_advanced",
    "has_passive_voice": true|false,
    "has_subjunctive": true|false,
    "has_relative_clauses": true|false,
    "complexity_assessment": "Descrição da complexidade sintática observada"
  }},
  "lexical_analysis": {{
    "vocabulary_level": "basic|intermediate|advanced|native",
    "avg_word_length": <número>,
    "vocabulary_diversity": "low|medium|high|very_high",
    "has_technical_vocabulary": true|false,
    "has_idiomatic_expressions": true|false,
    "complexity_assessment": "Descrição da complexidade lexical observada"
  }},
  "discursive_analysis": {{
    "connectors_level": "basic|intermediate|advanced|sophisticated",
    "cohesion_quality": "poor|basic|good|very_good|excellent",
    "has_discourse_markers": true|false,
    "complexity_assessment": "Descrição da complexidade discursiva observada"
  }},
  "key_indicators": [
    "Lista de 3-5 indicadores chave que justificam o nível identificado"
  ]
}}

**RETORNE APENAS O JSON, SEM TEXTO ADICIONAL.**
"""

    try:
        # Chamar LLM service
        async with session.post(
            f"{LLM_URL}/chat",
            json={
                "messages": [
                    {"role": "system", "content": "You are a CEFR language proficiency expert. Analyze text and return ONLY valid JSON."},
                    {"role": "user", "content": analysis_prompt}
                ],
                "model": SONNET_MODEL,
                "temperature": 0.3,  # Baixa temperatura para análise mais consistente
                "max_tokens": 2000
            },
            timeout=aiohttp.ClientTimeout(total=60)
        ) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                print(f"❌ LLM service error ({resp.status}): {error_text}")
                return _fallback_analysis(text, f"LLM service error: {resp.status}")
            
            result = await resp.json()
            llm_response = result.get("text") or result.get("response") or ""
            
            if not llm_response:
                print(f"❌ LLM returned empty response")
                return _fallback_analysis(text, "Empty LLM response")
            
            # Parse JSON do LLM
            try:
                # Limpar possível markdown ou texto extra
                llm_response = llm_response.strip()
                if "```json" in llm_response:
                    llm_response = llm_response.split("```json")[1].split("```")[0].strip()
                elif "```" in llm_response:
                    llm_response = llm_response.split("```")[1].split("```")[0].strip()
                
                analysis = json.loads(llm_response)
                
                # Validar campos obrigatórios
                required_fields = ["identified_level", "confidence", "explanation"]
                for field in required_fields:
                    if field not in analysis:
                        print(f"⚠️ Missing required field: {field}")
                        analysis[field] = "N/A" if field != "confidence" else 0.5
                
                # Garantir que confidence é um float entre 0 e 1
                if isinstance(analysis["confidence"], (int, float)):
                    analysis["confidence"] = max(0.0, min(1.0, float(analysis["confidence"])))
                else:
                    analysis["confidence"] = 0.5
                
                # Adicionar campos opcionais se faltarem
                analysis.setdefault("syntactic_analysis", {})
                analysis.setdefault("lexical_analysis", {})
                analysis.setdefault("discursive_analysis", {})
                analysis.setdefault("key_indicators", [])
                
                return analysis
                
            except json.JSONDecodeError as e:
                print(f"❌ JSON parsing error: {e}")
                print(f"Raw LLM response: {llm_response[:500]}")
                return _fallback_analysis(text, f"JSON parsing error: {e}")
    
    except asyncio.TimeoutError:
        print(f"❌ LLM request timeout")
        return _fallback_analysis(text, "Request timeout")
    except Exception as e:
        print(f"❌ Unexpected error in identify_cefr_level_llm: {e}")
        return _fallback_analysis(text, f"Unexpected error: {e}")


def _fallback_analysis(text: str, error_reason: str) -> Dict[str, Any]:
    """Análise de fallback simples quando o LLM falha"""
    words = text.split()
    word_count = len(words)
    
    # Heurística muito simples como fallback
    if word_count < 10:
        level = "A1"
    elif word_count < 20:
        level = "A2"
    elif word_count < 35:
        level = "B1"
    elif word_count < 50:
        level = "B2"
    elif word_count < 70:
        level = "C1"
    else:
        level = "C2"
    
    return {
        "identified_level": level,
        "confidence": 0.3,
        "explanation": f"Fallback analysis due to: {error_reason}. Simple heuristic based on word count ({word_count} words) suggests {level}.",
        "syntactic_analysis": {},
        "lexical_analysis": {},
        "discursive_analysis": {},
        "key_indicators": ["Fallback heuristic - not reliable"],
        "error": error_reason
    }


def generate_report(
    text: str,
    analysis: Dict[str, Any],
    target_level: Optional[str] = None,
    timestamp: Optional[datetime] = None
) -> str:
    """Gera relatório completo de análise CEFR baseado na análise do LLM"""
    if timestamp is None:
        timestamp = datetime.now()
    
    identified_level = analysis.get("identified_level", "UNKNOWN")
    confidence = analysis.get("confidence", 0.0)
    explanation = analysis.get("explanation", "Nenhuma explicação disponível")
    
    syntactic = analysis.get("syntactic_analysis", {})
    lexical = analysis.get("lexical_analysis", {})
    discursive = analysis.get("discursive_analysis", {})
    key_indicators = analysis.get("key_indicators", [])
    
    # Status de conformidade
    status = ""
    if target_level:
        if identified_level == target_level:
            status = "**Status:** ✅ CONFORME"
        else:
            status = f"**Status:** ⚠️ NÃO CONFORME (esperado: {target_level}, identificado: {identified_level})"
    
    report = f"""# Relatório de Análise de Nível CEFR (LLM-based)

**Data/Hora:** {timestamp.strftime('%Y-%m-%d %H:%M:%S')}
**Modelo:** Gemini Flash 2.5 (via OpenRouter)

## Texto Analisado
```
{text}
```

## Nível CEFR Identificado
**Nível:** {identified_level}  
**Confiança:** {confidence:.0%}  
{f'**Nível Esperado:** {target_level}' if target_level else ''}  
{status}

## Explicação Geral
{explanation}

## Análise Detalhada

### Análise Sintática
"""
    
    if syntactic:
        report += f"""
- **Comprimento médio de sentença:** {syntactic.get('avg_words_per_sentence', 'N/A')} palavras
- **Subordinação:** {'Presente' if syntactic.get('has_subordination') else 'Ausente'} (nível: {syntactic.get('subordination_level', 'N/A')})
- **Voz passiva:** {'Presente' if syntactic.get('has_passive_voice') else 'Ausente'}
- **Subjuntivo:** {'Presente' if syntactic.get('has_subjunctive') else 'Ausente'}
- **Orações relativas:** {'Presente' if syntactic.get('has_relative_clauses') else 'Ausente'}
- **Avaliação:** {syntactic.get('complexity_assessment', 'N/A')}
"""
    else:
        report += "\n*Análise sintática não disponível*\n"
    
    report += "\n### Análise Lexical\n"
    if lexical:
        report += f"""
- **Nível de vocabulário:** {lexical.get('vocabulary_level', 'N/A')}
- **Comprimento médio de palavras:** {lexical.get('avg_word_length', 'N/A')} caracteres
- **Diversidade vocabular:** {lexical.get('vocabulary_diversity', 'N/A')}
- **Vocabulário técnico:** {'Presente' if lexical.get('has_technical_vocabulary') else 'Ausente'}
- **Expressões idiomáticas:** {'Presente' if lexical.get('has_idiomatic_expressions') else 'Ausente'}
- **Avaliação:** {lexical.get('complexity_assessment', 'N/A')}
"""
    else:
        report += "\n*Análise lexical não disponível*\n"
    
    report += "\n### Análise Discursiva\n"
    if discursive:
        report += f"""
- **Nível de conectores:** {discursive.get('connectors_level', 'N/A')}
- **Qualidade de coesão:** {discursive.get('cohesion_quality', 'N/A')}
- **Marcadores discursivos:** {'Presente' if discursive.get('has_discourse_markers') else 'Ausente'}
- **Avaliação:** {discursive.get('complexity_assessment', 'N/A')}
"""
    else:
        report += "\n*Análise discursiva não disponível*\n"
    
    report += "\n## Indicadores-Chave do Nível Identificado\n\n"
    if key_indicators:
        for i, indicator in enumerate(key_indicators, 1):
            report += f"{i}. {indicator}\n"
    else:
        report += "*Nenhum indicador-chave fornecido*\n"
    
    report += f"""

## Metodologia

Esta análise foi realizada usando o modelo de linguagem **Gemini Flash 2.5** (via OpenRouter), que foi instruído com critérios CEFR baseados nos seguintes papers acadêmicos:

1. **Leal, S. E., et al. (2022).** NILC-Metrix: Assessing the complexity of written and spoken language in Brazilian Portuguese.
2. **Vajjala, S., & Rama, T. (2021).** Automated classification of written proficiency levels on the CEFR-scale through complexity contours and RNNs.
3. **Arnold, T., et al. (2018).** Predicting CEFRL levels in learner English on the basis of metrics and full texts.
4. **Ribeiro, E., et al. (2024).** Avaliação automática do nível de complexidade de textos em português europeu.

O LLM analisa o texto considerando:
- Complexidade sintática (subordinação, tempos verbais, estruturas complexas)
- Riqueza lexical (vocabulário, diversidade, comprimento de palavras)
- Coesão discursiva (conectores, marcadores, progressão textual)

Para mais detalhes sobre os critérios, consulte: `docs/CEFR_COMPLEXITY_MARKERS_IMPLEMENTATION.md`
"""
    
    return report


def save_report(report: str, filename: Optional[str] = None, output_dir: Optional[Path] = None) -> Path:
    """Salva relatório em arquivo"""
    if output_dir is None:
        output_dir = Path(__file__).parent / "reports"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"cefr_analysis_llm_{timestamp}.md"
    
    filepath = output_dir / filename
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report)
    
    return filepath


# ============================================================================
# MÉTRICAS QUANTITATIVAS (Baseadas em Arnold et al., 2018 e NILC-Metrix)
# ============================================================================

import re
from collections import Counter

async def calculate_syntactic_metrics(text: str, session: Optional[aiohttp.ClientSession] = None) -> Dict[str, float]:
    """
    Calcula métricas sintáticas quantitativas baseadas nos papers.
    Agora inclui Yngve/Frazier depth, T-units, e subordination index via dependency parser.
    """
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Basic metrics (heuristic-based, for fallback)
    basic_metrics = {
        "avg_words_per_sentence": 0.0,
        "subordination_ratio": 0.0,
        "passive_voice_ratio": 0.0,
        "subjunctive_ratio": 0.0,
        "relative_clauses_ratio": 0.0,
        "yngve_mean_depth": 0.0,
        "frazier_mean_depth": 0.0,
        "num_t_units": 0,
        "avg_words_per_tunit": 0.0,
        "subordination_index": 0.0
    }
    
    if not sentences:
        return basic_metrics
    
    words_per_sentence = [len(s.split()) for s in sentences]
    avg_words = sum(words_per_sentence) / len(words_per_sentence)
    
    # Subordinação (conectores subordinativos) - heuristic
    subordinating_conjunctions = [
        'porque', 'quando', 'se', 'embora', 'enquanto', 'conforme', 
        'já que', 'uma vez que', 'caso', 'caso contrário', 'mesmo que',
        'apesar de', 'embora', 'ainda que', 'conquanto', 'visto que'
    ]
    subord_count = sum(1 for conj in subordinating_conjunctions 
                      if re.search(rf'\b{conj}\b', text, re.IGNORECASE))
    subordination_ratio = subord_count / len(sentences)
    
    # Voz passiva (padrões comuns)
    passive_patterns = [
        r'\b(é|foi|será|são|foram|seriam)\s+(feito|dito|visto|conhecido|escrito|construído|recomendado)',
        r'\b(é|foi|será)\s+[a-z]+ado\b',
        r'\b(é|foi|será)\s+[a-z]+ido\b'
    ]
    passive_count = sum(1 for pattern in passive_patterns 
                       if re.search(pattern, text, re.IGNORECASE))
    passive_ratio = passive_count / len(sentences)
    
    # Subjuntivo (formas comuns)
    subjunctive_patterns = [
        r'\b(seja|fosse|estivesse|tivesse|fizer|fizerem|fizesse|fizessem)\b',
        r'\b(que|se|caso|embora)\b.*\b(seja|fosse|estivesse|tivesse)\b'
    ]
    subjunctive_count = sum(1 for pattern in subjunctive_patterns 
                           if re.search(pattern, text, re.IGNORECASE))
    subjunctive_ratio = subjunctive_count / len(sentences)
    
    # Orações relativas
    relative_pronouns = ['que', 'quem', 'onde', 'cujo', 'cuja', 'cujos', 'cujas']
    relative_count = sum(1 for pronoun in relative_pronouns 
                        if re.search(rf'\b{pronoun}\b', text, re.IGNORECASE))
    relative_ratio = relative_count / len(sentences)
    
    basic_metrics.update({
        "avg_words_per_sentence": round(avg_words, 2),
        "subordination_ratio": round(subordination_ratio, 3),
        "passive_voice_ratio": round(passive_ratio, 3),
        "subjunctive_ratio": round(subjunctive_ratio, 3),
        "relative_clauses_ratio": round(relative_ratio, 3)
    })
    
    # Try to get advanced metrics from linguistic_analysis service (Phase 2)
    if session:
        try:
            syntactic_calc = SyntacticComplexityCalculator()
            advanced_metrics = await syntactic_calc.calculate_all_syntactic_metrics(text)
            
            basic_metrics.update({
                "yngve_mean_depth": round(advanced_metrics.get("yngve_mean_depth", 0.0), 2),
                "frazier_mean_depth": round(advanced_metrics.get("frazier_mean_depth", 0.0), 2),
                "num_t_units": advanced_metrics.get("num_t_units", 0),
                "avg_words_per_tunit": round(advanced_metrics.get("avg_words_per_tunit", 0.0), 2),
                "subordination_index": round(advanced_metrics.get("subordination_index", 0.0), 3),
                "clause_density": round(advanced_metrics.get("clause_density", 0.0), 2)
            })
        except Exception as e:
            print(f"⚠️ Could not get advanced syntactic metrics: {e}")
    
    return basic_metrics


def detect_a2_markers(text: str) -> Dict[str, bool]:
    """
    Detecta marcadores específicos de nível A2.
    
    Returns:
        Dict com flags booleanos para cada marcador A2
    """
    text_lower = text.lower()
    
    # Pretérito perfeito (fui, visitei, comprei, fiz, vi, falei, gostei, etc.)
    preterito_perfeito_pattern = r'\b(fui|visitei|comprei|fiz|vi|falei|gostei|trabalhei|estudei|comi|bebi|saí|cheguei|parti|voltei)\b'
    has_preterito = bool(re.search(preterito_perfeito_pattern, text_lower))
    
    # Conectores "porque" ou "quando"
    has_connectors = bool(re.search(r'\b(porque|quando)\b', text_lower))
    
    # Vocabulário de rotina
    routine_vocab_pattern = r'\b(ontem|amanhã|trabalho|fim de semana|mercado|compras|escritório|cinema|amigos)\b'
    has_routine_vocab = bool(re.search(routine_vocab_pattern, text_lower))
    
    return {
        "has_preterito_perfeito": has_preterito,
        "has_connectors": has_connectors,
        "has_routine_vocabulary": has_routine_vocab
    }


def detect_b1_markers(text: str, syntactic_metrics: Optional[Dict[str, float]] = None) -> Dict[str, bool]:
    """
    Detecta marcadores específicos de nível B1.
    
    Args:
        text: Texto a analisar
        syntactic_metrics: Métricas sintáticas (opcional, para evitar recálculo)
    
    Returns:
        Dict com flags booleanos para cada marcador B1
    """
    text_lower = text.lower()
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Comprimento de frase >= 10 palavras
    if syntactic_metrics:
        avg_words = syntactic_metrics.get("avg_words_per_sentence", 0.0)
    else:
        words_per_sentence = [len(s.split()) for s in sentences]
        avg_words = sum(words_per_sentence) / len(words_per_sentence) if words_per_sentence else 0.0
    has_long_sentences = avg_words >= 10.0
    
    # Subordinação em pelo menos 2 frases
    subordinating_conjunctions = ['porque', 'quando', 'se', 'que', 'embora', 'enquanto']
    subord_count = sum(1 for conj in subordinating_conjunctions 
                      if re.search(rf'\b{conj}\b', text_lower))
    has_multiple_subord = subord_count >= 2
    
    # B1 DEVE ter pelo menos 1 conector além de "porque" e "quando"
    advanced_subord = ['se', 'embora', 'enquanto', 'já que', 'uma vez que', 'caso', 'mesmo que', 'apesar de']
    has_advanced_subord = any(re.search(rf'\b{conj}\b', text_lower) for conj in advanced_subord)
    
    # B1 DEVE ter pretérito imperfeito OU futuro
    preterito_imperfeito = bool(re.search(r'\b(era|tinha|fazia|gostava|trabalhava|estudava|morava|via|dizia)\b', text_lower))
    futuro = bool(re.search(r'\b(vou|vai|vamos|farei|será|terá|virá|dirá)\b', text_lower))
    has_different_tenses = preterito_imperfeito or futuro
    
    # B1 DEVE ter pelo menos 1 oração relativa
    relative_pronouns = ['que', 'quem', 'onde']
    has_relative_clause = any(re.search(rf'\b{pron}\b', text_lower) for pron in relative_pronouns)
    
    # Ausência de subjuntivo complexo (apenas "seja" em expressões fixas permitido)
    complex_subjunctive_pattern = r'\b(fosse|estivesse|tivesse|fizesse|fizessem|fossem|estivessem|tivessem)\b'
    has_complex_subjunctive = bool(re.search(complex_subjunctive_pattern, text_lower))
    # "seja" sozinho não conta como complexo
    seja_pattern = r'\bseja\b'
    has_simple_seja = bool(re.search(seja_pattern, text_lower))
    no_complex_subjunctive = not has_complex_subjunctive or (has_simple_seja and not has_complex_subjunctive)
    
    # Ausência de voz passiva complexa (apenas simples como "é feito", "foi construído")
    complex_passive_pattern = r'\b(seria|seriam|foram|são|serão)\s+[a-z]+(ado|ido)\s+por\b'
    has_complex_passive = bool(re.search(complex_passive_pattern, text_lower))
    no_complex_passive = not has_complex_passive
    
    return {
        "has_long_sentences": has_long_sentences,
        "has_multiple_subord": has_multiple_subord,
        "has_advanced_subord": has_advanced_subord,
        "has_different_tenses": has_different_tenses,
        "has_relative_clause": has_relative_clause,
        "no_complex_subjunctive": no_complex_subjunctive,
        "no_complex_passive": no_complex_passive
    }


def detect_c2_markers(text: str, syntactic_metrics: Optional[Dict[str, float]] = None) -> Dict[str, bool]:
    """
    Detecta marcadores específicos de nível C2.
    
    Args:
        text: Texto a analisar
        syntactic_metrics: Métricas sintáticas (opcional, para evitar recálculo)
    
    Returns:
        Dict com flags booleanos para cada marcador C2
    """
    text_lower = text.lower()
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Vocabulário acadêmico/formal
    academic_vocab_pattern = r'\b(depreende-se|transcende|meticulosa|inerente|contemple|entrelaçando-se|logrei|cuja obra|não obstante|considerando que)\b'
    has_academic_vocab = bool(re.search(academic_vocab_pattern, text_lower))
    
    # Comprimento de frase >= 20 palavras
    if syntactic_metrics:
        avg_words = syntactic_metrics.get("avg_words_per_sentence", 0.0)
    else:
        words_per_sentence = [len(s.split()) for s in sentences]
        avg_words = sum(words_per_sentence) / len(words_per_sentence) if words_per_sentence else 0.0
    has_very_long_sentences = avg_words >= 20.0
    
    # Múltiplas subordinações (3+)
    subordinating_conjunctions = [
        'porque', 'quando', 'se', 'embora', 'enquanto', 'conforme', 
        'já que', 'uma vez que', 'caso', 'mesmo que', 'apesar de', 
        'ainda que', 'conquanto', 'visto que', 'considerando que', 'não obstante'
    ]
    subord_count = sum(1 for conj in subordinating_conjunctions 
                      if re.search(rf'\b{conj}\b', text_lower))
    has_multiple_subordinations = subord_count >= 3
    
    # Voz passiva sofisticada
    sophisticated_passive_patterns = [
        r'\b(é|foi|será|são|foram)\s+[a-z]+(ado|ido)\s+[a-z]+\s+[a-z]+',  # Passiva com complemento
        r'\b(seria|seriam)\s+[a-z]+(ado|ido)',  # Condicional passiva
        r'\b(havia|houve)\s+sido\s+[a-z]+(ado|ido)'  # Passiva composta
    ]
    has_sophisticated_passive = any(
        re.search(pattern, text_lower) for pattern in sophisticated_passive_patterns
    )
    
    return {
        "has_academic_vocabulary": has_academic_vocab,
        "has_very_long_sentences": has_very_long_sentences,
        "has_multiple_subordinations": has_multiple_subordinations,
        "has_sophisticated_passive": has_sophisticated_passive
    }


def calculate_lexical_metrics(text: str) -> Dict[str, float]:
    """
    Calcula métricas lexicais quantitativas baseadas nos papers.
    Agora inclui MTLD, MATTR, Zipf-TTR, e hapax legomena.
    """
    # Basic metrics (legacy)
    words = [w.lower() for w in re.findall(r'\b\w+\b', text)]
    
    if not words:
        return {
            "type_token_ratio": 0.0,
            "avg_word_length": 0.0,
            "vocabulary_size": 0,
            "total_words": 0,
            "mtld": 0.0,
            "mattr": 0.0,
            "zipf_ttr": 0.0,
            "hapax_percentage": 0.0
        }
    
    unique_words = set(words)
    type_token_ratio = len(unique_words) / len(words) if words else 0.0
    avg_word_length = sum(len(w) for w in words) / len(words) if words else 0.0
    
    # Advanced lexical diversity metrics (Phase 2)
    if LexicalDiversityCalculator is None:
        mtld_data = {"mtld": 0.0}
        mattr_data = {"mattr": 0.0}
        zipf_data = {"zipf_ttr": 0.0}
        hapax_data = {"hapax_count": 0, "hapax_ratio": 0.0}
    else:
        lexical_calc = LexicalDiversityCalculator()
        mtld_data = lexical_calc.calculate_mtld(text)
        mattr_data = lexical_calc.calculate_mattr(text)
        zipf_data = lexical_calc.calculate_zipf_normalized_ttr(text)
        hapax_data = lexical_calc.calculate_hapax_legomena(text)
    
    # Word tokens and word types (most important features according to Arnold et al., 2018)
    word_tokens = len(words)  # Total word tokens
    word_types = len(unique_words)  # Unique word types
    tokens_types_ratio = word_types / word_tokens if word_tokens > 0 else 0.0
    
    return {
        "type_token_ratio": round(type_token_ratio, 3),
        "avg_word_length": round(avg_word_length, 2),
        "vocabulary_size": word_types,  # Same as word_types
        "total_words": word_tokens,  # Same as word_tokens
        "word_tokens": word_tokens,  # Explicit: total tokens (most important feature)
        "word_types": word_types,  # Explicit: unique types (most important feature)
        "tokens_types_ratio": round(tokens_types_ratio, 3),
        "mtld": round(mtld_data.get("mtld", 0.0), 2),
        "mattr": round(mattr_data.get("mattr", 0.0), 3),
        "zipf_ttr": round(zipf_data.get("zipf_ttr", 0.0), 3),
        "hapax_percentage": round(hapax_data.get("hapax_percentage", 0.0), 2)
    }


def score_level_by_metrics(
    syntactic_metrics: Dict[str, float],
    lexical_metrics: Dict[str, float],
    level: str,
    text: str = ""
) -> float:
    """
    Calcula um score (0-1) indicando quão bem o texto se encaixa no nível CEFR
    baseado apenas em métricas quantitativas.
    """
    # Critérios por nível (baseados nos papers)
    criteria = {
        "A1": {
            "avg_words": (3, 8),
            "subordination_max": 0.0,
            "passive_max": 0.0,
            "subjunctive_max": 0.0,
            "relative_max": 0.0,
            "ttr": (0.30, 0.40),
            "avg_word_len": (4, 5),
            "word_tokens": (5, 50),  # Based on Arnold et al. (2018)
            "word_types": (5, 30)
        },
        "A2": {
            "avg_words": (6, 10),  # Mínimo 6 palavras (não 3-5 como A1)
            "subordination_max": 0.1,
            "passive_max": 0.0,
            "subjunctive_max": 0.0,
            "relative_max": 0.05,
            "ttr": (0.40, 0.50),
            "avg_word_len": (5, 6),
            "word_tokens": (10, 80),  # Ajustado: mais baixo para capturar textos A2 mais curtos
            "word_types": (8, 40)  # Ajustado: mais baixo para capturar textos A2 mais curtos
        },
        "B1": {
            "avg_words": (10, 15),  # Mínimo 10 palavras (não 6-8 como A2)
            "subordination_max": 0.3,
            "passive_max": 0.1,
            "subjunctive_max": 0.1,
            "relative_max": 0.2,
            "ttr": (0.50, 0.60),
            "avg_word_len": (6, 7),
            "word_tokens": (50, 200),  # Based on Arnold et al. (2018)
            "word_types": (30, 100)
        },
        "B2": {
            "avg_words": (12, 20),
            "subordination_max": 0.5,
            "passive_max": 0.3,
            "subjunctive_max": 0.3,
            "relative_max": 0.4,
            "ttr": (0.60, 0.70),
            "avg_word_len": (7, 8),
            "word_tokens": (100, 400),  # Based on Arnold et al. (2018)
            "word_types": (60, 200)
        },
        "C1": {
            "avg_words": (15, 25),
            "subordination_max": 1.0,
            "passive_max": 0.6,
            "subjunctive_max": 0.6,
            "relative_max": 0.8,
            "ttr": (0.70, 0.80),
            "avg_word_len": (8, 9),
            "word_tokens": (200, 800),  # Based on Arnold et al. (2018)
            "word_types": (120, 400)
        },
        "C2": {
            "avg_words": (20, 50),
            "subordination_max": 1.0,
            "passive_max": 1.0,
            "subjunctive_max": 1.0,
            "relative_max": 1.0,
            "ttr": (0.80, 0.90),
            "avg_word_len": (9, 15),
            "word_tokens": (400, 10000),  # Based on Arnold et al. (2018), upper bound flexible
            "word_types": (200, 5000)
        }
    }
    
    crit = criteria.get(level, criteria["A1"])
    score = 0.0
    max_score = 0.0
    
    # Comprimento de frase (peso: 30%)
    max_score += 0.3
    avg_words = syntactic_metrics["avg_words_per_sentence"]
    min_w, max_w = crit["avg_words"]
    if min_w <= avg_words <= max_w:
        score += 0.3
    elif avg_words < min_w:
        score += 0.3 * (avg_words / min_w)  # Penalizar se muito curto
    else:
        score += 0.3 * (max_w / avg_words)  # Penalizar se muito longo
    
    # Subordinação (peso: 15%)
    max_score += 0.15
    subord = syntactic_metrics["subordination_ratio"]
    if subord <= crit["subordination_max"]:
        score += 0.15 * (subord / crit["subordination_max"]) if crit["subordination_max"] > 0 else 0.15
    else:
        score += 0.15 * (crit["subordination_max"] / subord) if subord > 0 else 0
    
    # Voz passiva (peso: 10%)
    max_score += 0.1
    passive = syntactic_metrics["passive_voice_ratio"]
    if passive <= crit["passive_max"]:
        score += 0.1 * (passive / crit["passive_max"]) if crit["passive_max"] > 0 else 0.1
    else:
        score += 0.1 * (crit["passive_max"] / passive) if passive > 0 else 0
    
    # Subjuntivo (peso: 10%)
    max_score += 0.1
    subj = syntactic_metrics["subjunctive_ratio"]
    if subj <= crit["subjunctive_max"]:
        score += 0.1 * (subj / crit["subjunctive_max"]) if crit["subjunctive_max"] > 0 else 0.1
    else:
        score += 0.1 * (crit["subjunctive_max"] / subj) if subj > 0 else 0
    
    # Type-Token Ratio (peso: 20%)
    max_score += 0.2
    ttr = lexical_metrics["type_token_ratio"]
    min_ttr, max_ttr = crit["ttr"]
    if min_ttr <= ttr <= max_ttr:
        score += 0.2
    elif ttr < min_ttr:
        score += 0.2 * (ttr / min_ttr)
    else:
        score += 0.2 * (max_ttr / ttr)
    
    # Comprimento médio de palavras (peso: 10%)
    max_score += 0.1
    avg_len = lexical_metrics["avg_word_length"]
    min_len, max_len = crit["avg_word_len"]
    if min_len <= avg_len <= max_len:
        score += 0.1
    elif avg_len < min_len:
        score += 0.1 * (avg_len / min_len)
    else:
        score += 0.1 * (max_len / avg_len)
    
    # Word Tokens (peso: 15% - MOST IMPORTANT according to Arnold et al., 2018)
    max_score += 0.15
    word_tokens = lexical_metrics.get("word_tokens", lexical_metrics.get("total_words", 0))
    if "word_tokens" in crit:
        min_tokens, max_tokens = crit["word_tokens"]
        if min_tokens <= word_tokens <= max_tokens:
            score += 0.15
        elif word_tokens < min_tokens:
            score += 0.15 * (word_tokens / min_tokens) if min_tokens > 0 else 0.0
        else:
            # For very long texts, cap the score
            score += 0.15 * (max_tokens / word_tokens) if word_tokens > 0 else 0.0
    else:
        # Fallback: use total_words if word_tokens not in criteria
        score += 0.15 * min(1.0, word_tokens / 100.0)  # Normalize by 100
    
    # Word Types (peso: 15% - MOST IMPORTANT according to Arnold et al., 2018)
    max_score += 0.15
    word_types = lexical_metrics.get("word_types", lexical_metrics.get("vocabulary_size", 0))
    if "word_types" in crit:
        min_types, max_types = crit["word_types"]
        if min_types <= word_types <= max_types:
            score += 0.15
        elif word_types < min_types:
            score += 0.15 * (word_types / min_types) if min_types > 0 else 0.0
        else:
            score += 0.15 * (max_types / word_types) if word_types > 0 else 0.0
    else:
        # Fallback: use vocabulary_size if word_types not in criteria
        score += 0.15 * min(1.0, word_types / 50.0)  # Normalize by 50
    
    # Note: MWS normalization is handled separately in identify_cefr_level_hybrid()
    # to allow access to quantitative_analysis data
    
    # Calcular score base
    base_score = score / max_score if max_score > 0 else 0.0
    
    # Adicionar bônus por marcadores específicos (se texto fornecido)
    if text:
        marker_bonus = 0.0
        
        if level == "A2":
            a2_markers = detect_a2_markers(text)
            if a2_markers["has_preterito_perfeito"]:
                marker_bonus += 0.25  # +25 pontos (aumentado de 20)
            if a2_markers["has_connectors"]:
                marker_bonus += 0.20  # +20 pontos (aumentado de 15)
            if a2_markers["has_routine_vocabulary"]:
                marker_bonus += 0.15  # +15 pontos (novo marcador)
        
        elif level == "B1":
            b1_markers = detect_b1_markers(text, syntactic_metrics)
            if b1_markers["has_long_sentences"]:
                marker_bonus += 0.15  # +15 pontos
            if b1_markers["has_multiple_subord"]:
                marker_bonus += 0.15  # +15 pontos
            if b1_markers["has_advanced_subord"]:
                marker_bonus += 0.20  # +20 pontos (marcador importante)
            if b1_markers["has_different_tenses"]:
                marker_bonus += 0.20  # +20 pontos (marcador importante)
            if b1_markers["has_relative_clause"]:
                marker_bonus += 0.15  # +15 pontos
            if b1_markers["no_complex_subjunctive"]:
                marker_bonus += 0.10  # +10 pontos (indica B1, não B2)
            if b1_markers["no_complex_passive"]:
                marker_bonus += 0.10  # +10 pontos (indica B1, não B2)
        
        elif level == "C2":
            c2_markers = detect_c2_markers(text, syntactic_metrics)
            if c2_markers["has_academic_vocabulary"]:
                marker_bonus += 0.30  # +30 pontos
            if c2_markers["has_very_long_sentences"]:
                marker_bonus += 0.20  # +20 pontos
            if c2_markers["has_multiple_subordinations"]:
                marker_bonus += 0.25  # +25 pontos
            if c2_markers["has_sophisticated_passive"]:
                marker_bonus += 0.20  # +20 pontos
        
        # Aplicar bônus (limitado a 1.0)
        base_score = min(1.0, base_score + marker_bonus)
    
    return round(base_score, 3)


def adjust_metrics_for_speech(
    syntactic_metrics: Dict[str, float],
    lexical_metrics: Dict[str, float]
) -> tuple[Dict[str, float], Dict[str, float]]:
    """
    Ajusta métricas para fala conversacional.
    
    Este sistema é especializado para avaliação de LINGUAGEM FALADA.
    Baseado em pesquisas que mostram diferenças sistemáticas entre fala e escrita.
    
    Ajustes aplicados (baseados em pesquisas acadêmicas):
    - Comprimento de frase: 30% mais curto em fala conversacional
    - TTR: 15% mais baixo em fala (repetição é normal e esperada)
    - Subordinação: 25% menos em fala (coordenação é preferida)
    
    Referências:
    - Leal et al. (2022): NILC-Metrix para português falado
    - Arnold et al. (2018): Diferenças entre avaliação escrita e oral
    - CEFR Companion Volume (2020): Descritores para produção oral
    """
    adjusted_syntactic = syntactic_metrics.copy()
    adjusted_lexical = lexical_metrics.copy()
    
    # Comprimento de frase: 30% mais curto em fala conversacional
    adjusted_syntactic["avg_words_per_sentence"] = round(
        adjusted_syntactic["avg_words_per_sentence"] / 1.3, 2
    )
    
    # Subordinação: 25% menos em fala (coordenação é preferida)
    adjusted_syntactic["subordination_ratio"] = round(
        adjusted_syntactic["subordination_ratio"] * 0.75, 3
    )
    
    # TTR: 15% mais baixo em fala (repetição é normal)
    adjusted_lexical["type_token_ratio"] = round(
        adjusted_lexical["type_token_ratio"] * 0.85, 3
    )
    
    return adjusted_syntactic, adjusted_lexical


async def get_akt_cefr_progress(
    user_id: str,
    session: aiohttp.ClientSession
) -> Optional[Dict[str, Any]]:
    """
    Obtém o progresso CEFR do estudante baseado no AKT.
    
    Args:
        user_id: ID do estudante
        session: Sessão HTTP
        
    Returns:
        Dicionário com progresso CEFR do AKT ou None se não disponível
    """
    try:
        url = f"{STUDENT_MODEL_URL}/api/student/{user_id}/cefr_progress"
        timeout = aiohttp.ClientTimeout(total=5)  # 5 second timeout
        async with session.get(url, timeout=timeout) as response:
            if response.status == 200:
                data = await response.json()
                return data
            elif response.status == 404:
                # Estudante não existe no sistema AKT (primeira vez)
                return None
            else:
                return None
    except aiohttp.ClientError:
        # Silenciosamente retorna None se AKT não disponível
        return None
    except Exception:
        # Qualquer outro erro também retorna None
        return None


def validate_with_akt(
    text_cefr_level: str,
    text_confidence: float,
    akt_progress: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Valida classificação CEFR do texto com progresso AKT (validação cruzada).
    
    Opção 1: Ajusta confiança baseado em convergência, não muda nível.
    
    Args:
        text_cefr_level: Nível CEFR identificado pelo texto
        text_confidence: Confiança da classificação do texto (0.0-1.0)
        akt_progress: Progresso CEFR do AKT (None se não disponível)
        
    Returns:
        Dicionário com confiança ajustada e informações de validação
    """
    if not akt_progress:
        # AKT não disponível (estudante novo ou serviço indisponível)
        return {
            "adjusted_confidence": text_confidence,
            "akt_available": False,
            "convergence": None,
            "validation_note": "AKT não disponível (estudante novo ou serviço indisponível)"
        }
    
    # Extrair nível estimado do AKT
    akt_estimated_level = akt_progress.get("current_estimated_level", "A1")
    akt_details = akt_progress.get("cefr_details", {})
    
    # Obter progresso do nível identificado pelo texto
    text_level_progress = akt_details.get(text_cefr_level, {}).get("progress", 0.0)
    
    # Calcular convergência
    levels_order = ["A1", "A2", "B1", "B2", "C1", "C2"]
    text_idx = levels_order.index(text_cefr_level) if text_cefr_level in levels_order else 0
    akt_idx = levels_order.index(akt_estimated_level) if akt_estimated_level in levels_order else 0
    
    # Diferença de níveis (0 = mesmo nível, 1 = adjacente, etc.)
    level_diff = abs(text_idx - akt_idx)
    
    # Calcular ajuste de confiança baseado em convergência
    confidence_adjustment = 0.0
    
    if level_diff == 0:
        # Mesmo nível → alta convergência
        if text_level_progress >= 0.5:
            # AKT confirma com progresso sólido
            confidence_adjustment = +0.15  # +15% confiança
        elif text_level_progress >= 0.2:
            # AKT confirma mas progresso baixo
            confidence_adjustment = +0.10  # +10% confiança
        else:
            # Mesmo nível mas progresso muito baixo (pode ser inconsistente)
            confidence_adjustment = +0.05  # +5% confiança
    elif level_diff == 1:
        # Níveis adjacentes → convergência moderada
        if text_level_progress >= 0.3:
            # Progresso razoável no nível do texto
            confidence_adjustment = +0.05  # +5% confiança
        else:
            # Progresso baixo, pode ser inconsistente
            confidence_adjustment = -0.05  # -5% confiança
    else:
        # Níveis distantes → baixa convergência
        if level_diff == 2:
            confidence_adjustment = -0.15  # -15% confiança
        else:
            confidence_adjustment = -0.25  # -25% confiança
    
    # Ajustar confiança (limitado entre 0.0 e 1.0)
    adjusted_confidence = max(0.0, min(1.0, text_confidence + confidence_adjustment))
    
    # Determinar status de convergência
    if level_diff == 0 and text_level_progress >= 0.5:
        convergence_status = "high"
        convergence_note = f"AKT confirma nível {text_cefr_level} com progresso sólido ({text_level_progress:.0%})"
    elif level_diff == 0:
        convergence_status = "moderate"
        convergence_note = f"AKT confirma nível {text_cefr_level} mas progresso baixo ({text_level_progress:.0%})"
    elif level_diff == 1:
        convergence_status = "low"
        convergence_note = f"AKT sugere {akt_estimated_level} mas texto indica {text_cefr_level} (níveis adjacentes)"
    else:
        convergence_status = "very_low"
        convergence_note = f"⚠️ Inconsistência: AKT sugere {akt_estimated_level} mas texto indica {text_cefr_level}"
    
    return {
        "adjusted_confidence": round(adjusted_confidence, 3),
        "akt_available": True,
        "akt_estimated_level": akt_estimated_level,
        "text_level_progress": round(text_level_progress, 3),
        "level_difference": level_diff,
        "convergence": convergence_status,
        "confidence_adjustment": round(confidence_adjustment, 3),
        "validation_note": convergence_note
    }


async def analyze_text_features_quantitative(
    text: str,
    session: aiohttp.ClientSession
) -> Dict[str, Any]:
    """
    Análise quantitativa completa com todas as métricas das Fases 2-5.
    
    Inclui:
    - Fase 2: Métricas lexicais avançadas (MTLD, MATTR, Zipf-TTR, hapax)
    - Fase 2: Métricas sintáticas avançadas (Yngve, Frazier, T-units)
    - Fase 2: Features específicas de fala (MWS, repetições, disfluências)
    - Fase 4: Coesão semântica (LSA), marcadores discursivos, coesão referencial
    - Fase 5: Métricas psicolinguísticas (AoA, concreteness, familiarity, imageability)
    
    Args:
        text: Texto a analisar
        session: Sessão HTTP
        
    Returns:
        Dicionário com todas as métricas quantitativas
    """
    # Basic metrics (legacy)
    basic_syntactic = await calculate_syntactic_metrics(text, session)
    basic_lexical = calculate_lexical_metrics(text)
    
    result = {
        "basic_syntactic": basic_syntactic,
        "basic_lexical": basic_lexical,
        "lexical_diversity": {},
        "syntactic_complexity": {},
        "speech_features": {},
        "semantic_cohesion": {},
        "discourse_markers": {},
        "referential_cohesion": {},
        "psycholinguistic": {}
    }
    
    # Lexical diversity metrics (Phase 2)
    if LexicalDiversityCalculator:
        try:
            lexical_calc = LexicalDiversityCalculator()
            result["lexical_diversity"] = lexical_calc.calculate_all_lexical_metrics(text)
        except Exception as e:
            print(f"⚠️ Error calculating lexical diversity: {e}")
    
    # Syntactic complexity metrics (Phase 2)
    if SyntacticComplexityCalculator:
        try:
            syntactic_calc = SyntacticComplexityCalculator()
            result["syntactic_complexity"] = await syntactic_calc.calculate_all_syntactic_metrics(text)
        except Exception as e:
            print(f"⚠️ Error calculating syntactic complexity: {e}")
    
    # Speech-specific features (Phase 2)
    if SpeechFeaturesAnalyzer:
        try:
            speech_analyzer = SpeechFeaturesAnalyzer()
            # Use "spoken" register by default (this is a speech-to-speech app)
            result["speech_features"] = speech_analyzer.calculate_all_speech_features(text, register="spoken")
        except Exception as e:
            print(f"⚠️ Error calculating speech features: {e}")
    
    # Semantic cohesion (Phase 4)
    if SemanticCohesionAnalyzer:
        try:
            cohesion_analyzer = SemanticCohesionAnalyzer()
            result["semantic_cohesion"] = cohesion_analyzer.calculate_cohesion(text)
        except Exception as e:
            print(f"⚠️ Error calculating semantic cohesion: {e}")
    
    # Discourse markers (Phase 4)
    if DiscourseMarkersAnalyzer:
        try:
            discourse_analyzer = DiscourseMarkersAnalyzer()
            result["discourse_markers"] = discourse_analyzer.analyze_markers(text)
        except Exception as e:
            print(f"⚠️ Error analyzing discourse markers: {e}")
    
    # Referential cohesion (Phase 4)
    if ReferentialCohesionAnalyzer:
        try:
            referential_analyzer = ReferentialCohesionAnalyzer()
            result["referential_cohesion"] = await referential_analyzer.analyze_cohesion(text)
        except Exception as e:
            print(f"⚠️ Error analyzing referential cohesion: {e}")
    
    # Psycholinguistic metrics (Phase 5)
    if PsycholinguisticMetricsCalculator:
        try:
            psycholinguistic_calc = PsycholinguisticMetricsCalculator()
            result["psycholinguistic"] = psycholinguistic_calc.calculate_metrics(text)
        except Exception as e:
            print(f"⚠️ Error calculating psycholinguistic metrics: {e}")
    
    return result


async def identify_cefr_level_hybrid(
    text: str,
    session: aiohttp.ClientSession,
    expected_level: Optional[str] = None,
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Classificador HÍBRIDO: Combina LLM (60%) + Métricas Quantitativas (40%)
    Especializado para LINGUAGEM FALADA (speech/conversational).
    Baseado nas melhores práticas dos papers acadêmicos.
    
    IMPORTANTE: Este sistema é para avaliação de FALA, não escrita.
    As métricas são automaticamente ajustadas para características da fala conversacional.
    
    Agora inclui todas as métricas da Fase 2 (MTLD, MATTR, Yngve, Frazier, T-units, MWS, etc.)
    E validação cruzada com AKT (Opção 1).
    
    Args:
        text: Transcrição de fala a ser analisada
        session: Sessão HTTP
        expected_level: Nível esperado (opcional)
        user_id: ID do estudante para validação AKT (opcional)
    
    Returns:
        Análise completa com nível identificado, confiança e justificativa
    """
    # 1. Análise LLM (qualitativa)
    llm_analysis = await identify_cefr_level_llm(text, session, expected_level)
    llm_level = llm_analysis.get("identified_level", "A1")
    llm_confidence = llm_analysis.get("confidence", 0.5)
    
    # 2. Métricas quantitativas (agora com Phase 2 metrics)
    syntactic_metrics = await calculate_syntactic_metrics(text, session)
    lexical_metrics = calculate_lexical_metrics(text)
    
    # 2.5. Ajustar métricas para fala conversacional (sempre aplicado, pois é app de speech)
    syntactic_metrics, lexical_metrics = adjust_metrics_for_speech(
        syntactic_metrics, lexical_metrics
    )
    
    # 2.6. Análise quantitativa completa (Phase 2)
    quantitative_analysis = await analyze_text_features_quantitative(text, session)
    
    # 2.7. Extract MWS normalized scores for scoring (Phase 4 - NILC-Metrix, 2022)
    speech_feat = quantitative_analysis.get("speech_features", {})
    mws_data = speech_feat.get("mean_word_span", {})
    mws_normalized = mws_data.get("normalized_mws", {})
    mws_spoken_percentile = mws_normalized.get("spoken", {}).get("percentile", 0.5) if mws_normalized else 0.5
    
    # 3. Score por nível usando métricas
    levels = ["A1", "A2", "B1", "B2", "C1", "C2"]
    metrics_scores = {}
    
    # Expected MWS percentiles per level (based on spoken register normalization)
    mws_level_expectations = {
        "A1": (0.0, 0.3),   # Low MWS
        "A2": (0.2, 0.4),   # Low-medium MWS
        "B1": (0.3, 0.6),   # Medium MWS
        "B2": (0.5, 0.7),   # Medium-high MWS
        "C1": (0.6, 0.9),   # High MWS
        "C2": (0.8, 1.0)    # Very high MWS
    }
    
    for level in levels:
        base_metrics_score = score_level_by_metrics(
            syntactic_metrics, lexical_metrics, level, text
        )
        
        # Add MWS bonus (Phase 4): Higher MWS percentile indicates higher proficiency
        mws_min, mws_max = mws_level_expectations.get(level, (0.0, 1.0))
        if mws_min <= mws_spoken_percentile <= mws_max:
            mws_bonus = 0.05  # +5% bonus
        elif mws_spoken_percentile < mws_min:
            mws_bonus = 0.05 * (mws_spoken_percentile / mws_min) if mws_min > 0 else 0.0
        else:
            mws_bonus = 0.05 * (mws_max / mws_spoken_percentile) if mws_spoken_percentile > 0 else 0.0
        
        metrics_scores[level] = min(1.0, base_metrics_score + mws_bonus)
    
    # 4. Nível com maior score nas métricas
    metrics_level = max(metrics_scores.items(), key=lambda x: x[1])[0]
    metrics_confidence = metrics_scores[metrics_level]
    
    # 4.5. Complexity contours (Phase 3) - Enhanced with statistical features
    contours_data = {}
    contour_features = {}  # Extract features for scoring
    if ComplexityContoursCalculator:
        try:
            contours_calc = ComplexityContoursCalculator()
            contours_data = contours_calc.calculate_contours(text)
            
            # Extract statistical features from contours (Vajjala & Rama, 2021)
            stats = contours_data.get("statistics", {})
            contour_features = {
                "mean_complexity": stats.get("mean_complexity", 0.0),
                "std_complexity": stats.get("std_complexity", 0.0),
                "variance": stats.get("variance", 0.0),
                "trend": stats.get("trend", 0.0),
                "coefficient_of_variation": stats.get("coefficient_of_variation", 0.0),
                "interquartile_range": stats.get("interquartile_range", 0.0)
            }
        except Exception as e:
            print(f"⚠️ Error calculating complexity contours: {e}")
    
    # 4.6. Pairwise classification (Phase 3)
    pairwise_result = {}
    if PairwiseClassifier:
        try:
            pairwise_classifier = PairwiseClassifier()
            # Prepare LLM confidence dict
            llm_conf_dict = {level: (llm_confidence if level == llm_level else 0.0) for level in levels}
            pairwise_result = pairwise_classifier.predict_level(quantitative_analysis, llm_conf_dict)
        except Exception as e:
            print(f"⚠️ Error in pairwise classification: {e}")
    
    # 5. Ensemble: Pesos específicos por nível (updated weights)
    ensemble_scores = {}
    pairwise_votes = pairwise_result.get("vote_distribution", {})
    
    # Calcular scores para todos os níveis primeiro
    for level in levels:
        llm_weight = 0.4 if level == llm_level else 0.0
        metrics_weight = 0.3 * metrics_scores[level]
        pairwise_weight = 0.3 * pairwise_votes.get(level, 0.0)
        ensemble_scores[level] = llm_weight * llm_confidence + metrics_weight + pairwise_weight
    
    # Determinar nível final
    final_level = max(ensemble_scores.items(), key=lambda x: x[1])[0]
    
    # Aplicar pesos específicos por nível (optimized based on papers)
    # Transition pairs (A1-A2, A2-B1, B1-B2) benefit more from pairwise (Arnold et al., 2018)
    if final_level == "A2":
        # A2: 25% LLM + 35% Métricas + 40% Pairwise (transition pair, pairwise is key)
        llm_weight = 0.25 if llm_level == "A2" else 0.0
        metrics_weight = 0.35
        pairwise_weight = 0.40
        final_confidence = (llm_weight * llm_confidence + 
                          metrics_weight * metrics_scores["A2"] + 
                          pairwise_weight * pairwise_votes.get("A2", 0.0))
    elif final_level == "B1":
        # B1: 30% LLM + 30% Métricas + 40% Pairwise (transition pair, pairwise is key)
        llm_weight = 0.30 if llm_level == "B1" else 0.0
        metrics_weight = 0.30
        pairwise_weight = 0.40
        final_confidence = (llm_weight * llm_confidence + 
                          metrics_weight * metrics_scores["B1"] + 
                          pairwise_weight * pairwise_votes.get("B1", 0.0))
    elif final_level == "B2":
        # B2: 30% LLM + 35% Métricas + 35% Pairwise (transition pair)
        llm_weight = 0.30 if llm_level == "B2" else 0.0
        metrics_weight = 0.35
        pairwise_weight = 0.35
        final_confidence = (llm_weight * llm_confidence + 
                          metrics_weight * metrics_scores["B2"] + 
                          pairwise_weight * pairwise_votes.get("B2", 0.0))
    else:
        # Outros níveis: 40% LLM + 30% Métricas + 30% Pairwise (padrão)
        llm_weight = 0.4 if final_level == llm_level else 0.0
        metrics_weight = 0.3
        pairwise_weight = 0.3
        final_confidence = (llm_weight * llm_confidence + 
                          metrics_weight * metrics_scores[final_level] + 
                          pairwise_weight * pairwise_votes.get(final_level, 0.0))
    
    # 6. Validação Cruzada com AKT (Opção 1)
    akt_validation = None
    if user_id:
        try:
            akt_progress = await get_akt_cefr_progress(user_id, session)
            if akt_progress:
                akt_validation = validate_with_akt(
                    final_level,
                    final_confidence,
                    akt_progress
                )
                # Ajustar confiança final baseado na validação AKT
                final_confidence = akt_validation["adjusted_confidence"]
        except Exception as e:
            print(f"⚠️ Error in AKT validation: {e}")
    
    # 7. Justificativa combinada (agora com Phase 2 metrics e AKT)
    lexical_div = quantitative_analysis.get("lexical_diversity", {})
    syntactic_comp = quantitative_analysis.get("syntactic_complexity", {})
    speech_feat = quantitative_analysis.get("speech_features", {})
    
    justification = f"""
**Análise Híbrida (LLM + Métricas Quantitativas Avançadas) - Linguagem Falada:**

**LLM Analysis:**
- Nível identificado: {llm_level} (confiança: {llm_confidence:.0%})
- Justificativa: {llm_analysis.get('explanation', 'N/A')[:200]}...

**Métricas Quantitativas Básicas (Ajustadas para Fala Conversacional):**
- Comprimento médio de frase: {syntactic_metrics['avg_words_per_sentence']:.1f} palavras
- Subordinação: {syntactic_metrics['subordination_ratio']:.1%} das frases
- Type-Token Ratio: {lexical_metrics['type_token_ratio']:.3f}
- Comprimento médio de palavras: {lexical_metrics['avg_word_length']:.1f} caracteres

**Métricas Lexicais Avançadas (Phase 2):**
- MTLD: {lexical_metrics.get('mtld', 0.0):.2f}
- MATTR: {lexical_metrics.get('mattr', 0.0):.3f}
- Zipf-TTR: {lexical_metrics.get('zipf_ttr', 0.0):.3f}
- Hapax legomena: {lexical_metrics.get('hapax_percentage', 0.0):.1f}%

**Métricas Sintáticas Avançadas (Phase 2):**
- Yngve depth: {syntactic_metrics.get('yngve_mean_depth', 0.0):.2f}
- Frazier depth: {syntactic_metrics.get('frazier_mean_depth', 0.0):.2f}
- T-units: {syntactic_metrics.get('num_t_units', 0)}
- Subordination index: {syntactic_metrics.get('subordination_index', 0.0):.3f}

**Features de Fala (Phase 2):**
- Mean Word Span: {speech_feat.get('mean_word_span', {}).get('mws', 0.0):.2f}
- Repetition rate: {speech_feat.get('repetitions', {}).get('repetition_rate', 0.0):.3f}
- Disfluency rate: {speech_feat.get('disfluencies', {}).get('disfluency_rate', 0.0):.3f}

**Nota**: Métricas ajustadas para fala conversacional (frases 30% mais curtas, TTR 15% mais baixo, subordinação 25% menos), conforme pesquisas sobre diferenças entre fala e escrita.

**Scores por Nível (Métricas):**
{chr(10).join([f"- {level}: {score:.0%}" for level, score in sorted(metrics_scores.items(), key=lambda x: x[1], reverse=True)])}

**Complexity Contours (Phase 3):**
- Mean complexity: {contours_data.get('statistics', {}).get('mean_complexity', 0.0):.3f}
- Complexity trend: {contours_data.get('statistics', {}).get('trend', 0.0):.3f}
- Number of windows: {contours_data.get('num_windows', 0)}

**Pairwise Classification (Phase 3):**
- Predicted level: {pairwise_result.get('predicted_level', 'N/A')}
- Confidence: {pairwise_result.get('confidence', 0.0):.0%}
- Vote distribution: {', '.join([f"{level}: {vote:.0%}" for level, vote in sorted(pairwise_result.get('vote_distribution', {}).items(), key=lambda x: x[1], reverse=True)[:3]])}

**Validação Cruzada com AKT (Opção 1):**
{akt_validation['validation_note'] if akt_validation and akt_validation.get('akt_available') else 'AKT não disponível (estudante novo ou serviço indisponível)'}
{f"- AKT nível estimado: {akt_validation['akt_estimated_level']}" if akt_validation and akt_validation.get('akt_available') else ""}
{f"- Progresso no nível {final_level}: {akt_validation['text_level_progress']:.0%}" if akt_validation and akt_validation.get('akt_available') else ""}
{f"- Convergência: {akt_validation['convergence']}" if akt_validation and akt_validation.get('akt_available') else ""}
{f"- Ajuste de confiança: {akt_validation['confidence_adjustment']:+.1%}" if akt_validation and akt_validation.get('akt_available') else ""}

**Decisão Final (Ensemble 40% LLM + 30% Métricas + 30% Pairwise + Validação AKT):**
- Nível final: {final_level} (confiança: {final_confidence:.0%})
"""
    
    return {
        "identified_level": final_level,
        "confidence": round(final_confidence, 2),
        "explanation": justification,
        "llm_analysis": llm_analysis,
        "syntactic_metrics": syntactic_metrics,
        "lexical_metrics": lexical_metrics,
        "quantitative_analysis": quantitative_analysis,
        "complexity_contours": contours_data,
        "pairwise_classification": pairwise_result,
        "metrics_scores": metrics_scores,
        "ensemble_scores": ensemble_scores,
        "akt_validation": akt_validation,  # NOVO: incluir validação AKT
        "method": "hybrid_llm_metrics_phase2_phase3_akt"
    }


# Manter compatibilidade com código existente
def identify_cefr_level(text: str) -> Dict[str, Any]:
    """
    Função síncrona de compatibilidade (deprecada).
    Retorna análise de fallback simples.
    Use identify_cefr_level_llm() para análise completa com LLM.
    """
    import asyncio
    
    async def _async_wrapper():
        async with aiohttp.ClientSession() as session:
            return await identify_cefr_level_llm(text, session)
    
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Se já estamos em um loop assíncrono, retornar fallback
            return _fallback_analysis(text, "Cannot run async in existing event loop")
        return loop.run_until_complete(_async_wrapper())
    except Exception as e:
        return _fallback_analysis(text, f"Async error: {e}")
