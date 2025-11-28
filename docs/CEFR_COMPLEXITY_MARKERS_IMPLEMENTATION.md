# Marcadores de Complexidade CEFR - Guia de Implementação

Este documento detalha os marcadores de complexidade identificados nos papers acadêmicos e como implementá-los no sistema de avaliação.

## Marcadores por Nível CEFR

### A1 (Iniciante)

**Sintáticos:**
- Comprimento médio de sentença: 3-5 palavras
- Estruturas: Apenas SVO (Sujeito-Verbo-Objeto)
- Subordinação: Nenhuma
- Coordenação: Apenas "e", "ou" básicos
- Tempos verbais: Apenas presente do indicativo, imperativo simples
- Voz passiva: Não permitida
- Subjuntivo: Não permitido
- Orações relativas: Não permitidas

**Lexicais:**
- Vocabulário: 500-1000 palavras mais frequentes
- Tipo/token ratio: Baixo (0.3-0.4)
- Comprimento médio de palavras: 4-5 caracteres
- Expressões idiomáticas: Nenhuma
- Vocabulário técnico: Nenhum

**Discursivos:**
- Conectores: Apenas "e", "ou", "mas" básicos
- Marcadores discursivos: Nenhum
- Referências anafóricas: Mínimas

### A2 (Elementar)

**Sintáticos:**
- Comprimento médio de sentença: 5-8 palavras
- Estruturas: SVO, algumas inversões simples
- Subordinação: Muito limitada (apenas "porque", "quando")
- Coordenação: "e", "ou", "mas", "porque"
- Tempos verbais: Presente, passado simples, futuro perifrástico
- Voz passiva: Não permitida
- Subjuntivo: Não permitido
- Orações relativas: Muito simples ("que" básico)

**Lexicais:**
- Vocabulário: 1000-2000 palavras mais frequentes
- Tipo/token ratio: 0.4-0.5
- Comprimento médio de palavras: 5-6 caracteres
- Expressões idiomáticas: Muito limitadas
- Vocabulário técnico: Mínimo

**Discursivos:**
- Conectores: "porque", "mas", "então"
- Marcadores discursivos: Mínimos
- Referências anafóricas: Limitadas

### B1 (Intermediário)

**Sintáticos:**
- Comprimento médio de sentença: 8-12 palavras
- Estruturas: SVO, inversões, algumas estruturas complexas
- Subordinação: Limitada ("porque", "quando", "se", "que")
- Coordenação: Todos os coordenativos básicos
- Tempos verbais: Presente, passado (simples e composto), futuro, condicional
- Voz passiva: Iniciando (muito simples)
- Subjuntivo: Não permitido
- Orações relativas: Simples ("que", "quem", "onde")

**Lexicais:**
- Vocabulário: 2000-4000 palavras
- Tipo/token ratio: 0.5-0.6
- Comprimento médio de palavras: 6-7 caracteres
- Expressões idiomáticas: Algumas básicas
- Vocabulário técnico: Limitado

**Discursivos:**
- Conectores: "porque", "mas", "então", "portanto", "além disso"
- Marcadores discursivos: Alguns básicos
- Referências anafóricas: Moderadas

### B2 (Intermediário Superior)

**Sintáticos:**
- Comprimento médio de sentença: 12-18 palavras
- Estruturas: Todas as estruturas básicas, algumas complexas
- Subordinação: Moderada (vários tipos)
- Coordenação: Todos os coordenativos
- Tempos verbais: Todos os tempos básicos e alguns compostos
- Voz passiva: Permitida (moderada)
- Subjuntivo: Iniciando (muito limitado)
- Orações relativas: Todas ("que", "quem", "onde", "cujo")

**Lexicais:**
- Vocabulário: 4000-6000 palavras
- Tipo/token ratio: 0.6-0.7
- Comprimento médio de palavras: 7-8 caracteres
- Expressões idiomáticas: Moderadas
- Vocabulário técnico: Moderado

**Discursivos:**
- Conectores: Amplo repertório
- Marcadores discursivos: Moderados
- Referências anafóricas: Boas

### C1 (Avançado)

**Sintáticos:**
- Comprimento médio de sentença: 18-25 palavras
- Estruturas: Todas as estruturas, incluindo complexas
- Subordinação: Avançada (múltiplos tipos, encadeamento)
- Coordenação: Todos os tipos, incluindo sofisticados
- Tempos verbais: Todos os tempos, incluindo compostos avançados
- Voz passiva: Uso natural e variado
- Subjuntivo: Uso natural e variado
- Orações relativas: Todas, incluindo complexas

**Lexicais:**
- Vocabulário: 6000-8000 palavras
- Tipo/token ratio: 0.7-0.8
- Comprimento médio de palavras: 8-9 caracteres
- Expressões idiomáticas: Uso natural
- Vocabulário técnico: Amplo

**Discursivos:**
- Conectores: Repertório completo e sofisticado
- Marcadores discursivos: Uso natural e variado
- Referências anafóricas: Sofisticadas

### C2 (Proficiência)

**Sintáticos:**
- Comprimento médio de sentença: 25+ palavras
- Estruturas: Todas, incluindo muito complexas e raras
- Subordinação: Muito avançada (encadeamento complexo)
- Coordenação: Todos os tipos, uso sofisticado
- Tempos verbais: Todos, incluindo usos raros e literários
- Voz passiva: Uso sofisticado e variado
- Subjuntivo: Uso sofisticado e variado
- Orações relativas: Todas, incluindo estruturas raras

**Lexicais:**
- Vocabulário: 8000+ palavras
- Tipo/token ratio: 0.8-0.9
- Comprimento médio de palavras: 9+ caracteres
- Expressões idiomáticas: Uso sofisticado e criativo
- Vocabulário técnico: Muito amplo e especializado

**Discursivos:**
- Conectores: Repertório completo, uso criativo
- Marcadores discursivos: Uso sofisticado e criativo
- Referências anafóricas: Muito sofisticadas

## Métricas a Implementar

### 1. Métricas Sintáticas

```python
def calculate_syntactic_complexity(text):
    return {
        "avg_words_per_sentence": calculate_avg_words(text),
        "max_words_per_sentence": calculate_max_words(text),
        "subordination_ratio": count_subordinating_conjunctions(text) / sentence_count,
        "coordination_ratio": count_coordinating_conjunctions(text) / sentence_count,
        "passive_voice_count": count_passive_voice(text),
        "subjunctive_count": count_subjunctive(text),
        "relative_clauses_count": count_relative_clauses(text),
        "syntactic_depth": calculate_syntactic_depth(text),  # Profundidade de árvore
        "complex_sentence_ratio": count_complex_sentences(text) / sentence_count
    }
```

### 2. Métricas Lexicais

```python
def calculate_lexical_complexity(text):
    words = tokenize(text)
    unique_words = set(words)
    
    return {
        "type_token_ratio": len(unique_words) / len(words) if words else 0,
        "avg_word_length": sum(len(w) for w in words) / len(words) if words else 0,
        "lexical_diversity": len(unique_words),
        "high_frequency_ratio": count_high_frequency_words(words) / len(words),
        "low_frequency_ratio": count_low_frequency_words(words) / len(words),
        "idiomatic_expressions_count": count_idiomatic_expressions(text),
        "technical_vocabulary_ratio": count_technical_words(words) / len(words)
    }
```

### 3. Métricas Discursivas

```python
def calculate_discursive_complexity(text):
    return {
        "connector_density": count_connectors(text) / word_count,
        "discourse_markers_count": count_discourse_markers(text),
        "anaphoric_references_count": count_anaphoric_references(text),
        "cohesion_score": calculate_cohesion_score(text),
        "coherence_score": calculate_coherence_score(text)
    }
```

## Score de Complexidade por Nível

### Fórmula Proposta

```python
def calculate_cefr_complexity_score(syntactic, lexical, discursive):
    """
    Calcula score de complexidade (0-100) baseado nas métricas
    """
    # Pesos baseados nos papers
    syntactic_weight = 0.4
    lexical_weight = 0.35
    discursive_weight = 0.25
    
    # Normalizar métricas (0-1)
    syntactic_score = normalize_syntactic(syntactic)
    lexical_score = normalize_lexical(lexical)
    discursive_score = normalize_discursive(discursive)
    
    # Score final
    total_score = (
        syntactic_score * syntactic_weight +
        lexical_score * lexical_weight +
        discursive_score * discursive_weight
    ) * 100
    
    return round(total_score, 2)
```

### Thresholds por Nível

```python
CEFR_COMPLEXITY_THRESHOLDS = {
    "A1": (0, 15),
    "A2": (15, 30),
    "B1": (30, 50),
    "B2": (50, 70),
    "C1": (70, 85),
    "C2": (85, 100)
}
```

## Implementação no Sistema

### 1. Expandir `analyze_text_complexity_llm`

Adicionar métricas adicionais ao prompt do Sonnet 4.5:

```python
prompt = f"""Analise o seguinte texto em português e forneça um JSON com as seguintes informações:

Métricas Sintáticas:
- "avg_words_per_sentence": média de palavras por frase (float)
- "max_words_per_sentence": máximo de palavras em uma única frase (int)
- "subordination_ratio": proporção de orações subordinadas (float 0-1)
- "coordination_ratio": proporção de orações coordenadas (float 0-1)
- "passive_voice_count": número de estruturas de voz passiva (int)
- "subjunctive_count": número de usos de subjuntivo (int)
- "relative_clauses_count": número de orações relativas (int)

Métricas Lexicais:
- "type_token_ratio": razão tipo/token (float 0-1)
- "avg_word_length": comprimento médio de palavras (float)
- "lexical_diversity": número de palavras únicas (int)
- "idiomatic_expressions_count": número de expressões idiomáticas (int)

Métricas Discursivas:
- "connector_density": densidade de conectores (float)
- "discourse_markers_count": número de marcadores discursivos (int)

Score Final:
- "complexity_score": score de complexidade de 0 a 100 (int)

Texto: "{text}"
Retorne APENAS o JSON, sem texto adicional."""
```

### 2. Criar Funções de Validação

```python
def validate_cefr_level(text_complexity, target_level):
    """
    Valida se o texto está conforme o nível CEFR alvo
    """
    thresholds = CEFR_COMPLEXITY_THRESHOLDS[target_level]
    score = text_complexity["complexity_score"]
    
    is_compliant = thresholds[0] <= score <= thresholds[1]
    
    violations = []
    
    # Verificar métricas específicas por nível
    if target_level in ["A1", "A2"]:
        if text_complexity["subjunctive_count"] > 0:
            violations.append("Subjuntivo não permitido neste nível")
        if text_complexity["passive_voice_count"] > 0:
            violations.append("Voz passiva não permitida neste nível")
        if text_complexity["avg_words_per_sentence"] > 8:
            violations.append(f"Sentenças muito longas para {target_level}")
    
    return {
        "is_compliant": is_compliant and len(violations) == 0,
        "violations": violations,
        "score": score,
        "target_level": target_level
    }
```

## Próximos Passos

1. ✅ Documentar papers encontrados
2. ⏳ Implementar métricas sintáticas adicionais
3. ⏳ Implementar métricas lexicais
4. ⏳ Implementar métricas discursivas
5. ⏳ Integrar métricas na análise do Sonnet 4.5
6. ⏳ Validar com corpus anotado
7. ⏳ Ajustar pesos e thresholds baseado em validação

## Referências

Ver `CEFR_COMPLEXITY_MARKERS_PAPERS.md` para lista completa de papers com referências em formato APA.

