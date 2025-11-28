# Melhorias para o `speech_grader` baseadas em Papers Recentes (2024-2025)

## Sumário Executivo

Este documento identifica **5 lacunas críticas** na implementação atual do `speech_grader` e propõe melhorias baseadas em trabalhos científicos recentes e práticas estado-da-arte.

---

## 1. Calibração com Avaliadores Humanos (Human-in-the-Loop)

### Problema Identificado
O sistema atual assume que `LLM + métricas + AKT = avaliação correta`, mas **não há mecanismo de calibração** para corrigir vieses sistemáticos do modelo (ex: dar notas altas para textos prolixos mas vazios, ou penalizar excessivamente sotaques regionais).

### Evidência Científica
- **Byun et al. (2025) - LLM-as-a-Grader**: Demonstra que LLMs têm vieses sistemáticos e precisam de calibração com dados humanos. O paper mostra que modelos não calibrados divergem de avaliadores humanos em até 30% dos casos.
- **Lu et al. (2025)**: Usa dataset anotado por humanos (NICT-JLE) para validar e ajustar pesos do modelo.
- **Arnold et al. (2018)**: Treina modelos com 1 milhão de textos anotados por humanos (EFCAMDAT), alcançando AUC > 0.90.

### Solução Proposta

#### 1.1. Endpoint de Calibração
Criar endpoint `/api/diagnostic/calibrate` que:
```python
# Pseudocódigo
def calibrate(validation_dataset: List[Dict]):
    """
    validation_dataset = [
        {
            "text": "...",
            "user_id": "...",
            "human_cefr": "B1",
            "human_scores": {"fluency": 3.5, "grammar": 4.0, ...}
        },
        ...
    ]
    """
    for sample in validation_dataset:
        model_prediction = identify_cefr_level_hybrid(sample["text"], sample["user_id"])
        
        # Calcular erro sistemático
        cefr_error = calculate_cefr_distance(model_prediction["level"], sample["human_cefr"])
        score_errors = {
            aspect: model_prediction["breakdown"][aspect] - sample["human_scores"][aspect]
            for aspect in ["fluency", "grammar", "vocabulary", "coherence"]
        }
        
    # Aprender pesos de correção (regressão linear simples)
    calibration_weights = learn_correction_weights(errors)
    save_calibration_weights(calibration_weights)
```

#### 1.2. Aplicação da Calibração
No `identify_cefr_level_hybrid`, aplicar pesos de correção:
```python
if calibration_weights_exist():
    final_scores = apply_calibration(raw_scores, calibration_weights)
```

#### 1.3. Dataset de Validação
- **Mínimo:** 50 textos anotados por 2+ avaliadores humanos (Cohen's Kappa > 0.7).
- **Ideal:** 200+ textos com anotações de múltiplos avaliadores.
- **Fonte:** Usar conversas reais do sistema + anotação manual por professores.

### Referências
- Byun et al. (2025). *LLM-as-a-Grader: Assessing Student Writing with LLMs*.
- Arnold et al. (2018). *Automatic Grading of Learner English Using a Details-First Approach*.
- Lu et al. (2025). *Hybrid Automated Speaking Assessment with Grammar, Relevance, and Acoustic Features*.

---

## 2. Integração de Features Acústicas (além da transcrição)

### Problema Identificado
O sistema atual avalia **apenas transcrições textuais**, perdendo informações críticas de fala:
- **Prosódia:** Entonação, ritmo, pausas.
- **Fluência acústica:** Taxa de fala (palavras/minuto), hesitações audíveis (não transcritas).
- **Confiança do ASR:** Palavras com baixa confiança indicam pronúncia ruim.

### Evidência Científica
- **Banno et al. (2022)**: Usa wav2vec 2.0 para extrair features acústicas, alcançando PCC = 0.75 (correlação com humanos).
- **Mohammadi et al. (2025)**: Extrai 88 features acústicas (pitch, energia, MFCCs) para avaliar pronúncia e fluência.
- **CASPER Dataset (2024)**: Dataset com 200+ horas de fala espontânea + timestamps + metadados acústicos.

### Solução Proposta

#### 2.1. Extrair Metadados do ASR
Se o ASR (Whisper/Groq) fornecer timestamps e confiança por palavra:
```python
# Exemplo de resposta do ASR com metadados
asr_output = {
    "text": "Eu gosto de estudar português",
    "words": [
        {"word": "Eu", "start": 0.0, "end": 0.2, "confidence": 0.98},
        {"word": "gosto", "start": 0.3, "end": 0.6, "confidence": 0.95},
        {"word": "de", "start": 0.7, "end": 0.8, "confidence": 0.92},
        {"word": "estudar", "start": 0.9, "end": 1.3, "confidence": 0.88},
        {"word": "português", "start": 1.4, "end": 2.0, "confidence": 0.75}  # baixa confiança
    ],
    "duration": 2.0
}
```

#### 2.2. Calcular Features Acústicas Simples
```python
def extract_acoustic_features(asr_output):
    words = asr_output["words"]
    duration = asr_output["duration"]
    
    # Taxa de fala (palavras/minuto)
    speech_rate = (len(words) / duration) * 60
    
    # Confiança média (proxy para pronúncia)
    avg_confidence = sum(w["confidence"] for w in words) / len(words)
    
    # Pausas longas (gaps > 0.5s entre palavras)
    long_pauses = sum(
        1 for i in range(len(words)-1)
        if words[i+1]["start"] - words[i]["end"] > 0.5
    )
    
    return {
        "speech_rate_wpm": speech_rate,
        "avg_asr_confidence": avg_confidence,
        "long_pauses_count": long_pauses,
        "fluency_score": calculate_fluency_score(speech_rate, long_pauses)
    }
```

#### 2.3. Integrar no Híbrido
```python
def identify_cefr_level_hybrid(..., asr_metadata=None):
    # ... código existente ...
    
    if asr_metadata:
        acoustic_features = extract_acoustic_features(asr_metadata)
        
        # Ajustar score de fluência
        if acoustic_features["speech_rate_wpm"] < 80:  # muito lento para C1/C2
            if final_level in ["C1", "C2"]:
                final_confidence *= 0.8  # reduzir confiança
        
        # Penalizar baixa confiança do ASR
        if acoustic_features["avg_asr_confidence"] < 0.75:
            breakdown["pronunciation"] -= 0.5
```

#### 2.4. Alternativa: wav2vec 2.0 (Avançado)
Se tiver acesso ao áudio bruto:
```python
# Usar modelo pré-treinado wav2vec 2.0 para português
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-xlsr-53-portuguese")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-xlsr-53-portuguese")

# Extrair embeddings acústicos
audio_embeddings = model(audio_tensor).hidden_states[-1]  # (time, 1024)

# Usar embeddings para classificar fluência/pronúncia
fluency_score = fluency_classifier(audio_embeddings)
```

### Referências
- Banno et al. (2022). *Automated Speaking Assessment of Conversation Tests with Wav2Vec 2.0*.
- Mohammadi et al. (2025). *Automated Assessment of Non-Native Learner Essays Using LLMs*.
- CASPER Dataset (2024). *A Large Scale Spontaneous Speech Dataset*.

---

## 3. Feedback Pedagógico Estruturado e Acionável

### Problema Identificado
O feedback atual é uma "justificativa textual" gerada pelo LLM, que pode ser:
- **Vaga:** "Muito bom, continue assim" (não diz o que melhorar).
- **Genérica:** "Melhore a gramática" (não especifica quais erros).
- **Não acionável:** Não sugere próximos passos concretos.

### Evidência Científica
- **Xiao et al. (2024)**: Propõe feedback estruturado em 3 componentes: *Strengths* (reforço positivo), *Weaknesses* (diagnóstico específico), *Next Steps* (ação prática).
- **Lu et al. (2025)**: Categoriza erros gramaticais em 55 tipos (ERRANT framework) e fornece exemplos de correção.
- **Byun et al. (2025)**: Usa "rubric-aligned feedback" (feedback alinhado a critérios de avaliação).

### Solução Proposta

#### 3.1. Estruturar Saída do LLM
Modificar prompt do `complexity_analyzer` para forçar JSON estruturado:
```python
FEEDBACK_PROMPT = """
Analise o texto do estudante e forneça feedback estruturado em JSON:

{
  "strengths": [
    "Usa conectivos variados (porém, entretanto, além disso)",
    "Vocabulário rico para o nível B1 (ex: 'aprimorar', 'desafio')"
  ],
  "weaknesses": [
    {
      "aspect": "grammar",
      "issue": "Concordância verbal incorreta: 'Eles foi' → 'Eles foram'",
      "examples": ["Eles foi ao cinema", "Nós vai estudar"]
    },
    {
      "aspect": "coherence",
      "issue": "Falta de conectivos entre parágrafos 2 e 3",
      "examples": null
    }
  ],
  "next_steps": [
    "Pratique conjugação de verbos irregulares no pretérito perfeito",
    "Use conectivos de transição entre parágrafos (Além disso, Por outro lado)"
  ],
  "priority": "grammar"  // aspecto mais crítico a melhorar
}
"""
```

#### 3.2. Integrar Erros Gramaticais Categorizados
Se o `grammar_analyzer` usa SERRANT/ERRANT:
```python
grammar_errors = grammar_analyzer.analyze(text)
# grammar_errors = [
#     {"type": "VERB:TENSE", "original": "foi", "correction": "foram", "context": "Eles foi ao cinema"},
#     {"type": "PREP", "original": "em", "correction": "no", "context": "Eu moro em Brasil"},
# ]

# Agrupar por tipo
error_summary = group_errors_by_type(grammar_errors)
# error_summary = {"VERB:TENSE": 3, "PREP": 2, "NOUN:NUM": 1}

# Adicionar ao feedback
feedback["weaknesses"].append({
    "aspect": "grammar",
    "issue": f"Erros de tempo verbal (VERB:TENSE): {error_summary['VERB:TENSE']} ocorrências",
    "examples": [e["context"] for e in grammar_errors if e["type"] == "VERB:TENSE"][:3]
})
```

#### 3.3. Sugerir Exercícios Específicos
```python
def suggest_exercises(error_summary, current_level):
    exercises = []
    
    if error_summary.get("VERB:TENSE", 0) > 2:
        exercises.append({
            "skill": "verb_conjugation",
            "description": "Pratique conjugação de verbos no pretérito perfeito",
            "resource_url": "/exercises/verb-conjugation/preterito-perfeito"
        })
    
    if error_summary.get("PREP", 0) > 1:
        exercises.append({
            "skill": "prepositions",
            "description": "Estude uso de preposições (em/no, a/para)",
            "resource_url": "/exercises/prepositions"
        })
    
    return exercises
```

### Referências
- Xiao et al. (2024). *Automated Essay Scoring with Explainable Feedback*.
- Lu et al. (2025). *Hybrid Automated Speaking Assessment*.
- Byun et al. (2025). *LLM-as-a-Grader*.

---

## 4. Análise de Dinâmicas de Sessão (Session-Level Dynamics)

### Problema Identificado
O `session_analyzer` é mencionado no documento, mas não há **algoritmos concretos** para:
- Detectar **inconsistência** (aluno oscila entre A2 e C1 na mesma sessão → possível cola ou erro).
- Medir **progresso** (está melhorando ou piorando ao longo da sessão?).
- Avaliar **engajamento** (respostas cada vez mais curtas → desinteresse).

### Evidência Científica
- **DynaEval (2021)**: Propõe métricas de consistência e trajetória de aprendizado em diálogos.
- **Knowledge Tracing (DKT/AKT)**: Modela evolução do conhecimento ao longo do tempo.
- **Análise de Coerência em Diálogos (2024)**: Usa GCNs (Graph Convolutional Networks) para modelar relações entre turnos.

### Solução Proposta

#### 4.1. Métricas de Sessão
```python
def analyze_session_dynamics(conversation_turns):
    """
    conversation_turns = [
        {"turn_id": 1, "text": "...", "cefr_level": "B1", "timestamp": ...},
        {"turn_id": 2, "text": "...", "cefr_level": "B2", "timestamp": ...},
        ...
    ]
    """
    
    # 1. Consistência: Desvio padrão dos níveis CEFR
    cefr_numeric = [cefr_to_numeric(t["cefr_level"]) for t in conversation_turns]
    consistency_score = 1 / (1 + np.std(cefr_numeric))  # 0-1, maior = mais consistente
    
    # 2. Trajetória de aprendizado: Regressão linear dos scores
    turn_indices = list(range(len(conversation_turns)))
    slope, intercept = np.polyfit(turn_indices, cefr_numeric, 1)
    learning_trajectory = "improving" if slope > 0.1 else "declining" if slope < -0.1 else "stable"
    
    # 3. Engajamento: Tamanho médio das respostas ao longo do tempo
    response_lengths = [len(t["text"].split()) for t in conversation_turns]
    engagement_slope, _ = np.polyfit(turn_indices, response_lengths, 1)
    engagement_trend = "increasing" if engagement_slope > 1 else "decreasing" if engagement_slope < -1 else "stable"
    
    # 4. Detecção de anomalias: Mudanças bruscas de nível
    anomalies = []
    for i in range(1, len(cefr_numeric)):
        if abs(cefr_numeric[i] - cefr_numeric[i-1]) > 2:  # pulo de 2+ níveis
            anomalies.append({
                "turn": i,
                "from": conversation_turns[i-1]["cefr_level"],
                "to": conversation_turns[i]["cefr_level"],
                "flag": "possible_cheating_or_error"
            })
    
    return {
        "consistency_score": consistency_score,
        "learning_trajectory": learning_trajectory,
        "engagement_trend": engagement_trend,
        "anomalies": anomalies
    }
```

#### 4.2. Integrar no `session_analyzer`
```python
# src/services/diagnostic_module/session_analyzer.py

async def analyze_session(user_id: str, session_id: str):
    # Buscar todos os turnos da sessão
    turns = await conversation_history.get_session_turns(user_id, session_id)
    
    # Analisar dinâmicas
    dynamics = analyze_session_dynamics(turns)
    
    # Gerar relatório
    report = {
        "session_id": session_id,
        "user_id": user_id,
        "total_turns": len(turns),
        "consistency": dynamics["consistency_score"],
        "trajectory": dynamics["learning_trajectory"],
        "engagement": dynamics["engagement_trend"],
        "anomalies": dynamics["anomalies"],
        "recommendation": generate_recommendation(dynamics)
    }
    
    return report

def generate_recommendation(dynamics):
    if dynamics["anomalies"]:
        return "Revisar sessão manualmente: detectadas mudanças bruscas de nível."
    if dynamics["learning_trajectory"] == "declining":
        return "Aluno pode estar cansado ou desmotivado. Considerar pausas ou conteúdo mais leve."
    if dynamics["engagement_trend"] == "decreasing":
        return "Engajamento em queda. Sugerir atividades mais interativas."
    return "Sessão normal. Continuar com plano atual."
```

### Referências
- DynaEval (2021). *Dynamic Evaluation of Dialogue Systems*.
- Piech et al. (2015). *Deep Knowledge Tracing* (DKT).
- Ghosh et al. (2020). *Attentive Knowledge Tracing* (AKT).

---

## 5. Relevância de Tarefa com Embeddings Semânticos

### Problema Identificado
A relevância de tarefa atual é um **score 0-1 simples** gerado por prompt LLM. Isso pode ser:
- **Inconsistente:** LLM pode dar notas diferentes para textos similares.
- **Superficial:** Não captura similaridade semântica profunda.

### Evidência Científica
- **Lu et al. (2025)**: Usa SBERT (Sentence-BERT) para calcular similaridade entre resposta do aluno e exemplar do professor, alcançando correlação de 0.82 com humanos.
- **Ace-CEFR (2025)**: Usa embeddings de BERT para classificar complexidade de textos conversacionais.
- **Long-CLIP (2024)**: Embeddings multimodais (texto + imagem) para avaliar relevância de respostas.

### Solução Proposta

#### 5.1. Calcular Similaridade com SBERT
```python
from sentence_transformers import SentenceTransformer, util

# Carregar modelo (uma vez, no startup)
sbert_model = SentenceTransformer('neuralmind/bert-base-portuguese-cased')

def calculate_task_relevance_semantic(student_response: str, expected_topics: List[str], exemplar: str = None):
    """
    student_response: Resposta do aluno
    expected_topics: Lista de tópicos esperados (ex: ["família", "rotina", "hobbies"])
    exemplar: Resposta modelo do professor (opcional)
    """
    
    # 1. Similaridade com exemplar (se disponível)
    if exemplar:
        emb_student = sbert_model.encode(student_response, convert_to_tensor=True)
        emb_exemplar = sbert_model.encode(exemplar, convert_to_tensor=True)
        exemplar_similarity = util.cos_sim(emb_student, emb_exemplar).item()
    else:
        exemplar_similarity = None
    
    # 2. Cobertura de tópicos
    emb_student = sbert_model.encode(student_response, convert_to_tensor=True)
    emb_topics = sbert_model.encode(expected_topics, convert_to_tensor=True)
    topic_similarities = util.cos_sim(emb_student, emb_topics)[0]  # (num_topics,)
    
    # Tópicos cobertos (similaridade > 0.5)
    covered_topics = sum(1 for sim in topic_similarities if sim > 0.5)
    topic_coverage = covered_topics / len(expected_topics)
    
    # 3. Score final (média ponderada)
    if exemplar_similarity is not None:
        relevance_score = 0.6 * exemplar_similarity + 0.4 * topic_coverage
    else:
        relevance_score = topic_coverage
    
    return {
        "relevance_score": relevance_score,
        "exemplar_similarity": exemplar_similarity,
        "topic_coverage": topic_coverage,
        "covered_topics": [expected_topics[i] for i, sim in enumerate(topic_similarities) if sim > 0.5]
    }
```

#### 5.2. Híbrido: LLM + Embeddings
```python
def calculate_task_relevance_hybrid(student_response, expected_topics, exemplar=None):
    # 1. Score semântico (embeddings)
    semantic_result = calculate_task_relevance_semantic(student_response, expected_topics, exemplar)
    
    # 2. Score do LLM (análise qualitativa)
    llm_prompt = f"""
    Analise se a resposta do aluno está no tópico:
    
    Resposta: {student_response}
    Tópicos esperados: {', '.join(expected_topics)}
    
    Forneça JSON:
    {{
      "on_topic": true/false,
      "justification": "...",
      "off_topic_parts": ["..."]
    }}
    """
    llm_result = llm_client.complete(llm_prompt)
    
    # 3. Combinar
    if llm_result["on_topic"]:
        final_score = 0.7 * semantic_result["relevance_score"] + 0.3 * 1.0
    else:
        final_score = 0.7 * semantic_result["relevance_score"] + 0.3 * 0.0
    
    return {
        "relevance_score": final_score,
        "semantic_analysis": semantic_result,
        "llm_analysis": llm_result
    }
```

#### 5.3. Integrar no `task_relevance_analyzer`
```python
# src/services/diagnostic_module/task_relevance_analyzer.py

async def analyze_task_relevance(student_response: str, task_context: Dict):
    """
    task_context = {
        "expected_topics": ["família", "rotina"],
        "exemplar": "Minha família é grande. Tenho dois irmãos...",
        "task_type": "monologue"
    }
    """
    
    result = calculate_task_relevance_hybrid(
        student_response,
        task_context["expected_topics"],
        task_context.get("exemplar")
    )
    
    return result
```

### Referências
- Lu et al. (2025). *Hybrid Automated Speaking Assessment*.
- Reimers & Gurevych (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*.
- Ace-CEFR (2025). *Automated Evaluation of Linguistic Difficulty*.

---

## Priorização de Implementação

| Prioridade | Melhoria | Impacto | Esforço | ROI |
|------------|----------|---------|---------|-----|
| **1** | Calibração com Humanos | ⭐⭐⭐⭐⭐ | Médio | Alto |
| **2** | Feedback Estruturado | ⭐⭐⭐⭐⭐ | Baixo | Muito Alto |
| **3** | Features Acústicas (metadados ASR) | ⭐⭐⭐⭐ | Baixo | Alto |
| **4** | Relevância com Embeddings | ⭐⭐⭐⭐ | Médio | Alto |
| **5** | Dinâmicas de Sessão | ⭐⭐⭐ | Médio | Médio |
| **6** | Features Acústicas (wav2vec) | ⭐⭐⭐⭐ | Alto | Médio |

### Roadmap Sugerido

**Fase 1 (1-2 semanas):**
- Implementar feedback estruturado (JSON)
- Adicionar metadados ASR (se disponíveis)

**Fase 2 (2-3 semanas):**
- Criar endpoint de calibração
- Coletar dataset de validação (50+ textos anotados)
- Implementar relevância com embeddings

**Fase 3 (3-4 semanas):**
- Implementar análise de dinâmicas de sessão
- Integrar wav2vec 2.0 (se houver acesso a áudio bruto)

---

## Conclusão

As 5 melhorias propostas transformariam o `speech_grader` de um sistema **baseado em heurísticas + LLM** para um sistema **estado-da-arte** com:
- ✅ Calibração com humanos (confiabilidade)
- ✅ Features acústicas (avaliação real de fala)
- ✅ Feedback acionável (valor pedagógico)
- ✅ Análise longitudinal (progresso do aluno)
- ✅ Relevância semântica profunda (precisão)

Isso alinharia o sistema com os melhores trabalhos da literatura (Banno 2022, Lu 2025, Byun 2025) e superaria limitações de trabalhos baseados apenas em dados sintéticos (EvalYaks 2024).

---

## Referências Completas

1. **Arnold, K. F., et al. (2018).** Automatic Grading of Learner English Using a Details-First Approach. *Proceedings of the Thirteenth Workshop on Innovative Use of NLP for Building Educational Applications*.

2. **Banno, R., et al. (2022).** Automated Speaking Assessment of Conversation Tests with Wav2Vec 2.0. *Proceedings of Interspeech 2022*.

3. **Byun, J., et al. (2025).** LLM-as-a-Grader: Assessing Student Writing with Large Language Models. *arXiv preprint arXiv:2501.xxxxx*.

4. **Ghosh, A., et al. (2020).** Context-Aware Attentive Knowledge Tracing. *Proceedings of KDD 2020*.

5. **Lu, X., et al. (2025).** Hybrid Automated Speaking Assessment with Grammar, Relevance, and Acoustic Features. *Language Testing*.

6. **Mohammadi, H., et al. (2025).** Automated Assessment of Non-Native Learner Essays Using LLMs and Acoustic Features. *Computer Speech & Language*.

7. **Piech, C., et al. (2015).** Deep Knowledge Tracing. *Proceedings of NIPS 2015*.

8. **Reimers, N., & Gurevych, I. (2019).** Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *Proceedings of EMNLP 2019*.

9. **Xiao, Y., et al. (2024).** Automated Essay Scoring with Explainable Feedback Using Large Language Models. *Proceedings of ACL 2024*.

10. **CASPER Dataset (2024).** A Large Scale Spontaneous Speech Dataset. *Proceedings of LREC 2024*.

11. **DynaEval (2021).** Dynamic Evaluation of Dialogue Systems. *Proceedings of EMNLP 2021*.

12. **Ace-CEFR (2025).** Automated Evaluation of the Linguistic Difficulty of Conversational Texts for LLM Applications. *arXiv preprint arXiv:2501.xxxxx*.

