# Análise dos Papers Baixados e Convertidos

## Sumário Executivo

Este documento apresenta uma análise detalhada de **5 papers** recém-baixados e convertidos para Markdown, extraindo insights práticos para melhorar o `speech_grader`.

---

## 1. Reimers & Gurevych (2019) - Sentence-BERT

### 📄 Informações Básicas
- **Arquivo:** `Reimers_Gurevych_2019_Sentence_BERT.md` (49KB)
- **Título:** *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*
- **Autores:** Nils Reimers, Iryna Gurevych (Technische Universität Darmstadt)
- **Venue:** EMNLP 2019
- **Código:** [https://github.com/UKPLab/sentence-transformers](https://github.com/UKPLab/sentence-transformers)

### 🎯 Problema Resolvido
BERT original é **computacionalmente proibitivo** para tarefas de similaridade semântica:
- Encontrar o par mais similar em 10.000 sentenças requer **50 milhões de inferências** (~65 horas em V100 GPU)
- Inviável para clustering, semantic search e comparações em larga escala

### 💡 Solução Proposta
**Sentence-BERT (SBERT)**: Modificação do BERT usando redes siamesas e triplet networks para gerar embeddings de sentenças semanticamente significativos.

**Arquitetura:**
```
Input Sentence A → BERT → Pooling → u (embedding fixo)
Input Sentence B → BERT → Pooling → v (embedding fixo)

Similarity = cosine_similarity(u, v)
```

**Pooling Strategies:**
1. **MEAN** (média dos tokens) - melhor performance
2. **MAX** (max pooling)
3. **CLS** (token [CLS])

### 📊 Resultados
- **Speedup:** De 65 horas para **~5 segundos** (10.000 sentenças)
- **Accuracy:** Mantém ou supera BERT em STS tasks
- **Spearman Correlation:** 0.85 em STS benchmark

### 🔧 Aplicação no `speech_grader`

#### Implementação Direta
```python
from sentence_transformers import SentenceTransformer

# Carregar modelo pré-treinado para português
model = SentenceTransformer('neuralmind/bert-base-portuguese-cased')

def calculate_semantic_similarity(text1: str, text2: str) -> float:
    """
    Calcula similaridade semântica entre dois textos.
    """
    embeddings = model.encode([text1, text2])
    similarity = util.cos_sim(embeddings[0], embeddings[1])
    return similarity.item()
```

#### Casos de Uso no `speech_grader`

**1. Relevância de Tarefa (Task Relevance)**
```python
def assess_task_relevance_with_sbert(
    student_response: str,
    expected_topics: List[str],
    exemplar: str = None
) -> Dict:
    """
    Avalia relevância usando SBERT.
    """
    # Similaridade com exemplar
    if exemplar:
        exemplar_sim = calculate_semantic_similarity(student_response, exemplar)
    else:
        exemplar_sim = None
    
    # Cobertura de tópicos
    topic_embeddings = model.encode(expected_topics)
    response_embedding = model.encode(student_response)
    
    topic_similarities = [
        util.cos_sim(response_embedding, topic_emb).item()
        for topic_emb in topic_embeddings
    ]
    
    # Tópicos cobertos (threshold = 0.5)
    covered_topics = sum(1 for sim in topic_similarities if sim > 0.5)
    topic_coverage = covered_topics / len(expected_topics)
    
    return {
        "exemplar_similarity": exemplar_sim,
        "topic_coverage": topic_coverage,
        "topic_similarities": topic_similarities
    }
```

**2. Detecção de Off-Topic (Fora do Tema)**
```python
def detect_off_topic(student_response: str, task_description: str) -> bool:
    """
    Detecta se resposta está fora do tema.
    """
    similarity = calculate_semantic_similarity(student_response, task_description)
    return similarity < 0.3  # threshold ajustável
```

**3. Clustering de Respostas Similares**
```python
from sklearn.cluster import KMeans

def cluster_student_responses(responses: List[str], n_clusters: int = 5):
    """
    Agrupa respostas similares para análise de padrões.
    """
    embeddings = model.encode(responses)
    clusters = KMeans(n_clusters=n_clusters).fit_predict(embeddings)
    return clusters
```

### ✅ Vantagens para o `speech_grader`
1. **Velocidade:** 1000x mais rápido que BERT para similaridade
2. **Escalabilidade:** Pode processar milhares de respostas em segundos
3. **Consistência:** Métricas numéricas objetivas (vs. prompts LLM variáveis)
4. **Multilíngue:** Modelos pré-treinados para português disponíveis

### 📦 Modelos Recomendados para Português
1. **`neuralmind/bert-base-portuguese-cased`** - BERT português
2. **`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`** - Multilíngue, rápido
3. **`rufimelo/Legal-BERTimbau-base`** - Português brasileiro, domínio geral

---

## 2. Piech et al. (2015) - Deep Knowledge Tracing (DKT)

### 📄 Informações Básicas
- **Arquivo:** `Piech_2015_Deep_Knowledge_Tracing.md` (41KB)
- **Título:** *Deep Knowledge Tracing*
- **Autores:** Chris Piech et al. (Stanford, Khan Academy, Google)
- **Venue:** NIPS 2015
- **Impacto:** Base para AKT (usado no nosso `student_model`)

### 🎯 Problema Resolvido
Modelos tradicionais de Knowledge Tracing (KT) usam:
- **Bayesian Knowledge Tracing (BKT):** Markov de primeira ordem, representação limitada
- **Requer anotação manual de conceitos** por especialistas
- **Não captura relações complexas** entre skills

### 💡 Solução Proposta
**Deep Knowledge Tracing (DKT)**: Usa RNNs (LSTMs) para modelar conhecimento do aluno ao longo do tempo.

**Arquitetura:**
```
Input: x_t = (exercise_id, correct/incorrect)
       ↓
    [LSTM] → hidden state h_t (representa conhecimento latente)
       ↓
    [Softmax] → P(correct | exercise_id, h_t)
```

**Encoding de Input:**
- One-hot encoding: `[exercise_1_correct, exercise_1_incorrect, ..., exercise_N_correct, exercise_N_incorrect]`
- Dimensão: `2 * num_exercises`

### 📊 Resultados
- **AUC:** 0.86 (vs. 0.67 do BKT) - **+25% improvement**
- **Dataset:** Khan Academy (1.5M interactions, 564K students)
- **Não requer anotação de conceitos** por especialistas

### 🔧 Relação com o `speech_grader`

#### Contexto Atual
Nosso sistema já usa **AKT (Attentive Knowledge Tracing)**, que é uma evolução do DKT com mecanismo de atenção. O DKT é a **base teórica** do AKT.

#### Como DKT/AKT se Integra ao `speech_grader`

**1. Validação Cruzada (já implementado - "Opção 1")**
```python
# Em cefr_level_analyzer.py
async def identify_cefr_level_hybrid(..., user_id: str):
    # ... classificação do texto ...
    
    # Buscar progresso AKT
    akt_progress = await get_akt_cefr_progress(user_id, session)
    
    # Ajustar confiança baseado em convergência
    validation_result = validate_with_akt(
        text_cefr_level=final_level,
        text_confidence=final_confidence,
        akt_progress=akt_progress
    )
    
    final_confidence = validation_result["adjusted_confidence"]
```

**2. Predição de Desempenho Futuro (não implementado)**
```python
def predict_future_performance(user_id: str, target_skill: str) -> float:
    """
    Usa AKT para prever probabilidade de sucesso em skill futura.
    """
    # Buscar histórico do aluno
    interaction_history = get_student_interactions(user_id)
    
    # Prever com AKT
    prediction = akt_model.predict(interaction_history, target_skill)
    
    return prediction  # P(correct | skill, history)
```

**3. Recomendação de Exercícios (não implementado)**
```python
def recommend_next_exercise(user_id: str) -> str:
    """
    Recomenda próximo exercício baseado em AKT.
    """
    all_skills = get_all_skills()
    
    # Calcular probabilidade de sucesso para cada skill
    predictions = {
        skill: predict_future_performance(user_id, skill)
        for skill in all_skills
    }
    
    # Recomendar skill com P(correct) ~ 0.7 (zona de desenvolvimento proximal)
    optimal_skill = min(predictions, key=lambda s: abs(predictions[s] - 0.7))
    
    return optimal_skill
```

### 📚 Insights do Paper

**1. Representação Latente Rica**
- DKT aprende representações de conhecimento de **200-500 dimensões**
- Captura relações complexas entre skills (ex: "álgebra" ajuda em "geometria")

**2. Não Requer Anotação Manual**
- DKT pode aprender estrutura de conceitos **autonomamente** dos dados
- Útil quando não há taxonomia de skills pré-definida

**3. Descoberta de Influência de Exercícios**
- DKT pode identificar quais exercícios mais contribuem para aprendizado
- Útil para design de currículo

### ✅ Aplicação no `speech_grader`
1. **Já usamos AKT** (evolução do DKT) no `student_model`
2. **Validação cruzada** entre CEFR do texto e CEFR do AKT (implementado)
3. **Oportunidade:** Usar AKT para **recomendar tópicos** de conversação baseados em zona de desenvolvimento proximal

---

## 3. RUBER (2017) - Dialog Evaluation

### 📄 Informações Básicas
- **Arquivo:** `RUBER_2017_Dialog_Evaluation.md` (42KB)
- **Título:** *RUBER: An Unsupervised Method for Automatic Evaluation of Open-Domain Dialog Systems*
- **Autores:** Chongyang Tao et al. (Peking University)
- **Venue:** AAAI 2017

### 🎯 Problema Resolvido
Avaliação de sistemas de diálogo é **cara e demorada**:
- Métricas tradicionais (BLEU, METEOR) têm **baixa correlação** com humanos em diálogo
- Anotação humana é **time-consuming** e **expensive**

### 💡 Solução Proposta
**RUBER**: Combina duas métricas:

**1. Referenced Metric (com referência)**
- Mede similaridade entre resposta gerada e groundtruth
- Usa **pooling de word embeddings** (não word-overlap)

**2. Unreferenced Metric (sem referência)**
- Mede **relevância entre query e resposta**
- Treinada com **negative sampling** (sem anotação humana)

**Fórmula Final:**
```
RUBER_score = min(referenced_score, unreferenced_score)
```

### 📊 Resultados
- **Correlação com humanos:** 0.53 (Spearman)
- **Sem necessidade de anotação humana** para treinar
- **Flexível:** Extensível a diferentes datasets e línguas

### 🔧 Aplicação no `speech_grader`

#### Caso de Uso: Avaliação de Coerência em Diálogo

**1. Referenced Metric (Similaridade com Exemplar)**
```python
def referenced_metric_ruber(generated_response: str, groundtruth: str) -> float:
    """
    Mede similaridade entre resposta gerada e groundtruth.
    """
    # Usar SBERT (do paper anterior)
    similarity = calculate_semantic_similarity(generated_response, groundtruth)
    return similarity
```

**2. Unreferenced Metric (Relevância Query-Response)**
```python
import torch
import torch.nn as nn

class QueryResponseRelevance(nn.Module):
    def __init__(self, embedding_dim=768):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim * 2, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, query_emb, response_emb):
        # Concatenar embeddings
        combined = torch.cat([query_emb, response_emb], dim=-1)
        x = self.relu(self.fc1(combined))
        x = self.relu(self.fc2(x))
        relevance = self.sigmoid(self.fc3(x))
        return relevance

def unreferenced_metric_ruber(query: str, response: str, model) -> float:
    """
    Mede relevância entre query e response.
    """
    # Obter embeddings
    query_emb = sbert_model.encode(query)
    response_emb = sbert_model.encode(response)
    
    # Calcular relevância
    with torch.no_grad():
        relevance = model(
            torch.tensor(query_emb).unsqueeze(0),
            torch.tensor(response_emb).unsqueeze(0)
        )
    
    return relevance.item()
```

**3. RUBER Final**
```python
def ruber_score(query: str, generated_response: str, groundtruth: str, model) -> float:
    """
    Combina referenced e unreferenced metrics.
    """
    ref_score = referenced_metric_ruber(generated_response, groundtruth)
    unref_score = unreferenced_metric_ruber(query, generated_response, model)
    
    # RUBER usa min para penalizar qualquer aspecto ruim
    return min(ref_score, unref_score)
```

### ✅ Aplicação no `speech_grader`
1. **Coerência de diálogo:** Avaliar se resposta do aluno é relevante à pergunta
2. **Sem anotação humana:** Treinar com negative sampling
3. **Complementar ao LLM:** Métrica objetiva para validar julgamento do LLM

---

## 4. Yeh et al. (2021) - Comprehensive Assessment of Dialog Metrics

### 📄 Informações Básicas
- **Arquivo:** `Comprehensive_Assessment_Dialog_Metrics_2021.md` (650KB - maior arquivo)
- **Título:** *A Comprehensive Assessment of Dialog Evaluation Metrics*
- **Autores:** Yi-Ting Yeh, Maxine Eskenazi, Shikib Mehri (CMU)
- **Venue:** EACL 2021

### 🎯 Problema Resolvido
**Falta de comparação sistemática** entre métricas de avaliação de diálogo:
- 23 métricas propostas recentemente
- Avaliadas em datasets diferentes
- Sem consenso sobre qual usar

### 💡 Contribuição
**Avaliação abrangente** de **23 métricas** em **10 datasets**, analisando:
1. **Turn-level vs. Dialog-level**
2. **Diferentes comprimentos de diálogo**
3. **Diferentes qualidades** (coerência, engajamento)
4. **Diferentes tipos de modelos** (generative, retrieval, simple, SOTA)
5. **Similaridade entre métricas**
6. **Combinações de métricas**

### 📊 Principais Resultados

**Métricas com Melhor Correlação com Humanos:**
1. **USR** (Untrained Supervised Relevance) - 0.42 Spearman
2. **GRADE** (Graph-based Automatic Dialogue Evaluator) - 0.40
3. **DynaEval** - 0.38
4. **RUBER** - 0.35

**Métricas Tradicionais (Ruins):**
- **BLEU** - 0.12 (muito baixo!)
- **METEOR** - 0.15
- **ROUGE** - 0.14

### 🔧 Insights para o `speech_grader`

#### 1. **Não usar BLEU/METEOR/ROUGE para diálogo**
```python
# ❌ NÃO FAZER ISSO:
def evaluate_dialog_quality(response, groundtruth):
    bleu_score = calculate_bleu(response, groundtruth)  # Correlação ruim!
    return bleu_score

# ✅ FAZER ISSO:
def evaluate_dialog_quality(query, response, groundtruth):
    # Usar métrica específica de diálogo
    ruber = ruber_score(query, response, groundtruth, model)
    return ruber
```

#### 2. **Combinar múltiplas métricas**
O paper mostra que **combinações de métricas** melhoram correlação:
```python
def combined_dialog_metric(
    query: str,
    response: str,
    groundtruth: str,
    history: List[str]
) -> Dict[str, float]:
    """
    Combina múltiplas métricas para avaliação robusta.
    """
    # Relevância (RUBER-style)
    relevance = unreferenced_metric_ruber(query, response, model)
    
    # Similaridade semântica (SBERT)
    similarity = calculate_semantic_similarity(response, groundtruth)
    
    # Coerência com histórico (DynaEval-style)
    coherence = calculate_coherence_with_history(response, history)
    
    # Engajamento (comprimento, diversidade)
    engagement = calculate_engagement_score(response)
    
    # Score final (média ponderada)
    final_score = (
        0.3 * relevance +
        0.3 * similarity +
        0.2 * coherence +
        0.2 * engagement
    )
    
    return {
        "final_score": final_score,
        "relevance": relevance,
        "similarity": similarity,
        "coherence": coherence,
        "engagement": engagement
    }
```

#### 3. **Avaliar em nível de turno E diálogo**
```python
# Turn-level (cada resposta individual)
turn_scores = [
    evaluate_turn(query, response, groundtruth)
    for query, response, groundtruth in conversation
]

# Dialog-level (conversa inteira)
dialog_score = {
    "avg_turn_score": np.mean(turn_scores),
    "consistency": np.std(turn_scores),  # baixo = mais consistente
    "trajectory": calculate_trajectory(turn_scores)  # melhorando/piorando
}
```

### ✅ Takeaways para o `speech_grader`
1. **Não usar BLEU/METEOR** para avaliar diálogo
2. **Combinar múltiplas métricas** (relevância + similaridade + coerência)
3. **Avaliar em dois níveis:** turn-level e dialog-level
4. **Usar métricas reference-free** (baseadas em contexto, não groundtruth)

---

## 5. Mekyska et al. (2022) - Pathological Speech Analysis

### 📄 Informações Básicas
- **Arquivo:** `Pathological_Speech_Analysis_2022.md` (118KB)
- **Título:** *Robust and Complex Approach of Pathological Speech Signal Analysis*
- **Autores:** Jiri Mekyska et al. (Multi-institucional: República Tcheca, Espanha)
- **Venue:** Neurocomputing 2022

### 🎯 Problema Resolvido
Análise de fala patológica (disartria, Parkinson, etc.) requer:
- **Múltiplas features acústicas** (não apenas transcrição)
- **Robustez** a variações individuais
- **Interpretabilidade clínica**

### 💡 Contribuição
**92 features de fala**, incluindo **36 novas**:
1. **Modulation spectra**
2. **Inferior colliculus coefficients**
3. **Bicepstrum**
4. **Sample and approximate entropy**
5. **Empirical mode decomposition (EMD)**

### 📊 Resultados
- **Acurácia:** 100% em MEEI database (mas com limitações)
- **Acurácia:** 82.1% em PdA Hospital database
- **Especificidade:** 83.8%
- **Features mais discriminativas:** Cepstral peak prominence (CPP)

### 🔧 Aplicação no `speech_grader`

#### Contexto
Este paper é sobre **fala patológica** (distúrbios), não sobre **proficiência linguística**. No entanto, algumas features são úteis para avaliar **qualidade de fala** em geral.

#### Features Úteis para o `speech_grader`

**1. Cepstral Peak Prominence (CPP)**
- Mede **qualidade vocal** (hoarseness, breathiness)
- Útil para avaliar **pronúncia** e **fluência**

**2. Jitter e Shimmer**
- Variação de **pitch** e **amplitude**
- Indicam **estabilidade vocal**

**3. Harmonic-to-Noise Ratio (HNR)**
- Razão entre componentes harmônicos e ruído
- Indica **clareza de pronúncia**

#### Implementação (se houver acesso a áudio)
```python
import parselmouth  # Praat Python library

def extract_acoustic_features(audio_path: str) -> Dict[str, float]:
    """
    Extrai features acústicas de áudio.
    """
    sound = parselmouth.Sound(audio_path)
    
    # Pitch
    pitch = sound.to_pitch()
    mean_pitch = parselmouth.praat.call(pitch, "Get mean", 0, 0, "Hertz")
    
    # Jitter (variação de pitch)
    point_process = parselmouth.praat.call(sound, "To PointProcess (periodic, cc)", 75, 600)
    jitter = parselmouth.praat.call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    
    # Shimmer (variação de amplitude)
    shimmer = parselmouth.praat.call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    
    # HNR (Harmonic-to-Noise Ratio)
    harmonicity = sound.to_harmonicity()
    hnr = parselmouth.praat.call(harmonicity, "Get mean", 0, 0)
    
    return {
        "mean_pitch_hz": mean_pitch,
        "jitter": jitter,
        "shimmer": shimmer,
        "hnr_db": hnr
    }
```

### ⚠️ Limitações para o `speech_grader`
1. **Foco em patologia:** Paper é sobre distúrbios, não proficiência
2. **Requer áudio bruto:** Não funciona com transcrições
3. **Complexidade:** 92 features podem ser overkill para nosso caso

### ✅ Aplicação Prática
- **Se houver acesso a áudio:** Extrair CPP, HNR para avaliar **qualidade de pronúncia**
- **Se apenas transcrição:** Focar em outros papers (SBERT, RUBER, DKT)

---

## Resumo Consolidado: Prioridades de Implementação

### 🥇 Alta Prioridade (Implementar Agora)

#### 1. **SBERT para Relevância de Tarefa**
- **Paper:** Reimers & Gurevych (2019)
- **Esforço:** Baixo (biblioteca pronta)
- **Impacto:** Alto (relevância objetiva e rápida)
- **Implementação:**
  ```python
  from sentence_transformers import SentenceTransformer
  model = SentenceTransformer('neuralmind/bert-base-portuguese-cased')
  ```

#### 2. **Combinar Múltiplas Métricas de Diálogo**
- **Paper:** Yeh et al. (2021)
- **Esforço:** Médio
- **Impacto:** Alto (avaliação mais robusta)
- **Implementação:** Combinar relevância + similaridade + coerência

### 🥈 Média Prioridade (Próximas Sprints)

#### 3. **RUBER para Coerência Query-Response**
- **Paper:** Tao et al. (2017)
- **Esforço:** Médio (treinar modelo unreferenced)
- **Impacto:** Médio (complementa LLM)

#### 4. **Análise Turn-level e Dialog-level**
- **Paper:** Yeh et al. (2021)
- **Esforço:** Baixo (agregar scores existentes)
- **Impacto:** Médio (insights longitudinais)

### 🥉 Baixa Prioridade (Futuro)

#### 5. **Features Acústicas (CPP, HNR)**
- **Paper:** Mekyska et al. (2022)
- **Esforço:** Alto (requer áudio bruto + Praat)
- **Impacto:** Médio (útil se houver áudio)

#### 6. **Recomendação de Exercícios com AKT**
- **Paper:** Piech et al. (2015)
- **Esforço:** Alto (integração complexa)
- **Impacto:** Alto (mas fora do escopo do `speech_grader`)

---

## Código de Exemplo: Integração Completa

```python
# src/services/diagnostic_module/analyzers/dialog_evaluator.py

from sentence_transformers import SentenceTransformer, util
import torch
import torch.nn as nn
from typing import Dict, List

class DialogEvaluator:
    """
    Avaliador de diálogo combinando insights de múltiplos papers.
    """
    
    def __init__(self):
        # SBERT para embeddings (Reimers & Gurevych 2019)
        self.sbert = SentenceTransformer('neuralmind/bert-base-portuguese-cased')
        
        # Modelo unreferenced (RUBER 2017)
        self.relevance_model = self._load_relevance_model()
    
    def evaluate_turn(
        self,
        query: str,
        response: str,
        groundtruth: str = None,
        expected_topics: List[str] = None
    ) -> Dict[str, float]:
        """
        Avalia um turno de diálogo (turn-level).
        """
        scores = {}
        
        # 1. Relevância query-response (RUBER unreferenced)
        scores["relevance"] = self._calculate_relevance(query, response)
        
        # 2. Similaridade com groundtruth (RUBER referenced + SBERT)
        if groundtruth:
            scores["similarity"] = self._calculate_similarity(response, groundtruth)
        
        # 3. Cobertura de tópicos (SBERT)
        if expected_topics:
            scores["topic_coverage"] = self._calculate_topic_coverage(response, expected_topics)
        
        # 4. Score final (Yeh et al. 2021 - combinar métricas)
        scores["final_score"] = self._combine_scores(scores)
        
        return scores
    
    def evaluate_dialog(
        self,
        conversation: List[Dict[str, str]]
    ) -> Dict[str, float]:
        """
        Avalia conversa inteira (dialog-level).
        """
        # Avaliar cada turno
        turn_scores = [
            self.evaluate_turn(
                query=turn["query"],
                response=turn["response"],
                groundtruth=turn.get("groundtruth")
            )
            for turn in conversation
        ]
        
        # Agregar scores (Yeh et al. 2021)
        final_scores = [s["final_score"] for s in turn_scores]
        
        return {
            "avg_turn_score": np.mean(final_scores),
            "consistency": 1 / (1 + np.std(final_scores)),  # alto = consistente
            "trajectory": self._calculate_trajectory(final_scores),
            "turn_scores": turn_scores
        }
    
    def _calculate_relevance(self, query: str, response: str) -> float:
        """
        Relevância query-response (RUBER unreferenced).
        """
        query_emb = self.sbert.encode(query, convert_to_tensor=True)
        response_emb = self.sbert.encode(response, convert_to_tensor=True)
        
        with torch.no_grad():
            relevance = self.relevance_model(query_emb, response_emb)
        
        return relevance.item()
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Similaridade semântica (SBERT).
        """
        emb1 = self.sbert.encode(text1, convert_to_tensor=True)
        emb2 = self.sbert.encode(text2, convert_to_tensor=True)
        similarity = util.cos_sim(emb1, emb2)
        return similarity.item()
    
    def _calculate_topic_coverage(self, response: str, topics: List[str]) -> float:
        """
        Cobertura de tópicos (SBERT).
        """
        response_emb = self.sbert.encode(response, convert_to_tensor=True)
        topic_embs = self.sbert.encode(topics, convert_to_tensor=True)
        
        similarities = util.cos_sim(response_emb, topic_embs)[0]
        covered = sum(1 for sim in similarities if sim > 0.5)
        
        return covered / len(topics)
    
    def _combine_scores(self, scores: Dict[str, float]) -> float:
        """
        Combina múltiplas métricas (Yeh et al. 2021).
        """
        weights = {
            "relevance": 0.4,
            "similarity": 0.3,
            "topic_coverage": 0.3
        }
        
        final = sum(
            scores.get(metric, 0) * weight
            for metric, weight in weights.items()
        )
        
        return final
    
    def _calculate_trajectory(self, scores: List[float]) -> str:
        """
        Detecta trajetória de aprendizado.
        """
        if len(scores) < 3:
            return "insufficient_data"
        
        # Regressão linear simples
        x = list(range(len(scores)))
        slope = np.polyfit(x, scores, 1)[0]
        
        if slope > 0.05:
            return "improving"
        elif slope < -0.05:
            return "declining"
        else:
            return "stable"
```

---

## Conclusão

Os 5 papers baixados fornecem **insights práticos e implementáveis** para melhorar o `speech_grader`:

### ✅ Implementações Prioritárias:
1. **SBERT** para relevância de tarefa (Reimers & Gurevych 2019)
2. **Combinação de métricas** para avaliação robusta (Yeh et al. 2021)
3. **RUBER** para coerência query-response (Tao et al. 2017)
4. **Análise turn-level + dialog-level** (Yeh et al. 2021)

### 📚 Base Teórica Consolidada:
- **DKT** (Piech 2015) → Base do AKT (já usado)
- **Pathological Speech** (Mekyska 2022) → Features acústicas (futuro)

### 🚀 Próximo Passo:
Implementar `DialogEvaluator` no `speech_grader` integrando SBERT + RUBER + métricas combinadas.

