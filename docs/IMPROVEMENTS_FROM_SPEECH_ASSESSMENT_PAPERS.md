# Melhorias no Sistema de Avaliação CEFR Baseadas em Papers de Avaliação de Fala

## 📋 Resumo Executivo

Análise de 5 papers acadêmicos sobre avaliação de linguagem falada e classificação CEFR identificou **23 melhorias específicas** que podemos implementar no nosso sistema atual. As principais áreas de melhoria são:

1. **Métricas Quantitativas Avançadas** (8 melhorias)
2. **Métodos de Classificação** (4 melhorias)
3. **Features Específicas para Fala** (5 melhorias)
4. **Análise Sequencial e Contours** (3 melhorias)
5. **Métricas Psic linguísticas** (3 melhorias)

---

## 🔍 Papers Analisados

1. **NILC-Metrix (2022)**: 200 métricas para português brasileiro (escrito e falado)
2. **Arnold et al. (2018)**: Classificação CEFR usando métricas de complexidade e modelos supervisionados
3. **Vajjala & Rama (2021)**: Complexity contours + RNNs para classificação CEFR
4. **SpeechLMScore (2022)**: Avaliação não-supervisionada de qualidade de fala usando speech language models
5. **LLM-Eval (2023)**: Avaliação multidimensional unificada para conversas usando LLMs

---

## 🎯 Melhorias Identificadas

### 1. **Métricas Quantitativas Avançadas**

#### 1.1. Diversidade Lexical Mais Robusta
**Problema Atual**: Usamos apenas TTR básico, que é sensível ao comprimento do texto.

**Melhorias**:
- ✅ **MTLD (Measure of Textual Lexical Diversity)**: Menos sensível ao comprimento
  - Fórmula: `types/factors`, onde factors são segmentos que atingiram ponto de estabilização do TTR
  - **Fonte**: Arnold et al. (2018), NILC-Metrix
- ✅ **MATTR (Mean of Moving TTR)**: TTR médio usando técnica de janela deslizante
  - **Fonte**: Arnold et al. (2018)
- ✅ **HDD-D (Hypergeometric Distribution D)**: Probabilidade de encontrar tokens em amostra aleatória
  - **Fonte**: Arnold et al. (2018)
- ✅ **Yule's K**: Medida de diversidade lexical baseada em frequências
  - Fórmula: `10^4 * (Σ(fX * X²) - tokens) / tokens²`
  - **Fonte**: Arnold et al. (2018)

**Implementação Sugerida**:
```python
def calculate_mtld(text: str, min_ttr: float = 0.72) -> float:
    """Calculate MTLD - less sensitive to text length"""
    # Implementation based on Arnold et al. (2018)
    pass

def calculate_mattr(text: str, window_size: int = 50) -> float:
    """Calculate MATTR using sliding window"""
    # Implementation based on Arnold et al. (2018)
    pass
```

#### 1.2. Métricas de Complexidade Sintática Avançadas
**Problema Atual**: Não calculamos métricas sintáticas detalhadas como Yngve, Frazier, ou complexidade de T-units.

**Melhorias**:
- ✅ **Yngve Index**: Mede complexidade baseada em padrão de ramificação sintática
  - **Fonte**: NILC-Metrix (2022)
- ✅ **Frazier Index**: Abordagem bottom-up de complexidade sintática
  - **Fonte**: NILC-Metrix (2022)
- ✅ **T-unit Metrics**: Mean Length of T-unit (MLT), Clauses per T-unit (C/T)
  - **Fonte**: Arnold et al. (2018), Vajjala & Rama (2021)
- ✅ **Dependent Clauses Ratio**: DC/C, DC/T
  - **Fonte**: Arnold et al. (2018)

**Implementação Sugerida**:
```python
def calculate_yngve_index(parse_tree) -> float:
    """Calculate Yngve's complexity index"""
    # Based on NILC-Metrix implementation
    pass

def calculate_t_unit_metrics(text: str) -> Dict[str, float]:
    """Calculate T-unit based metrics"""
    return {
        "mlt": mean_length_t_unit,
        "c_per_t": clauses_per_t_unit,
        "dc_per_c": dependent_clauses_per_clause
    }
```

#### 1.3. Métricas de Coesão e Coerência
**Problema Atual**: Não medimos coesão referencial ou coerência semântica.

**Melhorias**:
- ✅ **LSA-Semantic Cohesion**: Overlap semântico entre sentenças usando LSA
  - **Fonte**: NILC-Metrix (2022) - 11 métricas de coesão semântica
- ✅ **Referential Cohesion**: Overlap de palavras de conteúdo em sentenças adjacentes
  - **Fonte**: NILC-Metrix (2022) - 9 métricas
- ✅ **Givenness e Span**: Métricas de informação prévia e span
  - **Fonte**: NILC-Metrix (2022)

**Implementação Sugerida**:
```python
def calculate_lsa_cohesion(sentences: List[str], lsa_model) -> Dict[str, float]:
    """Calculate semantic cohesion using LSA"""
    # Based on NILC-Metrix (2022)
    return {
        "mean_adjacent_sentence_overlap": ...,
        "std_adjacent_sentence_overlap": ...,
        "mean_all_sentence_pairs": ...,
        "givenness": ...,
        "span": ...
    }
```

#### 1.4. Métricas de Frequência Lexical Normalizadas
**Problema Atual**: Não normalizamos frequências lexicais usando escala Zipf.

**Melhorias**:
- ✅ **Zipf-Scale Normalization**: Normalizar frequências usando escala logarítmica de Zipf
  - **Fonte**: Arnold et al. (2018), NILC-Metrix (2022)
- ✅ **Frequency per Million (fpm)**: Normalização por milhão de palavras
  - **Fonte**: NILC-Metrix (2022)

---

### 2. **Métodos de Classificação**

#### 2.1. Complexity Contours (Sliding Window)
**Problema Atual**: Calculamos apenas médias globais de complexidade, perdendo variação local.

**Melhorias**:
- ✅ **Sliding Window Technique**: Calcular métricas por janela deslizante (sentence-by-sentence)
  - **Fonte**: Vajjala & Rama (2021) - "Complexity Contours"
  - **Resultado**: RNNs treinados em contours superam modelos baseados em médias
  - **AUC Improvement**: +5-10% em alguns níveis CEFR

**Implementação Sugerida**:
```python
def generate_complexity_contours(
    text: str,
    metrics: List[str],
    window_size: int = 1  # sentences
) -> Dict[str, List[float]]:
    """
    Generate complexity contours using sliding window.
    Returns a dict mapping metric_name -> [value1, value2, ...]
    """
    sentences = split_sentences(text)
    contours = {metric: [] for metric in metrics}
    
    for i in range(len(sentences)):
        window = sentences[max(0, i-window_size+1):i+1]
        window_text = " ".join(window)
        
        for metric in metrics:
            value = calculate_metric(window_text, metric)
            contours[metric].append(value)
    
    return contours
```

#### 2.2. Pairwise Classification Models
**Problema Atual**: Classificamos diretamente em 6 classes (A1-C2), mas a distinção entre níveis adjacentes é mais precisa.

**Melhorias**:
- ✅ **Pairwise Binary Classifiers**: 5 modelos binários (A1=>A2, A2=>B1, B1=>B2, B2=>C1, C1=>C2)
  - **Fonte**: Arnold et al. (2018)
  - **Resultado**: AUC de 0.916 para A1=>A2, 0.904 para A2=>B1
  - **Vantagem**: Mais robusto para dados desbalanceados

**Implementação Sugerida**:
```python
class PairwiseCEFRClassifier:
    """Pairwise binary classifiers for CEFR levels"""
    
    def __init__(self):
        self.models = {
            "A1=>A2": train_binary_classifier(...),
            "A2=>B1": train_binary_classifier(...),
            "B1=>B2": train_binary_classifier(...),
            "B2=>C1": train_binary_classifier(...),
            "C1=>C2": train_binary_classifier(...)
        }
    
    def predict(self, text: str) -> str:
        """Predict CEFR level using pairwise voting"""
        votes = {"A1": 0, "A2": 0, "B1": 0, "B2": 0, "C1": 0, "C2": 0}
        
        # A1=>A2: if True, vote A2; else vote A1
        if self.models["A1=>A2"].predict(text):
            votes["A2"] += 1
        else:
            votes["A1"] += 1
        
        # ... similar for other pairs
        
        return max(votes, key=votes.get)
```

#### 2.3. RNN-based Classification com Contours
**Problema Atual**: Usamos apenas LLM para classificação, sem aproveitar informação sequencial.

**Melhorias**:
- ✅ **RNN Classifier com Complexity Contours**: Usar RNN (GRU) para classificar sequências de métricas
  - **Fonte**: Vajjala & Rama (2021)
  - **Arquitetura**: 2 camadas GRU (200 hidden units) + 3 camadas fully connected (512, 256, 6)
  - **Resultado**: Melhor que modelos baseados em médias

**Implementação Sugerida**:
```python
import torch
import torch.nn as nn

class CEFRRNNClassifier(nn.Module):
    """RNN classifier for CEFR levels using complexity contours"""
    
    def __init__(self, input_dim=57, hidden_dim=200, num_classes=6):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=2, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, contours):
        # contours: (batch, seq_len, input_dim)
        gru_out, _ = self.gru(contours)
        last_output = gru_out[:, -1, :]  # Last timestep
        x = self.relu(self.fc1(last_output))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)
```

#### 2.4. Gradient Boosted Trees para Features Importantes
**Problema Atual**: Não identificamos quais features são mais importantes para cada nível.

**Melhorias**:
- ✅ **Gradient Boosted Trees (GBT)**: Identificar features mais importantes
  - **Fonte**: Arnold et al. (2018)
  - **Features Top 5**: `wordtokens`, `wordtypes`, `W` (words), `DC.C` (dependent clauses per clause)
  - **AUC**: 0.916 para A1=>A2, 0.904 para A2=>B1

---

### 3. **Features Específicas para Fala**

#### 3.1. Ajustes para Fala Conversacional
**Problema Atual**: Aplicamos ajustes básicos, mas não métricas específicas de fala.

**Melhorias**:
- ✅ **Pause Detection**: Identificar pausas e hesitações
  - **Fonte**: SpeechLMScore (2022) - usa duração de tokens
- ✅ **Disfluency Markers**: "uh", "um", repetições
  - **Impacto**: Reduzir complexidade aparente em fala natural
- ✅ **Prosodic Features**: Entonação, ritmo (se disponível via TTS/STT)
  - **Fonte**: SpeechLMScore (2022)

**Implementação Sugerida**:
```python
def adjust_for_speech_features(
    text: str,
    audio_features: Optional[Dict] = None
) -> Dict[str, Any]:
    """Adjust complexity metrics for spoken language"""
    adjustments = {
        "pause_count": count_pauses(text),
        "disfluency_markers": count_disfluencies(text),
        "repetitions": detect_repetitions(text)
    }
    
    if audio_features:
        adjustments["prosody"] = extract_prosody(audio_features)
    
    return adjustments
```

#### 3.2. SpeechLMScore Integration
**Problema Atual**: Não avaliamos qualidade de fala usando speech language models.

**Melhorias**:
- ✅ **SpeechLMScore**: Avaliar qualidade de fala usando speech language model
  - **Fonte**: SpeechLMScore (2022)
  - **Método**: Mapear fala para tokens discretos, calcular log-probabilidade média
  - **Vantagem**: Não-supervisionado, não requer anotações humanas

**Implementação Sugerida**:
```python
def calculate_speech_lm_score(
    audio: np.ndarray,
    speech_lm_model
) -> float:
    """
    Calculate SpeechLMScore: average log-probability of speech
    using speech language model
    """
    # Tokenize speech to discrete units
    tokens = tokenize_speech(audio)
    
    # Calculate log-probability for each token
    log_probs = []
    for i, token in enumerate(tokens):
        context = tokens[:i]
        prob = speech_lm_model.predict_proba(token, context)
        log_probs.append(np.log(prob))
    
    return np.mean(log_probs)
```

#### 3.3. Multi-Word Sequences (MWS) para Fala
**Problema Atual**: Não analisamos sequências de palavras (n-grams) específicas de fala.

**Melhorias**:
- ✅ **Register-Specific N-grams**: Bigrams, trigrams, 4-grams, 5-grams de fala conversacional
  - **Fonte**: Vajjala & Rama (2021)
  - **Método**: Normalizar frequência de n-grams usando corpus de referência (COCA spoken)
  - **Fórmula**: `Norm_n,s,r = (|Cn,s,r| / |Un,s|) * log(Σ freq_n,r(c) / |Un,s|)`

**Implementação Sugerida**:
```python
def calculate_mws_features(
    text: str,
    ngram_range: Tuple[int, int] = (1, 5),
    register: str = "spoken"
) -> Dict[str, float]:
    """
    Calculate Multi-Word Sequence features for spoken language
    Based on Vajjala & Rama (2021)
    """
    features = {}
    reference_ngrams = load_reference_ngrams(register)  # COCA spoken
    
    for n in range(ngram_range[0], ngram_range[1] + 1):
        text_ngrams = extract_ngrams(text, n)
        common_ngrams = set(text_ngrams) & set(reference_ngrams[n])
        
        norm_score = (len(common_ngrams) / len(set(text_ngrams))) * \
                     np.log(sum(reference_ngrams[n][ng] for ng in common_ngrams) / 
                            len(set(text_ngrams)))
        
        features[f"mws_{n}gram_{register}"] = norm_score
    
    return features
```

---

### 4. **Métricas Psic linguísticas**

#### 4.1. Age of Acquisition (AoA)
**Problema Atual**: Não consideramos quando palavras são adquiridas na língua nativa.

**Melhorias**:
- ✅ **Age of Acquisition**: Média de idade de aquisição das palavras
  - **Fonte**: NILC-Metrix (2022) - 6 índices
  - **Impacto**: Palavras adquiridas mais cedo = texto mais fácil

**Implementação Sugerida**:
```python
def calculate_aoa_metrics(text: str, aoa_lexicon: Dict[str, float]) -> Dict[str, float]:
    """Calculate Age of Acquisition metrics"""
    words = extract_content_words(text)
    aoas = [aoa_lexicon.get(word.lower(), None) for word in words]
    aoas = [a for a in aoas if a is not None]
    
    return {
        "mean_aoa": np.mean(aoas),
        "std_aoa": np.std(aoas),
        "min_aoa": np.min(aoas),
        "max_aoa": np.max(aoas),
        "median_aoa": np.median(aoas),
        "words_with_aoa": len(aoas) / len(words)
    }
```

#### 4.2. Concreteness, Familiarity, Imageability
**Problema Atual**: Não medimos concreteness, familiaridade ou imageabilidade.

**Melhorias**:
- ✅ **Concreteness**: Média de concreteness das palavras (concreto = mais fácil)
  - **Fonte**: NILC-Metrix (2022) - 6 índices
- ✅ **Familiarity**: Média de familiaridade (similar a frequência subjetiva)
  - **Fonte**: NILC-Metrix (2022) - 6 índices
- ✅ **Imageability**: Capacidade de formar imagem mental
  - **Fonte**: NILC-Metrix (2022) - 6 índices

**Implementação Sugerida**:
```python
def calculate_psycholinguistic_metrics(
    text: str,
    psych_lexicon: Dict[str, Dict[str, float]]
) -> Dict[str, float]:
    """Calculate psycholinguistic metrics (AoA, concreteness, familiarity, imageability)"""
    words = extract_content_words(text)
    
    metrics = {
        "mean_concreteness": [],
        "mean_familiarity": [],
        "mean_imageability": [],
        "mean_aoa": []
    }
    
    for word in words:
        if word.lower() in psych_lexicon:
            entry = psych_lexicon[word.lower()]
            metrics["mean_concreteness"].append(entry.get("concreteness", 0))
            metrics["mean_familiarity"].append(entry.get("familiarity", 0))
            metrics["mean_imageability"].append(entry.get("imageability", 0))
            metrics["mean_aoa"].append(entry.get("aoa", 0))
    
    return {
        k: np.mean(v) if v else 0.0
        for k, v in metrics.items()
    }
```

---

### 5. **Análise Sequencial e Contours**

#### 5.1. Complexity Contours para Análise Local
**Problema Atual**: Perdemos variação local de complexidade ao calcular apenas médias globais.

**Melhorias**:
- ✅ **Sentence-by-Sentence Complexity**: Calcular métricas por sentença
  - **Fonte**: Vajjala & Rama (2021)
  - **Uso**: Alimentar RNN com sequências de métricas
  - **Vantagem**: Captura padrões de variação dentro do texto

**Implementação Sugerida**:
```python
def analyze_complexity_contours(
    text: str,
    metrics: List[str]
) -> Dict[str, Any]:
    """
    Analyze complexity variation using contours
    Returns statistics about complexity progression
    """
    contours = generate_complexity_contours(text, metrics)
    
    analysis = {}
    for metric, values in contours.items():
        analysis[metric] = {
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
            "trend": calculate_trend(values),  # increasing/decreasing/stable
            "volatility": np.std(values) / np.mean(values) if np.mean(values) > 0 else 0
        }
    
    return analysis
```

#### 5.2. Information-Theoretic Measures
**Problema Atual**: Não usamos medidas de complexidade de Kolmogorov.

**Melhorias**:
- ✅ **Kolmogorov Complexity**: Usar algoritmo Deflate para comprimir texto
  - **Fonte**: Vajjala & Rama (2021), Arnold et al. (2018)
  - **Método**: Relacionar tamanho do arquivo comprimido ao original
  - **3 métricas**: Compressão geral, por sentença, por parágrafo

**Implementação Sugerida**:
```python
import zlib

def calculate_kolmogorov_complexity(text: str) -> Dict[str, float]:
    """
    Calculate Kolmogorov complexity using Deflate algorithm
    Based on Vajjala & Rama (2021)
    """
    original_size = len(text.encode('utf-8'))
    compressed = zlib.compress(text.encode('utf-8'))
    compressed_size = len(compressed)
    
    compression_ratio = compressed_size / original_size if original_size > 0 else 0
    
    return {
        "compression_ratio": compression_ratio,
        "complexity_score": 1 - compression_ratio,  # Higher = more complex
        "original_size": original_size,
        "compressed_size": compressed_size
    }
```

---

## 📊 Priorização de Implementação

### 🔴 Alta Prioridade (Impacto Alto, Esforço Médio)
1. **Complexity Contours (Sliding Window)** - Melhora significativa na classificação
2. **Métricas de Diversidade Lexical Robustas (MTLD, MATTR)** - Menos sensíveis ao comprimento
3. **Pairwise Classification Models** - Melhor para dados desbalanceados
4. **Métricas de Coesão Semântica (LSA)** - Importante para coerência

### 🟡 Média Prioridade (Impacto Médio, Esforço Médio)
5. **Métricas Sintáticas Avançadas (Yngve, Frazier, T-units)** - Requer parser sintático
6. **Multi-Word Sequences (MWS)** - Requer corpus de referência
7. **Métricas Psic linguísticas** - Requer léxico especializado
8. **RNN Classifier com Contours** - Requer treinamento

### 🟢 Baixa Prioridade (Impacto Médio, Esforço Alto)
9. **SpeechLMScore Integration** - Requer speech language model treinado
10. **Information-Theoretic Measures** - Complementar, não essencial

---

## 🛠️ Plano de Implementação Sugerido

### Fase 1: Métricas Quantitativas (2-3 semanas)
- [ ] Implementar MTLD, MATTR, HDD-D, Yule's K
- [ ] Implementar métricas de T-units (requer parser)
- [ ] Implementar normalização Zipf para frequências

### Fase 2: Complexity Contours (1-2 semanas)
- [ ] Implementar sliding window technique
- [ ] Gerar contours para todas as métricas existentes
- [ ] Integrar contours no classificador híbrido atual

### Fase 3: Classificação Avançada (2-3 semanas)
- [ ] Implementar pairwise binary classifiers
- [ ] Treinar modelos GBT para identificar features importantes
- [ ] (Opcional) Implementar RNN classifier com contours

### Fase 4: Features Específicas para Fala (1-2 semanas)
- [ ] Implementar detecção de pausas e disfluências
- [ ] Implementar MWS features para fala conversacional
- [ ] Integrar ajustes de prosódia (se disponível)

### Fase 5: Métricas Psic linguísticas (2-3 semanas)
- [ ] Adquirir/criar léxico de AoA, concreteness, familiarity, imageability para português
- [ ] Implementar cálculo de métricas psic linguísticas
- [ ] Integrar no classificador híbrido

---

## 📚 Referências Completas

1. **Leal, S. E., Duran, M. S., Scarton, C. E., Hartmann, N. S., & Aluísio, S. M. (2022).** NILC-Metrix: Assessing the complexity of written and spoken language in Brazilian Portuguese. *arXiv preprint arXiv:2201.03445*.

2. **Arnold, T., Ballier, N., Gaillat, T., & Lissòn, P. (2018).** Predicting CEFRL levels in learner English on the basis of metrics and full texts. *arXiv preprint arXiv:1806.11099*.

3. **Vajjala, S., & Rama, T. (2021).** Automated classification of written proficiency levels on the CEFR-scale through complexity contours and RNNs. In *Proceedings of the 16th Workshop on Innovative Use of NLP for Building Educational Applications* (pp. 180-190). Association for Computational Linguistics.

4. **Maiti, S., Peng, Y., Saeki, T., & Watanabe, S. (2022).** SpeechLMScore: Evaluating speech generation using speech language model. *arXiv preprint arXiv:2212.04559*.

5. **Lin, Y.-T., & Chen, Y.-N. (2023).** LLM-EVAL: Unified multi-dimensional automatic evaluation for open-domain conversations with large language models. *arXiv preprint arXiv:2305.13711*.

---

## ✅ Conclusão

Os papers analisados fornecem uma base sólida para melhorar significativamente nosso sistema de avaliação CEFR. As melhorias mais impactantes são:

1. **Complexity Contours**: Captura variação local de complexidade
2. **Métricas Robustas de Diversidade Lexical**: Menos sensíveis ao comprimento
3. **Pairwise Classification**: Melhor para dados desbalanceados
4. **Features Específicas para Fala**: Ajustes para linguagem conversacional

A implementação gradual dessas melhorias deve resultar em um sistema mais preciso e robusto para classificação CEFR de fala em português.

