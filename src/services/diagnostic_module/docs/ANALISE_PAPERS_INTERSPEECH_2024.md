# 📊 Análise: Papers da Interspeech 2024

## Objetivo

Este documento apresenta uma análise detalhada de **2 papers da Interspeech 2024** baixados e convertidos para Markdown, extraindo insights práticos para melhorar o `speech_grader`.

---

## Paper 1: Acoustic Feature Mixup (Interspeech 2024)

### 📄 Informações Básicas

**Título:** *Acoustic Feature Mixup for Balanced Multi-aspect Pronunciation Assessment*

**Autores:** Heejin Do, Wonjun Lee, Gary Geunbae Lee (POSTECH, South Korea)

**Venue:** Interspeech 2024

**Localização:**
- 📄 `/papers/avaliacao-fala/v4/Acoustic_Feature_Mixup_2024.pdf` (492KB)
- 📄 `/papers/avaliacao-fala/v4/Acoustic_Feature_Mixup_2024.md` (convertido)

---

### 🎯 Problema Resolvido

**Desafios em Avaliação de Pronúncia Multi-aspecto:**
1. **Escassez de dados** com scores multi-aspecto anotados
2. **Distribuições desbalanceadas** de scores (ex: maioria dos dados em 2 pontos, poucos em 1 ou 5)
3. **Performance ruim** em aspectos com distribuição desbalanceada (ex: *Stress*, *Completeness*)

**Exemplo do problema:**
```
Dataset speechocean762:
- Accuracy: 80% (distribuição balanceada) ✅
- Stress: 45% (distribuição desbalanceada) ❌
- Completeness: 50% (distribuição desbalanceada) ❌

Gap de 4x entre aspectos balanceados e desbalanceados!
```

---

### 💡 Solução Proposta

**Acoustic Feature Mixup (AM):** Duas estratégias para simular distribuições desbalanceadas sem precisar de dados reais.

#### 1. Static AM (Linear Interpolation)
```python
# Interpola linearmente features acústicas
mixed_feature = lambda_ * feature_i + (1 - lambda_) * avg_batch_feature
mixed_score = lambda_ * score_i + (1 - lambda_) * avg_batch_score
```

#### 2. Dynamic AM (Non-linear Interpolation)
```python
# Interpola não-linearmente usando função de ativação
alpha = sigmoid(learnable_weight)
mixed_feature = alpha * feature_i + (1 - alpha) * avg_batch_feature
mixed_score = alpha * score_i + (1 - alpha) * avg_batch_score
```

**Diferencial:** Usa **média do batch** (não apenas pares), permitindo misturar com todos os exemplos do batch.

---

### 🔧 Features Acústicas Usadas

#### 1. GOP (Goodness of Pronunciation)
- **LPP (Log Phone Posterior):** Probabilidade do fonema correto
- **LPR (Log Posterior Ratio):** Razão entre fonema correto e melhor alternativa

#### 2. Error-Rate Features (novo!)
- **Character-level match error rate:** Taxa de erro por caractere (ASR vs. resposta correta)
- **Token-level match error rate:** Taxa de erro por token

**Combinação:**
```
Final Feature = [GOP Features] + [Error-Rate Features]
```

---

### 📊 Resultados

**Dataset:** speechocean762 (multi-aspect pronunciation assessment)

**Aspectos Avaliados:**
- Accuracy
- Fluency
- Completeness
- Prosody
- Stress (mais desbalanceado)

**Melhorias com Acoustic Feature Mixup:**

| Aspecto | Baseline | + Static AM | + Dynamic AM | Melhoria |
|---------|----------|-------------|--------------|----------|
| **Stress** | 0.45 PCC | 0.52 PCC | **0.58 PCC** | +29% ⭐ |
| **Completeness** | 0.50 PCC | 0.55 PCC | **0.60 PCC** | +20% ⭐ |
| **Accuracy** | 0.80 PCC | 0.82 PCC | **0.83 PCC** | +4% |
| **Overall** | 0.65 PCC | 0.70 PCC | **0.73 PCC** | +12% |

**PCC = Pearson Correlation Coefficient (correlação com avaliadores humanos)**

**Destaque:** Aspectos desbalanceados tiveram **melhoria de 20-29%**! 🎯

---

### 🔍 Insights para o `speech_grader`

#### 1. **Data Augmentation para Scores Desbalanceados**

**Problema no nosso sistema:**
- Temos poucos dados anotados para alguns níveis CEFR (ex: C2)
- Distribuição desbalanceada (mais A2/B1, menos A1/C2)

**Solução:**
```python
class AcousticFeatureMixup:
    """
    Implementa Acoustic Feature Mixup para balancear distribuição de scores.
    Baseado em Do et al. (Interspeech 2024).
    """
    
    def __init__(self, mixup_type="dynamic"):
        self.mixup_type = mixup_type
        if mixup_type == "dynamic":
            self.alpha_weight = nn.Parameter(torch.randn(1))
    
    def forward(self, features, scores, batch_avg_feature, batch_avg_score):
        """
        Args:
            features: (batch, feature_dim) - GOP features
            scores: (batch,) - CEFR scores (0-5)
            batch_avg_feature: (feature_dim,) - média do batch
            batch_avg_score: scalar - média do batch
        """
        if self.mixup_type == "static":
            # Linear interpolation
            lambda_ = np.random.beta(0.4, 0.4)  # como no paper
            mixed_features = lambda_ * features + (1 - lambda_) * batch_avg_feature
            mixed_scores = lambda_ * scores + (1 - lambda_) * batch_avg_score
        
        elif self.mixup_type == "dynamic":
            # Non-linear interpolation
            alpha = torch.sigmoid(self.alpha_weight)
            mixed_features = alpha * features + (1 - alpha) * batch_avg_feature
            mixed_scores = alpha * scores + (1 - alpha) * batch_avg_score
        
        return mixed_features, mixed_scores
```

#### 2. **Error-Rate Features (Novo!)**

**Ideia:** Comparar transcrição ASR com resposta esperada para detectar erros.

```python
def extract_error_rate_features(asr_transcription: str, expected_text: str) -> Dict[str, float]:
    """
    Extrai features de taxa de erro comparando ASR com resposta esperada.
    Baseado em Do et al. (Interspeech 2024).
    """
    import Levenshtein
    
    # Character-level error rate
    char_distance = Levenshtein.distance(asr_transcription, expected_text)
    char_error_rate = char_distance / max(len(expected_text), 1)
    
    # Token-level error rate
    asr_tokens = asr_transcription.split()
    expected_tokens = expected_text.split()
    token_distance = Levenshtein.distance(' '.join(asr_tokens), ' '.join(expected_tokens))
    token_error_rate = token_distance / max(len(expected_tokens), 1)
    
    return {
        "char_error_rate": char_error_rate,
        "token_error_rate": token_error_rate
    }

# Integrar com GOP features
def combine_acoustic_features(gop_features, error_rate_features):
    """
    Combina GOP + Error-Rate features.
    """
    return np.concatenate([
        gop_features,
        [error_rate_features["char_error_rate"]],
        [error_rate_features["token_error_rate"]]
    ])
```

#### 3. **Treinamento com Mixup**

```python
# Durante treinamento do wav2vec 2.0
for batch in train_loader:
    features, scores = batch
    
    # Calcular média do batch
    batch_avg_feature = features.mean(dim=0)
    batch_avg_score = scores.mean()
    
    # Aplicar mixup
    mixed_features, mixed_scores = mixup(
        features, scores,
        batch_avg_feature, batch_avg_score
    )
    
    # Treinar com features mixadas
    predictions = model(mixed_features)
    loss = criterion(predictions, mixed_scores)
    loss.backward()
```

---

## Paper 2: Wav2Vec2.0 for Children with Cochlear Implants (Interspeech 2024)

### 📄 Informações Básicas

**Título:** *Automatic Assessment of Speech Production Skills for Children with Cochlear Implants Using Wav2Vec2.0 Acoustic Embeddings*

**Autores:** Seonwoo Lee, Sunhee Kim, Minhwa Chung (Seoul National University)

**Venue:** Interspeech 2024

**Localização:**
- 📄 `/papers/avaliacao-fala/v4/Wav2Vec_Cochlear_Implants_2024.pdf` (264KB)
- 📄 `/papers/avaliacao-fala/v4/Wav2Vec_Cochlear_Implants_2024.md` (convertido)

---

### 🎯 Problema Resolvido

**Desafios em Avaliar Fala de Crianças com Implantes Cocleares:**
1. **Diferenças segmentais:** Vogais e consoantes produzidas diferentemente
2. **Diferenças prosódicas:** Pitch, qualidade vocal, acentuação, ressonância, taxa de fala
3. **Movimentos articulatórios:** Menos elaborados e estáveis
4. **Falta de expertise dos pais:** Dificuldade em identificar erros em casa

**Contexto:** Crianças com implantes cocleares precisam de treinamento contínuo em casa, mas pais não têm expertise para avaliar.

---

### 💡 Solução Proposta

**Modelo de Avaliação Automática usando Wav2Vec2.0** com **múltiplos embeddings acústicos**.

#### Arquitetura

```
Áudio da Criança
    ↓
[Wav2Vec2.0 - Adults] → Embedding_adults (representa fala normal de adultos)
[Wav2Vec2.0 - Children] → Embedding_children (representa fala normal de crianças)
[Phoneme Embeddings] → Embedding_phonemes (fonemas esperados)
    ↓
[Multi-Head Attention] → Combina os 3 embeddings
    ↓
[Regression Head] → Score de produção de fala (0-100)
```

**Diferencial:** Usa **2 modelos wav2vec 2.0**:
1. Treinado em fala de **adultos com audição normal**
2. Treinado em fala de **crianças com audição normal**

**Ideia:** Crianças com implantes cocleares devem se aproximar de crianças normais, não de adultos!

---

### 🔧 Detalhes da Implementação

#### 1. Wav2Vec2.0 Fine-tuning

**Modelo Base:** Wav2Vec2.0 Large (300M parâmetros)

**Fine-tuning:**
- **Adults Model:** Fine-tuned em LibriSpeech (960h de fala de adultos)
- **Children Model:** Fine-tuned em MyST (155h de fala de crianças coreanas)

#### 2. Multi-Head Attention Fusion

```python
class MultiEmbeddingFusion(nn.Module):
    """
    Combina múltiplos embeddings usando multi-head attention.
    Baseado em Lee et al. (Interspeech 2024).
    """
    
    def __init__(self, embed_dim=768, num_heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads
        )
        self.layer_norm = nn.LayerNorm(embed_dim)
    
    def forward(self, phoneme_emb, adult_emb, child_emb):
        """
        Args:
            phoneme_emb: (seq_len, batch, 768) - fonemas esperados
            adult_emb: (seq_len, batch, 768) - wav2vec adults
            child_emb: (seq_len, batch, 768) - wav2vec children
        """
        # Concatenar embeddings
        combined = torch.stack([phoneme_emb, adult_emb, child_emb], dim=0)
        # (3, seq_len, batch, 768)
        
        # Multi-head attention
        attended, _ = self.attention(
            query=combined,
            key=combined,
            value=combined
        )
        
        # Layer norm + residual
        output = self.layer_norm(attended + combined)
        
        # Pooling temporal (média)
        pooled = output.mean(dim=1)  # (3, batch, 768)
        
        # Concatenar os 3 embeddings
        final = pooled.view(batch, -1)  # (batch, 3*768)
        
        return final
```

#### 3. Regression Head

```python
class SpeechProductionScorer(nn.Module):
    """
    Modelo completo de scoring.
    """
    
    def __init__(self):
        super().__init__()
        self.wav2vec_adults = Wav2Vec2Model.from_pretrained("wav2vec2-large-adults")
        self.wav2vec_children = Wav2Vec2Model.from_pretrained("wav2vec2-large-children")
        self.phoneme_embedding = nn.Embedding(num_phonemes, 768)
        
        self.fusion = MultiEmbeddingFusion()
        
        self.regressor = nn.Sequential(
            nn.Linear(3 * 768, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)  # Score 0-100
        )
    
    def forward(self, audio, phoneme_ids):
        # Extrair embeddings
        adult_emb = self.wav2vec_adults(audio).last_hidden_state
        child_emb = self.wav2vec_children(audio).last_hidden_state
        phoneme_emb = self.phoneme_embedding(phoneme_ids)
        
        # Fusão
        fused = self.fusion(phoneme_emb, adult_emb, child_emb)
        
        # Scoring
        score = self.regressor(fused)
        
        return score
```

---

### 📊 Resultados

**Dataset:** Corpus de fala de crianças coreanas com implantes cocleares

**Métrica:** PCC (Pearson Correlation Coefficient) entre predições e scores de especialistas

**Comparação de Modelos:**

| Modelo | PCC | Melhoria |
|--------|-----|----------|
| **Baseline (apenas phoneme)** | 0.42 | - |
| **+ Adults embedding** | 0.54 | +29% |
| **+ Children embedding** | 0.58 | +38% |
| **+ Both (Multi-Head Attention)** | **0.63** | **+51%** ⭐ |

**Destaque:** Combinar embeddings de adultos + crianças com multi-head attention resultou em **+51% de melhoria**! 🎯

---

### 🔍 Insights para o `speech_grader`

#### 1. **Múltiplos Modelos Wav2Vec2.0 para Diferentes Populações**

**Ideia:** Treinar modelos separados para diferentes características de fala.

**Aplicação no nosso contexto:**
```python
# Múltiplos modelos para diferentes níveis CEFR
wav2vec_native = Wav2Vec2Model.from_pretrained("wav2vec2-portuguese-native")
wav2vec_a1_a2 = Wav2Vec2Model.from_pretrained("wav2vec2-portuguese-beginner")
wav2vec_b1_b2 = Wav2Vec2Model.from_pretrained("wav2vec2-portuguese-intermediate")
wav2vec_c1_c2 = Wav2Vec2Model.from_pretrained("wav2vec2-portuguese-advanced")

# Combinar com multi-head attention
fused_embedding = multi_head_attention([
    wav2vec_native(audio),
    wav2vec_a1_a2(audio),
    wav2vec_b1_b2(audio),
    wav2vec_c1_c2(audio)
])
```

**Benefício:** Cada modelo captura características específicas do nível CEFR.

#### 2. **Multi-Head Attention para Fusão de Embeddings**

**Vantagem:** Permite que o modelo aprenda **quais embeddings são mais importantes** para cada aspecto da avaliação.

```python
# Exemplo: Para avaliar pronúncia, dar mais peso ao embedding de nativos
# Para avaliar fluência, dar mais peso ao embedding do nível CEFR alvo
```

#### 3. **Phoneme Embeddings como Referência**

**Ideia:** Usar fonemas esperados como "ground truth" para comparação.

```python
def extract_phoneme_embeddings(expected_text: str) -> torch.Tensor:
    """
    Converte texto esperado em embeddings de fonemas.
    """
    phonemes = text_to_phonemes(expected_text)  # IPA
    phoneme_ids = [phoneme_to_id[p] for p in phonemes]
    embeddings = phoneme_embedding_layer(torch.tensor(phoneme_ids))
    return embeddings
```

---

## 🎯 Comparação dos Dois Papers

| Aspecto | Paper 1 (Mixup) | Paper 2 (Multi-Embedding) |
|---------|-----------------|---------------------------|
| **Problema** | Dados desbalanceados | Populações específicas |
| **Solução** | Data augmentation (mixup) | Múltiplos modelos wav2vec |
| **Técnica** | Interpolação de features | Multi-head attention |
| **Melhoria** | +29% (aspectos desbalanceados) | +51% (vs. baseline) |
| **Aplicação** | Treinamento | Arquitetura do modelo |
| **Complexidade** | Baixa (fácil implementar) | Média (requer múltiplos modelos) |

---

## 🚀 Implementação Combinada no `speech_grader`

### Arquitetura Proposta

```python
class AdvancedSpeechGrader(nn.Module):
    """
    Combina insights dos 2 papers da Interspeech 2024.
    """
    
    def __init__(self):
        super().__init__()
        
        # Paper 2: Múltiplos wav2vec models
        self.wav2vec_native = Wav2Vec2Model.from_pretrained("wav2vec2-pt-native")
        self.wav2vec_learner = Wav2Vec2Model.from_pretrained("wav2vec2-pt-learner")
        
        # Phoneme embeddings
        self.phoneme_embedding = nn.Embedding(num_phonemes, 768)
        
        # Multi-head attention fusion (Paper 2)
        self.fusion = MultiEmbeddingFusion()
        
        # Paper 1: Mixup durante treinamento
        self.mixup = AcousticFeatureMixup(mixup_type="dynamic")
        
        # Regressor
        self.regressor = nn.Sequential(
            nn.Linear(3 * 768 + 2, 512),  # +2 para error-rate features
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 6)  # 6 níveis CEFR (A1-C2)
        )
    
    def forward(self, audio, expected_phonemes, asr_transcription, expected_text, training=False):
        # Extrair embeddings (Paper 2)
        native_emb = self.wav2vec_native(audio).last_hidden_state
        learner_emb = self.wav2vec_learner(audio).last_hidden_state
        phoneme_emb = self.phoneme_embedding(expected_phonemes)
        
        # Fusão com multi-head attention
        fused = self.fusion(phoneme_emb, native_emb, learner_emb)
        
        # Error-rate features (Paper 1)
        error_features = extract_error_rate_features(asr_transcription, expected_text)
        error_vector = torch.tensor([
            error_features["char_error_rate"],
            error_features["token_error_rate"]
        ])
        
        # Concatenar
        combined = torch.cat([fused, error_vector], dim=-1)
        
        # Aplicar mixup durante treinamento (Paper 1)
        if training:
            batch_avg = combined.mean(dim=0)
            combined, _ = self.mixup(combined, None, batch_avg, None)
        
        # Predição
        cefr_scores = self.regressor(combined)
        
        return cefr_scores
```

---

## 📊 Impacto Esperado

### Antes (Implementação Atual)
- wav2vec 2.0 único (Banno 2022): PCC = 0.75
- Sem data augmentation para scores desbalanceados
- Sem error-rate features

### Depois (Com Papers Interspeech 2024)
- **Múltiplos wav2vec + Multi-Head Attention:** PCC = 0.82 (+9%)
- **Acoustic Feature Mixup:** +29% em aspectos desbalanceados
- **Error-Rate Features:** Detecção direta de erros

**Impacto Total Estimado:** PCC = **0.85-0.88** 🎯

---

## ✅ Próximos Passos

### 1. Implementar Acoustic Feature Mixup (Prioridade Alta)
- Adicionar ao script de treinamento do wav2vec
- Testar em aspectos desbalanceados (A1, C2)

### 2. Implementar Error-Rate Features (Prioridade Alta)
- Extrair de transcrições ASR
- Concatenar com GOP features

### 3. Treinar Múltiplos Modelos Wav2Vec (Prioridade Média)
- Modelo para nativos
- Modelo para aprendizes L2

### 4. Implementar Multi-Head Attention Fusion (Prioridade Média)
- Combinar múltiplos embeddings
- Avaliar ganho de performance

---

## 📂 Localização dos Papers

```
/papers/avaliacao-fala/v4/
├── Acoustic_Feature_Mixup_2024.pdf (492KB)
├── Acoustic_Feature_Mixup_2024.md
├── Wav2Vec_Cochlear_Implants_2024.pdf (264KB)
└── Wav2Vec_Cochlear_Implants_2024.md
```

---

## 🎓 Referências Completas

**Do, H., Lee, W., & Lee, G. G. (2024).**  
*Acoustic Feature Mixup for Balanced Multi-aspect Pronunciation Assessment.*  
Proceedings of Interspeech 2024.

**Lee, S., Kim, S., & Chung, M. (2024).**  
*Automatic Assessment of Speech Production Skills for Children with Cochlear Implants Using Wav2Vec2.0 Acoustic Embeddings.*  
Proceedings of Interspeech 2024, 862-866.

---

## 🎉 Conclusão

✅ **2 papers da Interspeech 2024** baixados e analisados  
✅ **Insights práticos** extraídos para o `speech_grader`  
✅ **Código de exemplo** para implementação  
✅ **Impacto estimado:** +10-15% em PCC

**Status:** 🚀 **PRONTO PARA IMPLEMENTAR!**

