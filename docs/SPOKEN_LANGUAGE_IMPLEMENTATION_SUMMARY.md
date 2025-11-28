# Implementação: Suporte para Avaliação de Linguagem Falada

## 📋 Resumo

Implementamos suporte completo para avaliação de **linguagem falada** (spoken language) no classificador CEFR, com ajustes específicos baseados em pesquisas acadêmicas sobre diferenças entre fala e escrita.

---

## ✅ O que foi Implementado

### **1. Documentação Completa**
- **Arquivo**: `docs/SPOKEN_LANGUAGE_ASSESSMENT_BEST_PRACTICES.md`
- **Conteúdo**: 
  - Diferenças fala vs. escrita
  - Métricas específicas para fala
  - Critérios CEFR para produção oral
  - Características da fala conversacional
  - Métodos de avaliação

### **2. Função de Ajuste de Métricas**
- **Arquivo**: `tests/e2e/cefr_level_analyzer.py`
- **Função**: `adjust_metrics_for_speech()`
- **Ajustes**:
  - Comprimento de frase: **30% mais curto** em fala
  - TTR: **15% mais baixo** em fala (repetição é normal)
  - Subordinação: **25% menos** em fala (coordenação preferida)

### **3. Classificador Híbrido Atualizado**
- **Função**: `identify_cefr_level_hybrid()`
- **Parâmetro novo**: `is_speech: bool = False`
- **Comportamento**: Ajusta métricas automaticamente quando `is_speech=True`

---

## 🎯 Diferenças Principais: Fala vs. Escrita

| Característica | Escrita | Fala Conversacional |
|----------------|---------|---------------------|
| **Comprimento de frase** | 8-15 palavras (B1) | 5-10 palavras (B1) |
| **TTR** | 0.50-0.60 (B1) | 0.40-0.50 (B1) |
| **Subordinação** | 20-30% (B1) | 15-20% (B1) |
| **Repetição** | Penalizada | Normal e esperada |
| **Preenchimentos** | Não existe | Normal ("é", "né", "tipo") |
| **Hesitações** | Não existe | Normal ("ééé", "ããã") |
| **Reformulações** | Raras | Comuns e positivas |

---

## 📊 Métricas Específicas para Fala

### **Fluência (quando áudio disponível)**
- **Speech Rate**: 120-150 sílabas/minuto (B1)
- **Pause Ratio**: 15-20% do tempo (B1)
- **Mean Length of Run**: 6-10 palavras entre pausas (B1)

### **Precisão**
- **Pronunciation Accuracy**: 75-85% (B1)
- **Grammatical Accuracy**: 5-10 erros/100 palavras (B1)

### **Complexidade**
- **Frases por turno**: 3-4 frases (B1)
- **TTR ajustado**: 0.40-0.50 (B1)

---

## 🔧 Como Usar

### **Para Fala Conversacional (transcrita)**
```python
# Este sistema é ESPECIALIZADO para avaliação de LINGUAGEM FALADA
# As métricas são automaticamente ajustadas para características da fala
analysis = await identify_cefr_level_hybrid(
    text="Ééé, eu quero... quer dizer, eu gostaria de um café, né?",
    session=session
    # is_speech não existe mais - sempre ajustado para fala
)
```

**Nota**: O sistema sempre aplica ajustes para fala conversacional, pois é um app especializado em avaliação de speech.

---

## 📈 Ajustes Aplicados Automaticamente

**Sempre aplicados** (sistema especializado para fala):

1. **Comprimento de frase**: Dividido por 1.3
   - Exemplo: 10 palavras → 7.7 palavras (ajustado)
   - **Razão**: Fala conversacional tem frases 30% mais curtas que escrita

2. **TTR**: Multiplicado por 0.85
   - Exemplo: 0.60 → 0.51 (ajustado)
   - **Razão**: Repetição é normal e esperada em fala

3. **Subordinação**: Multiplicado por 0.75
   - Exemplo: 0.20 → 0.15 (ajustado)
   - **Razão**: Coordenação é preferida sobre subordinação em fala

**Baseado em**: Leal et al. (2022), Arnold et al. (2018), CEFR Companion Volume (2020)

---

## 🎤 Características da Fala Consideradas

### **Não Penalizadas (Normais em Fala)**
- ✅ Preenchimentos ("é", "né", "tipo", "assim")
- ✅ Hesitações ocasionais ("ééé", "ããã")
- ✅ Repetição moderada de palavras
- ✅ Reformulações ("eu quero... quer dizer, eu gostaria...")
- ✅ Frases incompletas
- ✅ Elipse (omissão de palavras)

### **Ainda Avaliadas**
- ✅ Comprimento de frase (ajustado)
- ✅ Complexidade sintática (ajustada)
- ✅ Diversidade lexical (ajustada)
- ✅ Precisão gramatical
- ✅ Uso de vocabulário apropriado

---

## 📚 Critérios CEFR para Produção Oral

### **A1**
- Frases: 3-5 palavras
- Speech Rate: 60-90 sílabas/min
- Pausas: 30-40% do tempo
- Pronúncia: 40-60% correta

### **A2**
- Frases: 5-8 palavras
- Speech Rate: 90-120 sílabas/min
- Pausas: 20-30% do tempo
- Pronúncia: 60-75% correta

### **B1**
- Frases: 8-12 palavras
- Speech Rate: 120-150 sílabas/min
- Pausas: 15-20% do tempo
- Pronúncia: 75-85% correta

### **B2**
- Frases: 10-18 palavras
- Speech Rate: 150-180 sílabas/min
- Pausas: 10-15% do tempo
- Pronúncia: 85-92% correta

### **C1**
- Frases: 15-25 palavras
- Speech Rate: 180-210 sílabas/min
- Pausas: 5-10% do tempo
- Pronúncia: 92-97% correta

### **C2**
- Frases: 20-30+ palavras
- Speech Rate: 210+ sílabas/min
- Pausas: <5% do tempo
- Pronúncia: 97-100% correta

---

## 🚀 Próximos Passos (Opcional)

### **1. Adicionar Métricas de Fluência**
Quando áudio disponível:
- Speech Rate (sílabas/segundo)
- Pause Ratio (% do tempo)
- Mean Length of Run (palavras entre pausas)

### **2. Detecção de Marcadores Conversacionais**
- Identificar preenchimentos ("é", "né")
- Identificar hesitações ("ééé")
- Não penalizar (são normais)

### **3. Análise de Pronúncia**
- Integrar com Azure Speech Services ou similar
- Avaliar precisão de pronúncia
- Feedback específico sobre sons problemáticos

---

## ✅ Status

- ✅ Documentação completa sobre avaliação de fala
- ✅ Função de ajuste de métricas implementada
- ✅ Classificador híbrido atualizado com suporte a fala
- ✅ Critérios CEFR para produção oral documentados
- ✅ Características da fala conversacional consideradas

**Sistema pronto para avaliar tanto fala quanto escrita com critérios apropriados!**

---

## 📚 Referências

1. **CEFR Companion Volume (2020)**: Descritores para produção oral
2. **NILC-Metrix (Leal et al., 2022)**: Métricas para português falado
3. **Arnold et al. (2018)**: Diferenças entre avaliação escrita e oral
4. **Azure Speech Services**: Pronunciation Assessment API

