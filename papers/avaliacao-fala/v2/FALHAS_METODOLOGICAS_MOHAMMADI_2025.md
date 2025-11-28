# Análise Crítica: Falhas Metodológicas - Mohammadi et al. (2025)

## 📋 Resumo Executivo

Este documento identifica **12 falhas metodológicas críticas e importantes** no paper "Automatic Proficiency Assessment in L2 English Learners" (Mohammadi et al., 2025).

---

## ❌ FALHAS CRÍTICAS

### 1. **Problema com Dataset EFCAMDAT**

**Problema:**
- EFCAMDAT é um dataset de **TEXTOS ESCRITOS** (essays/redações)
- O paper usa para avaliação de **FALA** (speaking assessment)
- Inconsistência fundamental: texto escrito vs fala são modalidades completamente diferentes

**Evidência:**
> "The EF-Cambridge open language dataset comprises **1,180,310 essays written by learners**" (linha 267)

**Impacto:**
- Resultados podem não generalizar para fala real
- BERT foi usado com transcrições, mas EFCAMDAT original é texto escrito
- Características de fala (prosódia, fluência, hesitações) não estão presentes em textos escritos

**Severidade:** 🔴 **CRÍTICA**

---

### 2. **Dataset Privado Não Disponível**

**Problema:**
- Dataset privado usado extensivamente mas **não disponibilizado**
- Impossível reproduzir resultados
- Impossível validar metodologia
- Violação de princípios de ciência aberta e reprodutibilidade

**Evidência:**
> "**Private:** The private dataset contains structured interview recordings..." (linha 279)
> - Nenhuma menção de disponibilização ou acesso

**Impacto:**
- Resultados não são verificáveis
- Comunidade científica não pode validar ou melhorar
- Limita avanço da área

**Severidade:** 🔴 **CRÍTICA**

---

### 3. **Métricas Incompletas**

**Problema:**
- CNN e ResNet: **'NC' (Not Computed)** para Precision/Recall/F1
- Apenas accuracy reportada para modelos com baixa performance
- Dificulta comparação justa entre modelos

**Evidência (Tabela VI):**
```
|Model|Accuracy|Macro Precision|Macro Recall|Macro F1|
|CNN|29.2%|NC|NC|NC|
|ResNet|31.4%|NC|NC|NC|
```

**Impacto:**
- Análise incompleta de performance
- Não sabemos se modelos ruins têm viés para classes específicas
- Dificulta identificar problemas de classificação

**Severidade:** 🔴 **CRÍTICA**

---

### 4. **Falta de Análise de Erros**

**Problema:**
- Não analisam quais níveis são mais confundidos
- Não mostram **matriz de confusão**
- Não identificam padrões de erro
- Não explicam por que certos níveis são mais difíceis

**Impacto:**
- Dificulta entender limitações do modelo
- Não sabemos se modelo confunde níveis adjacentes (esperado) ou distantes (problema)
- Impossível melhorar modelo sem entender erros

**Severidade:** 🔴 **CRÍTICA**

---

## ⚠️ FALHAS IMPORTANTES

### 5. **Desbalanceamento de Dados**

**Problema:**
- EFCAMDAT: A1=191k, A2=129k, B1=61k, B2=18k, C1=5k
- Distribuição **altamente desbalanceada** (A1 tem **38x mais** que C1)
- Usaram apenas 2,400 amostras (2k train, 200 val, 200 test)
- Amostragem pode não representar distribuição real

**Evidência (Tabela III):**
```
|Level|#ofAnswers|
|A1|191,663|
|A2|129,591|
|B1|61,506|
|B2|18,187|
|C1|5,115|
```

**Impacto:**
- Modelos podem ter viés para níveis mais frequentes (A1, A2)
- Performance em níveis raros (C1) pode ser subestimada
- Resultados podem não generalizar para distribuições balanceadas

**Severidade:** 🟡 **IMPORTANTE**

---

### 6. **Problemas com Transcrições**

**Problema:**
- Whisper usado para transcrições automáticas
- Paper admite: **"Whisper occasionally produced errors"**
- Erros em overlapping speech e ruído
- Apenas **"subset"** foi revisado manualmente

**Evidência:**
> "Despite its strong performance, Whisper occasionally produced errors due to overlapping speech and real-world noise conditions. A subset of the data was manually reviewed to verify segmentation accuracy." (linhas 186-189)

**Impacto:**
- Erros de transcrição podem afetar avaliação baseada em texto
- Não sabemos quantos erros existem
- Ground truth pode estar contaminado

**Severidade:** 🟡 **IMPORTANTE**

---

### 7. **Tamanho de Test Set Pequeno**

**Problema:**
- **ANGLISH:** apenas **6 speakers** no test set
- **Private:** 12 speakers por nível (L3-L5)
- Test sets muito pequenos para generalização confiável

**Evidência:**
> "composed of 6 speakers (one male and one female from each proficiency level)" (linha 323)

**Impacto:**
- Intervalos de confiança amplos
- Resultados instáveis (pequenas mudanças podem alterar resultados drasticamente)
- Generalização não confiável

**Severidade:** 🟡 **IMPORTANTE**

---

### 8. **Falta de Baseline Humano**

**Problema:**
- Não reportam **concordância inter-avaliadores**
- Não comparam com avaliação humana
- Não mencionam se há ground truth validado
- Não sabemos se 85% é bom ou ruim vs humanos

**Impacto:**
- Não sabemos se modelo é melhor ou pior que humanos
- Não sabemos se 85% accuracy é aceitável
- Dificulta interpretar resultados

**Severidade:** 🟡 **IMPORTANTE**

---

### 9. **Mistura de Tarefas Diferentes**

**Problema:**
- **ANGLISH:** classificação de proficiência (3 classes: NES, FR1, FR2)
- **EFCamDat:** classificação CEFR (5 classes: A1-C1)
- **Private:** classificação de níveis (3 classes: L3-L5)
- Diferentes granularidades dificultam comparação

**Impacto:**
- Resultados não são diretamente comparáveis
- Dificulta entender qual abordagem é melhor
- Mistura diferentes definições de "proficiência"

**Severidade:** 🟡 **IMPORTANTE**

---

### 10. **Speaker Diarization Não Validado**

**Problema:**
- Usaram PyAnnote + SpeechBrain para diarização
- Identificaram learner como **"speaker com mais tempo"**
- Apenas **"subset"** foi revisado manualmente
- Erros de diarização podem contaminar dados

**Evidência:**
> "the learner was identified as the speaker with the longest total speaking time" (linha 178)

**Impacto:**
- Dados podem conter fala do entrevistador
- Modelos podem aprender características do entrevistador, não do learner
- Contaminação de dados

**Severidade:** 🟡 **IMPORTANTE**

---

### 11. **Multi-Task Learning Questionável**

**Problema:**
- Adicionaram classificação de **gênero** como tarefa auxiliar
- Melhorou accuracy de 75% para 80%
- Mas **gênero não é relevante** para proficiência
- Pode introduzir **viés de gênero** no modelo

**Evidência:**
> "Multi-task: A dual-layer DNN... for joint gender (binary) and proficiency (3-class) classification" (linha 354)

**Impacto:**
- Modelo pode usar gênero como proxy para proficiência
- Viés de gênero pode ser incorporado no modelo
- Não é ético usar gênero para avaliar proficiência

**Severidade:** 🟡 **IMPORTANTE**

---

### 12. **Falta de Ablation Study**

**Problema:**
- Não testam impacto de cada componente
- Não comparam com baselines mais simples
- Não analisam contribuição de cada feature
- Não sabem o que realmente importa

**Impacto:**
- Não sabemos o que realmente importa no modelo
- Dificulta melhorar ou simplificar modelo
- Não sabemos se complexidade é necessária

**Severidade:** 🟡 **IMPORTANTE**

---

## 📊 Resumo das Falhas

### **Críticas (❌):**
1. Dataset EFCAMDAT usado para fala (é texto escrito)
2. Dataset privado não disponível (não reproduzível)
3. Métricas incompletas (NC para modelos ruins)
4. Falta de análise de erros

### **Importantes (⚠️):**
1. Desbalanceamento de dados
2. Problemas com transcrições (Whisper)
3. Test sets muito pequenos
4. Falta de baseline humano
5. Mistura de tarefas diferentes
6. Speaker diarization não validado
7. Multi-task learning questionável
8. Falta de ablation study

---

## 💡 Recomendações

### **Para Reproduzir/Validar:**
1. Usar datasets de **fala real** (não texto escrito)
2. Disponibilizar dataset privado ou usar apenas públicos
3. Reportar todas as métricas (Precision, Recall, F1)
4. Mostrar matriz de confusão e análise de erros

### **Para Melhorar:**
1. Balancear datasets ou usar técnicas de balanceamento
2. Validar transcrições manualmente (100% ou amostra representativa)
3. Aumentar tamanho de test sets
4. Comparar com baseline humano
5. Usar tarefas consistentes entre datasets
6. Validar speaker diarization completamente
7. Remover gênero de multi-task learning
8. Realizar ablation study

---

## 🔍 Conclusão

O paper tem **várias falhas metodológicas críticas** que limitam sua confiabilidade e reprodutibilidade. As principais preocupações são:

1. **Uso incorreto de dataset** (texto escrito para fala)
2. **Falta de reprodutibilidade** (dataset privado)
3. **Análise incompleta** (métricas faltando, sem análise de erros)
4. **Viés potencial** (gênero, desbalanceamento)

**Recomendação:** Usar este paper com **cautela** e considerar as limitações ao aplicar metodologia similar.

---

**Última atualização:** 2025-11-23

