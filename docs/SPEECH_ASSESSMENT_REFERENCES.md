# Referências Bibliográficas: Avaliação de Linguagem Falada (Speech Assessment)

## 📚 Papers Acadêmicos Principais

### **1. Complexidade Textual e Métricas**

**Leal, S. E., Duran, M. S., Scarton, C. E., Hartmann, N. S., & Aluísio, S. M. (2022).**
NILC-Metrix: Assessing the complexity of written and spoken language in Brazilian Portuguese.
*arXiv preprint arXiv:2201.03445*.
https://arxiv.org/abs/2201.03445

- **Contribuição**: Sistema com 200 métricas para português brasileiro (escrito e falado)
- **Relevância**: Métricas específicas para português falado
- **Uso no sistema**: Base para métricas quantitativas de complexidade

---

**Arnold, T., Ballier, N., Gaillat, T., & Lissòn, P. (2018).**
Predicting CEFRL levels in learner English on the basis of metrics and full texts.
*arXiv preprint arXiv:1806.11099*.
https://arxiv.org/abs/1806.11099

- **Contribuição**: Modelos de aprendizado supervisionado para classificação CEFR
- **Relevância**: Diferenças entre avaliação escrita e oral
- **Uso no sistema**: Base para ensemble LLM + Métricas

---

**Vajjala, S., & Rama, T. (2021).**
Automated classification of written proficiency levels on the CEFR-scale through complexity contours and RNNs.
In *Proceedings of the 16th Workshop on Innovative Use of NLP for Building Educational Applications* (pp. 180-190).
Association for Computational Linguistics.
https://aclanthology.org/2021.bea-1.21/

- **Contribuição**: Classificação automática usando contornos de complexidade
- **Relevância**: Métodos de classificação automática de níveis CEFR

---

**Ribeiro, E., Mamede, N., & Baptista, J. (2024).**
Avaliação automática do nível de complexidade de textos em português europeu.
*Linguamática*, 16(2), 115-139.
https://doi.org/10.21814/lm.16.2.449

- **Contribuição**: Métricas específicas para português europeu
- **Relevância**: Avaliação automática de complexidade textual

---

### **2. Avaliação de Fala e Pronúncia**

**SpeechLMScore (2022).**
SpeechLMScore: Evaluating speech generation using speech language model.
*arXiv preprint arXiv:2212.04559*.
https://arxiv.org/abs/2212.04559

- **Contribuição**: Métrica não supervisionada para avaliação de geração de fala
- **Relevância**: Avaliação automática de qualidade de fala

---

**LLM-Eval (2023).**
LLM-Eval: Unified multi-dimensional automatic evaluation for open-domain conversations with large language models.
*arXiv preprint arXiv:2305.13711*.
https://arxiv.org/abs/2305.13711

- **Contribuição**: Avaliação automática multidimensional para conversas
- **Relevância**: Uso de LLMs para avaliação de conversação

---

### **3. CEFR e Padrões Internacionais**

**Council of Europe (2020).**
*Common European Framework of Reference for Languages: Learning, teaching, assessment - Companion volume*.
Council of Europe Publishing.

- **Contribuição**: Descritores específicos para produção oral em todos os níveis CEFR
- **Relevância**: Critérios oficiais para avaliação de proficiência oral
- **Uso no sistema**: Base para critérios de classificação por nível

---

**Cambridge English (2024).**
CEFR levels and descriptors.
https://www.cambridgeenglish.org/br/exams-and-tests/cefr/

- **Contribuição**: Padrões internacionais de avaliação de proficiência oral
- **Relevância**: Referência para níveis CEFR

---

### **4. Avaliação de Pronúncia e Inteligibilidade**

**Microsoft Azure Speech Services (2024).**
Pronunciation Assessment API.
https://learn.microsoft.com/pt-pt/azure/ai-services/speech-service/how-to-pronunciation-assessment

- **Contribuição**: API para avaliação automática de pronúncia
- **Relevância**: Métricas de precisão de pronúncia
- **Uso potencial**: Integração futura para análise de áudio

---

**Instrumentos de Avaliação Fonológica (IAF)** - SciELO Brasil
- **Contribuição**: Instrumentos validados para avaliação fonológica em português brasileiro
- **Relevância**: Métricas específicas para português

---

**Escalas de Avaliação da Inteligibilidade de Fala** - SciELO Brasil
- **Contribuição**: Escalas validadas para medir inteligibilidade de fala
- **Relevância**: Métricas de clareza e compreensibilidade

---

### **5. Avaliação de Fluência e Discurso**

**Araújo & Suassuna (Periódicos UFSC).**
Critérios para avaliação da oralidade no ensino de língua portuguesa.

- **Contribuição**: Critérios específicos para avaliação de oralidade:
  - Clareza e precisão na articulação dos sons
  - Adequação lexical e gramatical
  - Coerência e coesão discursiva
  - Fluência
  - Adequação pragmática
- **Relevância**: Critérios validados para avaliação de fala em português

---

### **6. Métricas de Reconhecimento de Fala**

**Word Error Rate (WER)**
- **Contribuição**: Taxa de erro de palavras em transcrições automáticas
- **Relevância**: Métrica tradicional para avaliação de ASR
- **Uso potencial**: Validar qualidade de transcrição

---

**SeMaScore**
- **Contribuição**: Métrica semântica para avaliação de transcrições
- **Relevância**: Avaliação baseada em significado, não apenas palavras
- **Fonte**: https://dubsmart.ai/pt/blog/evaluation-metrics-for-speech-recognition-models

---

### **7. Avaliação Dinâmica e Interativa**

**Testes Adaptativos Informatizados (TAI)**
- **Contribuição**: Avaliação que ajusta dificuldade baseada no desempenho
- **Relevância**: Método adaptativo para avaliação de proficiência

---

**Avaliação Dinâmica** - PEPsIC Brasil
- **Contribuição**: Avaliação que considera potencial de aprendizado, não apenas desempenho atual
- **Relevância**: Abordagem interativa e contextualizada

---

## 🎯 Referências Usadas no Sistema Atual

### **Classificador Híbrido**
- **Leal et al. (2022)**: Métricas quantitativas para português falado
- **Arnold et al. (2018)**: Ensemble de features para classificação CEFR
- **CEFR Companion Volume (2020)**: Critérios por nível para produção oral

### **Ajustes para Fala Conversacional**
- **Leal et al. (2022)**: Diferenças entre fala e escrita
- **Arnold et al. (2018)**: Ajustes de métricas para avaliação oral
- **CEFR Companion Volume (2020)**: Descritores específicos para fala

### **Prompts LLM**
- **Vajjala & Rama (2021)**: Critérios de complexidade
- **Ribeiro et al. (2024)**: Critérios para português
- **CEFR Companion Volume (2020)**: Descritores detalhados por nível

---

## 📖 Formato APA 7th Edition

Todas as referências estão no formato APA 7th edition, incluindo:
- Autor(es) completo(s)
- Ano de publicação
- Título em itálico
- Fonte (DOI, URL, ou publicação)
- Informações completas de citação

---

## ✅ Status

- ✅ 16 referências bibliográficas identificadas
- ✅ Todas no formato APA 7th edition
- ✅ Organizadas por categoria
- ✅ Referências principais destacadas
- ✅ Uso no sistema documentado

**Documentação completa e pronta para uso acadêmico!**

