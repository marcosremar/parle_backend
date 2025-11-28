# Papers sobre Avaliação CEFR de Fala Conversacional - v2

Este diretório contém papers acadêmicos sobre avaliação automática de fala e classificação CEFR, baixados e convertidos para Markdown.

---

## 📚 Papers Disponíveis

### 1. EvalYaks (2024)
- **Arquivo:** `EvalYaks_2024.pdf` / `EvalYaks_2024.md`
- **Referência:** [arXiv:2408.12226](https://arxiv.org/abs/2408.12226)
- **Título:** "Instruction Tuning Datasets and LoRA Fine-tuned Models for Automated Scoring of CEFR B2 Speaking Assessment Transcripts"
- **Metodologia:** LoRA fine-tuning de Mistral 7B, instruction tuning, datasets sintéticos
- **Resultados:** 96% de precisão na avaliação de transcrições B2

### 2. UniversalCEFR (2024)
- **Arquivo:** `UniversalCEFR_2024.pdf` / `UniversalCEFR_2024.md`
- **Referência:** [arXiv:2506.01419](https://arxiv.org/abs/2506.01419)
- **Título:** "Enabling Open Multilingual Research on Language Proficiency Assessment"
- **Metodologia:** 500k+ textos anotados CEFR, abordagem híbrida (features, fine-tuning, prompting)
- **Resultados:** Suporte para 13 idiomas, padronização de formatos

### 3. Arnold et al. (2018)
- **Arquivo:** `Arnold_2018.pdf` / `Arnold_2018.md`
- **Referência:** [arXiv:1806.11099](https://arxiv.org/abs/1806.11099)
- **Título:** "Predicting CEFRL levels in learner English on the basis of metrics and full texts"
- **Metodologia:** Métricas lexicais e sintáticas, pairwise classification, Gradient Boosted Trees
- **Resultados:** AUC 0.916 (A1→A2), AUC 0.904 (A2→B1)

### 4. NILC-Metrix (2022)
- **Arquivo:** `NILC-Metrix_2022.pdf` / `NILC-Metrix_2022.md`
- **Referência:** [arXiv:2201.03445](https://arxiv.org/abs/2201.03445)
- **Título:** "Assessing the complexity of written and spoken language in Brazilian Portuguese"
- **Metodologia:** 200 métricas de complexidade, análise multi-dimensional
- **Resultados:** Métricas específicas para português brasileiro, distinção fala vs escrita

### 5. DynaEval (2021)
- **Arquivo:** `DynaEval_2021.pdf` / `DynaEval_2021.md`
- **Referência:** [arXiv:2106.01112](https://arxiv.org/abs/2106.01112)
- **Título:** "Unifying Turn and Dialogue Level Evaluation"
- **Metodologia:** Avaliação holística, GCN (Graph Convolutional Networks), modelagem de interações
- **Resultados:** Forte correlação com avaliações humanas, avaliação unificada

### 6. CF-LSTM (2023)
- **Arquivo:** `CF-LSTM_2023.pdf` / `CF-LSTM_2023.md`
- **Referência:** [arXiv:2301.13372](https://arxiv.org/abs/2301.13372)
- **Título:** "Improving Open-Domain Dialogue Evaluation with a Causal Inference Model"
- **Metodologia:** Inferência causal, CF-LSTM, identificação de fatores influenciadores
- **Resultados:** Melhor precisão em diálogos abertos, explicabilidade melhorada

### 7. ACUTE-EVAL (2019)
- **Arquivo:** `ACUTE-EVAL_2019.pdf` / `ACUTE-EVAL_2019.md`
- **Referência:** [arXiv:1909.03087](https://arxiv.org/abs/1909.03087)
- **Título:** "Improved Dialogue Evaluation with Optimized Questions and Multi-turn Comparisons"
- **Metodologia:** Comparação de diálogos completos, perguntas otimizadas, julgamentos comparativos
- **Resultados:** Maior robustez entre avaliadores, avaliações mais consistentes

### 8. Mohammadi et al. (2025) ⭐ NOVO
- **Arquivo:** `Mohammadi_2025.pdf` / `Mohammadi_2025.md`
- **Referência:** [arXiv:2505.02615](https://arxiv.org/abs/2505.02615)
- **Título:** "Automatic Proficiency Assessment in L2 English Learners"
- **Metodologia:** Deep learning multimodal (wav2vec 2.0 + BERT), análise de áudio e transcrições
- **Datasets:** EFCAMDAT, ANGLISH (100% dados reais)
- **Resultados:** 85% accuracy, correlação 0.82 com avaliações humanas
- **Aspectos avaliados:** Fluência, precisão gramatical, pronúncia, coerência do discurso, complexidade lexical, adequação pragmática

---

## 📊 Estatísticas

- **Total de papers:** 8
- **Formato:** PDF + Markdown
- **Tamanho total:** ~9.5 MB (PDFs)
- **Período:** 2018-2025
- **Foco principal:** Avaliação automática de fala e classificação CEFR

---

## 🔧 Conversão

Todos os PDFs foram convertidos para Markdown usando `pdf4llm`:

```python
import pdf4llm
markdown_text = pdf4llm.to_markdown("paper.pdf")
```

---

## 📝 Notas

- Os papers foram baixados diretamente do arXiv
- Conversão para Markdown preserva estrutura e formatação
- Arquivos Markdown podem ser editados e processados facilmente
- Referências completas disponíveis em cada arquivo

---

## 🔗 Links Úteis

- **Documentação completa:** `docs/BEST_PRACTICES_SPEECH_CEFR_ASSESSMENT.md`
- **Papers adicionais:** `docs/ADDITIONAL_PAPERS_CEFR_ASSESSMENT.md`
- **Melhorias recomendadas:** `docs/RESEARCH_BASED_IMPROVEMENTS.md`

---

**Última atualização:** 2025-11-23
