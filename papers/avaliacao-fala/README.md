# Papers sobre Avaliação de Linguagem Falada (Speech Assessment)

Esta pasta contém papers acadêmicos sobre avaliação de linguagem falada, especialmente relacionados a:
- Classificação automática de níveis CEFR
- Métricas de complexidade para português falado
- Avaliação de pronúncia e fluência
- Uso de LLMs para avaliação de fala

---

## 📚 Papers Disponíveis

### 1. **NILC-Metrix (2022)**
- **Arquivo**: `nilc_metrix_2022.pdf` / `nilc_metrix_2022.md`
- **Título**: NILC-Metrix: Assessing the complexity of written and spoken language in Brazilian Portuguese
- **Autores**: Leal, S. E., Duran, M. S., Scarton, C. E., Hartmann, N. S., & Aluísio, S. M.
- **Fonte**: arXiv:2201.03445
- **Relevância**: Sistema com 200 métricas para português brasileiro (escrito e falado)
- **Uso no sistema**: Base para métricas quantitativas de complexidade

### 2. **Arnold et al. (2018)**
- **Arquivo**: `arnold_cefr_2018.pdf` / `arnold_cefr_2018.md`
- **Título**: Predicting CEFRL levels in learner English on the basis of metrics and full texts
- **Autores**: Arnold, T., Ballier, N., Gaillat, T., & Lissòn, P.
- **Fonte**: arXiv:1806.11099
- **Relevância**: Modelos de aprendizado supervisionado para classificação CEFR
- **Uso no sistema**: Base para ensemble LLM + Métricas

### 3. **Vajjala & Rama (2021)**
- **Arquivo**: `vajjala_rama_2021.pdf` / `vajjala_rama_2021.md`
- **Título**: Automated classification of written proficiency levels on the CEFR-scale through complexity contours and RNNs
- **Autores**: Vajjala, S., & Rama, T.
- **Fonte**: ACL Anthology 2021.bea-1.21
- **Relevância**: Classificação automática usando contornos de complexidade e RNNs
- **Uso no sistema**: Métodos de classificação automática de níveis CEFR

### 4. **SpeechLMScore (2022)**
- **Arquivo**: `speech_lm_score_2022.pdf` / `speech_lm_score_2022.md`
- **Título**: SpeechLMScore: Evaluating speech generation using speech language model
- **Fonte**: arXiv:2212.04559
- **Relevância**: Métrica não supervisionada para avaliação de geração de fala
- **Uso no sistema**: Referência para avaliação automática de qualidade de fala

### 5. **LLM-Eval (2023)**
- **Arquivo**: `llm_eval_2023.pdf` / `llm_eval_2023.md`
- **Título**: LLM-Eval: Unified multi-dimensional automatic evaluation for open-domain conversations with large language models
- **Fonte**: arXiv:2305.13711
- **Relevância**: Avaliação automática multidimensional para conversas
- **Uso no sistema**: Referência para uso de LLMs na avaliação de conversação

---

## 🔄 Como Baixar e Converter Novos Papers

Execute o script `download_and_convert.py`:

```bash
python3 papers/avaliacao-fala/download_and_convert.py
```

O script:
1. Baixa PDFs de URLs ou arXiv IDs
2. Converte automaticamente para Markdown usando `pdf4llm`
3. Salva ambos os formatos na pasta

---

## 📖 Formato

- **PDFs**: Formato original dos papers
- **Markdown**: Conversão usando `pdf4llm` para fácil leitura e processamento

---

## 🔗 Referências Completas (APA 7th Edition)

1. **Leal, S. E., Duran, M. S., Scarton, C. E., Hartmann, N. S., & Aluísio, S. M. (2022).** NILC-Metrix: Assessing the complexity of written and spoken language in Brazilian Portuguese. *arXiv preprint arXiv:2201.03445*.

2. **Arnold, T., Ballier, N., Gaillat, T., & Lissòn, P. (2018).** Predicting CEFRL levels in learner English on the basis of metrics and full texts. *arXiv preprint arXiv:1806.11099*.

3. **Vajjala, S., & Rama, T. (2021).** Automated classification of written proficiency levels on the CEFR-scale through complexity contours and RNNs. In *Proceedings of the 16th Workshop on Innovative Use of NLP for Building Educational Applications* (pp. 180-190). Association for Computational Linguistics.

4. **SpeechLMScore (2022).** SpeechLMScore: Evaluating speech generation using speech language model. *arXiv preprint arXiv:2212.04559*.

5. **LLM-Eval (2023).** LLM-Eval: Unified multi-dimensional automatic evaluation for open-domain conversations with large language models. *arXiv preprint arXiv:2305.13711*.

---

## ✅ Status

- ✅ 5 papers baixados
- ✅ 5 PDFs convertidos para Markdown
- ✅ Todos os arquivos organizados na pasta `papers/avaliacao-fala/`

