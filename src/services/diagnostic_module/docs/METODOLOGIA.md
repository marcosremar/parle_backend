## Fundamentação metodológica do `diagnostic_module`

Este documento descreve, em texto contínuo, como os principais trabalhos acadêmicos embasam a metodologia do `diagnostic_module` para avaliação de proficiência oral em níveis CEFR (A1–C2), com foco em **fala espontânea**, **complexidade linguística** e **avaliação multi-aspecto**. As referências completas em formato **APA** são apresentadas ao final.

---

## 1. CEFR como eixo normativo e a noção de complexidade

O ponto de partida conceitual do módulo é o *Companion Volume* do CEFR (Council of Europe, 2020), que fornece descritores detalhados de produção oral para cada nível de A1 a C2. Esses descritores orientam dois elementos centrais do serviço:  
(a) os **prompts dados ao LLM** quando pedimos uma estimativa de nível, e  
(b) a **leitura pedagógica** dos indicadores numéricos produzidos pelos analisadores de complexidade (por exemplo, quando um certo grau de subordinação e extensão frasal é mais típico de B1 ou de C1).  

No plano quantitativo, seguimos a tradição de tratar o nível CEFR como uma função de métricas de complexidade. Arnold et al. (2018) mostram que combinações de métricas lexicais, sintáticas e de comprimento textual permitem prever com boa acurácia níveis CEFR em redações de aprendizes. Vajjala e Rama (2021) acrescentam a ideia de *complexity contours*, enfatizando que a evolução da complexidade ao longo do texto é tão informativa quanto os valores médios. Esses trabalhos motivam nosso uso de múltiplas métricas e contornos em vez de indicadores isolados.  

Para português, Ribeiro et al. (2024) e, sobretudo, Leal et al. (2022) (NILC-Metrix) fornecem um arsenal de métricas específicas para português escrito e falado. Na prática, o `diagnostic_module` herda dessa linha de pesquisa a decisão de:  
- combinar **métricas sintáticas** (profundidade de dependência, subordinação, T-units),  
- **métricas lexicais** (diversidade, raridade, hapax, etc.) e  
- medidas agregadas de **densidade e comprimento**,  
e depois normalizar/ajustar esses valores para o regime de fala espontânea, em vez de aplicar diretamente thresholds concebidos para escrita.

---

## 2. Da escrita para a fala: ajuste para fala espontânea e proficiência oral

Embora Arnold et al. (2018) e Vajjala e Rama (2021) trabalhem principalmente com textos escritos, o nosso foco é fala. A ponte entre esses dois mundos é feita principalmente por NILC-Metrix (Leal et al., 2022), que explicitamente trata **português falado** e discute diferenças sistemáticas entre fala e escrita (frases mais curtas, mais disfluências, mais repetições). Isso justifica os módulos do `diagnostic_module` que:  
- recalibram expectativas de comprimento de frase e densidade sintática,  
- toleram maior fragmentação frasal,  
- e tratam disfluências como sinal de fluência/planejamento e não apenas “ruído”.  

Além disso, trabalhos da tradição de avaliação de oralidade em português (Araújo & Suassuna, s.d.) e estudos sobre as escalas orais do CELPE-Bras ajudam a traduzir descritores gerais do CEFR para critérios mais concretos como **clareza articulatória**, **adequação lexical e gramatical**, **fluência**, **coesão** e **adequação pragmática**. Esses critérios aparecem nas rubricas internas do módulo e na forma como explicamos, em linguagem natural, o porquê de um texto ser classificado como B1, B2 ou C1.

---

## 3. Avaliação de fala espontânea com dados reais

Para ancorar a avaliação em dados de fala real, usamos como referência o trabalho de Banno et al. (2022), que avalia proficiência oral em inglês L2 no exame Linguaskill combinando **wav2vec 2.0**, um *standard grader* com *hand-crafted features* e um grader baseado em BERT. O resultado são correlações em torno de 0,93–0,94 com as notas humanas e mais de 80% das predições dentro de meio nível de diferença, o que estabelece um patamar de desempenho realista para sistemas automatizados.  

Nesse sentido, Banno et al. (2022) nos orienta em dois pontos:  
1. A importância de separar **partes/tarefas diferentes** (respostas curtas espontâneas, leitura em voz alta, monólogos longos, descrição de gráficos), algo que inspirou nossa separação por tipo de tarefa e cenário.  
2. A evidência de que **fala lida** e **fala espontânea longa** se comportam de modos distintos em termos de erro e correlação, reforçando a necessidade de tratar cada tipo de input com expectativas e métricas adequadas.  

Já Bhat e Yoon (2015) focam na **complexidade sintática em fala espontânea**, propondo métricas e abordagens específicas para scoring desse tipo de material. Esse trabalho legitima o uso de métricas como profundidade de árvore de dependência, razão de orações por T-unit e índices de subordinação diretamente sobre transcrições de fala, desde que acompanhadas de ajustes para disfluências e fragmentação.

---

## 4. CEFR, fala e textos conversacionais para LLMs

Para o recorte **CEFR + fala/conversação**, dois trabalhos são particularmente relevantes para o `diagnostic_module`. O primeiro é EvalYaks (Scaria et al., 2024), que mostra como *instruction tuning* e LoRA sobre um modelo de 7B parâmetros podem produzir um avaliador altamente preciso (≈96% de acurácia aceitável) para a seção de fala B2 do CEFR, ainda que em cima de dados sintéticos. O valor para nós é metodológico:  
- decompor a avaliação em múltiplos critérios (gramática, vocabulário, gestão de discurso, interação);  
- alinhar explicitamente prompts e decisões de modelo com rubricas CEFR.  

O segundo é Ace-CEFR (Kogan et al., 2025), que introduz um dataset de textos conversacionais rotulados em níveis de dificuldade CEFR. Embora trabalhe principalmente com texto escrito conversacional, o trabalho é crucial para pensar **textos de diálogo curtos** como objeto de medição de dificuldade e não apenas como “respostas isoladas”. Essa visão casa diretamente com o nosso uso de trechos de conversas (multi-turno) em avaliações tanto off-line (tests/e2e) quanto em produção.

---

## 5. Avaliação automática multi-aspecto e diálogo

O `diagnostic_module` adota uma visão **multi-aspecto** da proficiência, separando pelo menos três eixos:  
- **delivery/fluência** (ritmo, pausas, disfluências, extensão frasal),  
- **uso da língua** (complexidade gramatical e lexical, perfil de erros),  
- **conteúdo/relevância da tarefa** (até que ponto a resposta cumpre a tarefa proposta).  

Essa decomposição é diretamente inspirada em trabalhos de avaliação de fala automática multi-aspecto e de avaliação de diálogo. Em particular, Lu et al. (2025) propõem um modelo híbrido que introduz:  
1. um módulo de **relevância multifacetada**, que combina pergunta, imagem, exemplar e resposta para medir a adequação do conteúdo;  
2. um vetor de **erros gramaticais finos**, obtido via GEC e anotadores como SERRANT, para representar o perfil de erros por tipo.  

No nosso contexto, não reproduzimos a stack completa (Phi-4, Long-CLIP, SERRANT), mas reutilizamos a ideia de representar **relevância de tarefa** e **perfil de erros categorizados** como *features explícitas* usadas tanto na decisão de nível quanto na geração de feedback pedagógico.

Trabalhos como DynaEval, CF-LSTM e ACUTE-EVAL (Li et al., 2019) contribuem mais no plano conceitual, ao tratar avaliação de diálogo como um problema que considera **multi-turnos, contexto e comparações relativas**. Isso reforça a importância de:  
- analisar sessões inteiras (não só turns isolados),  
- olhar para tendências (melhora/piora ao longo da sessão),  
- e usar comparações internas (entre respostas do mesmo aluno, entre cenários) para calibrar decisões.

---

## 6. Síntese para a metodologia do `diagnostic_module`

Em resumo, a metodologia implementada (e planejada) no `diagnostic_module` combina:

- **Normas CEFR e escalas oficiais**:  
  Descritores do CEFR (Council of Europe, 2020), materiais de Cambridge e estudos sobre CELPE-Bras orientam os critérios de nível e o vocabulário usado nos relatórios.

- **Métricas quantitativas de complexidade**:  
  A linha Arnold et al. (2018), Vajjala e Rama (2021), Leal et al. (2022) e Ribeiro et al. (2024) fundamenta o uso de dezenas de métricas lexicais e sintáticas, ajustadas para fala espontânea.

- **Evidência de avaliação de fala com dados reais**:  
  Banno et al. (2022) e Bhat e Yoon (2015) mostram que é possível obter boa correlação com avaliadores humanos usando representações de áudio e métricas sintáticas especializadas para fala, reforçando nossa aposta em métricas específicas de fala + LLM.

- **Modelos e datasets CEFR para fala/conversação**:  
  EvalYaks (Scaria et al., 2024) e Ace-CEFR (Kogan et al., 2025) indicam caminhos para alinhar diretamente LLMs e datasets de conversação ao espaço CEFR, o que inspira tanto nossos scripts de geração de conversas quanto os testes de validação.

- **Arquiteturas multi-aspecto**:  
  Lu et al. (2025), DynaEval, CF-LSTM e ACUTE-EVAL motivam a separação explícita de aspectos (delivery, language use, content) e o uso de perfis mais ricos de erros e relevância, que estamos incorporando progressivamente ao serviço.

Assim, cada decisão técnica no `diagnostic_module` (quais features calcular, como pesar aspectos, como estruturar relatórios) é ancorada em um conjunto específico de trabalhos, garantindo que a avaliação produzida seja **alinhada com a pesquisa atual em avaliação de fala e CEFR**, e não apenas baseada em heurísticas ad hoc.

---

## 7. Escalas orais, Celpe-Bras e avaliação analítica

Além da literatura de complexidade e de modelagem automática, a linha de pesquisa em torno do **Celpe-Bras** e de exames orais de português/inglês fornece uma base sólida para a definição de **dimensões analíticas** no `diagnostic_module`. Estudos psicométricos sobre as escalas orais do Celpe-Bras, como o trabalho de dimensionalidade das escalas de proficiência oral (Ferreira, 2020) e a análise dos componentes da habilidade oral (por exemplo, compreensão, competência interacional, fluência, adequação lexical, adequação gramatical e pronúncia) discutidos em artigos da área de avaliação em línguas adicionais, mostram que é possível decompor a nota global em subescalas com boa consistência interna e poder discriminativo.  

Esses resultados dialogam diretamente com pesquisas que comparam escalas **holísticas (tipo CEFR)** com **escalas analíticas** específicas para pronúncia, gramática, vocabulário e fluência em inglês L2, evidenciando alta concordância entre avaliadores experientes, mas também ganhos de transparência e feedback quando se adota uma abordagem analítica. Trabalhos sobre **competência interacional** como critério na produção oral e sobre **precisão gramatical** em exames profissionais de língua estrangeira reforçam a ideia de que aspectos como uso de estratégias comunicativas, gestão de turnos, precisão morfossintática e adequação lexical devem aparecer explicitamente na rubrica.  

No `diagnostic_module`, essa literatura sustenta:  
- a escolha de dimensões analíticas (compreensão do enunciado, competência interacional, fluência, léxico, gramática, pronúncia) que podem ser refletidas em campos separados de saída;  
- a decisão de alinhar essas dimensões tanto com descritores CEFR quanto com as escalas orais de exames de referência (Celpe-Bras, EPPLE, etc.);  
- e o desenho de relatórios que não apresentam apenas um “nível CEFR final”, mas um **perfil multi-dimensional** da produção oral, em linha com o que os estudos de escalas analíticas apontam como boas práticas de avaliação.

---

## 8. Implementação prática no serviço `speech_grader`

Na prática, toda essa fundamentação se concretiza no serviço `speech_grader` (antigo `diagnostic_module`), localizado em `src/services/diagnostic_module/`. A seguir, descrevemos, em alto nível, como implementar os componentes principais na aplicação.

- **Entradas principais**  
  - transcrição de fala do aluno (`user_text`), opcionalmente com o texto da IA (`ai_text`);  
  - contexto de tarefa (pergunta, cenário, exemplar de resposta) vindo do `orchestrator`/`pedagogical_policy`;  
  - informações de progresso e nível previstos pelo `student_model` (AKT);  
  - lista opcional de skills-alvo (`valid_skills`) para amarrar análise à trilha pedagógica.  

- **Passo 1 – Extração de métricas linguísticas (NILC, Arnold, Bhat, Vajjala)**  
  - No `speech_grader`, o analisador de complexidade chama o serviço `linguistic_analysis` para obter:  
    - métricas sintáticas (profundidade de dependência, subordinação, T-units, Yngve/Frazier);  
    - métricas lexicais (MTLD, MATTR, raridade, hapax);  
    - métricas discursivas/coesão (conectores, correferência, etc.).  
  - Esses valores são ajustados para fala espontânea (limiares diferentes de escrita).

- **Passo 2 – Perfil de erros gramaticais via LLM (Lu 2025)**  
  - Em `grammar_analyzer`, implementar uma rota/função que:  
    - solicita ao LLM (via `llm_client`) a correção do texto do aluno e a listagem de erros por tipo (verbo, concordância, morfologia, ordem, etc.);  
    - converte essa lista em um vetor numérico normalizado (erros por 100 palavras + distribuição por tipo).  
  - Esse vetor é armazenado no campo `grammar_error_profile` da resposta.

- **Passo 3 – Relevância da tarefa e similaridade com exemplar (Lu 2025, Ace-CEFR)**  
  - Criar um `task_relevance_analyzer` que recebe pergunta, contexto, resposta e, se houver, um exemplar de boa resposta:  
    - o LLM julga quão bem a resposta cumpre a tarefa (score 0–1) e explica rapidamente (“respondeu parcialmente à pergunta, não menciona X…”);  
    - se houver exemplar, o LLM também estima a similaridade de conteúdo/estratégia discursiva (0–1).  
  - Esses scores alimentam campos como `task_relevance` e `exemplar_similarity`.

- **Passo 4 – Julgamento multi-aspecto por LLM (EvalYaks, CEFR, Celpe-Bras)**  
  - Em `complexity_analyzer`, além das métricas numéricas, chamar o LLM com uma rubrica que combine:  
    - descritores CEFR (Council of Europe, 2020);  
    - critérios analíticos (Celpe-Bras, Araújo & Suassuna);  
    - critérios de EvalYaks (gramática, vocabulário, discurso, interação).  
  - O LLM retorna um nível estimado e justificativa por aspecto (fluência, gramática, vocabulário, interação, conteúdo), preenchendo um campo `breakdown` na resposta.

- **Passo 5 – Núcleo híbrido de decisão (Arnold, Banno, Ace-CEFR)**  
  - Combinar, em uma função central, três fontes:  
    - métricas quantitativas (features numéricas);  
    - julgamentos LLM por aspecto;  
    - task relevance + grammar error profile.  
  - Inicialmente, essa combinação pode ser por pesos calibrados manualmente (inspirados em Arnold, Banno); em uma fase posterior, pode-se treinar um modelo leve (regressão / ranking) usando dados rotulados (conversas anotadas com níveis CEFR).

- **Passo 6 – Integração com o `student_model` (AKT) e o `orchestrator`**  
  - O `speech_grader` expõe endpoints como `/api/diagnostic/estimate_level` e `/api/diagnostic/analyze_turn`;  
  - o `orchestrator` chama esses endpoints a cada turno/conversa, passando contexto;  
  - o `student_model` recebe o nível/indicadores e ajusta o estado de domínio (AKT), que por sua vez realimenta a confiança do próprio `speech_grader` (quando há grande divergência entre texto e histórico, a confiança do diagnóstico cai e pode acionar revisão humana).

Implementando esses passos de forma incremental, o serviço `speech_grader` deixa de ser apenas um analisador de complexidade textual e passa a operar como um **avaliador multi-aspecto de fala**, alinhado aos descritores CEFR, às escalas Celpe-Bras e às melhores práticas recentes em avaliação automática com LLMs.

---

## Referências (formato APA)

Arnold, T., Ballier, N., Gaillat, T., & Lissón, P. (2018). Predicting CEFRL levels in learner English on the basis of metrics and full texts. *arXiv preprint arXiv:1806.11099*.  

Araújo, V., & Suassuna, L. (s.d.). Critérios para avaliação da oralidade no ensino de língua portuguesa. *Revista Fórum Linguístico*. Universidade Federal de Santa Catarina.  

Banno, F., Gales, M., Kyriakopoulos, K., Malinin, A., van Dalen, R., Wang, Y., & Rashid, M. (2022). L2 Proficiency Assessment Using Self-Supervised Speech Representations. *arXiv preprint arXiv:2211.08849*.  

Bhat, S., & Yoon, S. (2015). Automatic assessment of syntactic complexity for spontaneous speech scoring. *Speech Communication, 67*, 42–57.  

Council of Europe. (2020). *Common European Framework of Reference for Languages: Learning, teaching, assessment – Companion volume*. Council of Europe Publishing.  

Kogan, D., Schumacher, M., Nguyen, S., Suzuki, M., Smith, M., Bellows, C. S., & Bernstein, J. (2025). Ace-CEFR: A Dataset for Automated Evaluation of the Linguistic Difficulty of Conversational Texts for LLM Applications. Manuscrito em preparação / preprint.  

Leal, S. E., Duran, M. S., Scarton, C. E., Hartmann, N. S., & Aluísio, S. M. (2022). NILC-Metrix: Assessing the complexity of written and spoken language in Brazilian Portuguese. *arXiv preprint arXiv:2201.03445*.  

Li, J., Galley, M., Brockett, C., Gao, J., & Dolan, B. (2019). ACUTE-EVAL: Improved Dialogue Evaluation with Optimized Questions and Multi-turn Comparisons. In *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics* (pp. 6235–6245). Association for Computational Linguistics.  

Lu, H.-C., Lin, J.-K., Lin, H.-Y., Wang, C.-C., & Chen, B. (2025). Advancing Automated Speaking Assessment Leveraging Multifaceted Relevance and Grammar Information. *arXiv preprint arXiv:2506.16285*.  

Ribeiro, E., Mamede, N., & Baptista, J. (2024). Avaliação automática do nível de complexidade de textos em português europeu. *Linguamática, 16*(2), 115–139.  

Scaria, N., Kennedy, S. J. J., Latinovich, T., & Subramani, D. (2024). EvalYaks: Instruction Tuning Datasets and LoRA Fine-tuned Models for Automated Scoring of CEFR B2 Speaking Assessment Transcripts. Manuscrito em preparação / preprint.  

Vajjala, S., & Rama, T. (2021). Automated classification of written proficiency levels on the CEFR-scale through complexity contours and RNNs. In *Proceedings of the 16th Workshop on Innovative Use of NLP for Building Educational Applications* (pp. 180–190). Association for Computational Linguistics.  

Ferreira, L. M. L. (2020). Um estudo sobre a dimensionalidade das escalas de avaliação da proficiência oral do Certificado de Proficiência em Língua Portuguesa para Estrangeiros. *Estudos de Psicologia (Campinas)*, 37, e200010.  

Artigos sobre componentes da habilidade oral e escalas analíticas em exames como Celpe-Bras e EPPLE (por exemplo, trabalhos em periódicos de avaliação em línguas adicionais que analisam compreensão, competência interacional, fluência, adequação lexical, adequação gramatical e pronúncia).  

