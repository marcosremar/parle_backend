FoLiBi (Forgetting-aware Linear Bias)
Por quê: speech-to-speech gera muitas interações; forgetting é crítico
Benefício: +2.58% AUC, especialmente em históricos longos
Esforço: 2-3 dias
Complexidade: baixa (fórmula simples)

Monotonic Attention Mechanism — refinar decay exponencial existente
Context-aware Distance — melhorar distância temporal





####
## Visão geral

Para um **MVP de speech-to-speech tutoring** que usa **AKT + CEFR (CECRL)**, os conceitos mais úteis são os que:

- Ajudam a mapear domínio de **skills linguísticas finas → níveis CEFR**
- São **simples o bastante** para implementar logo
- Dão **estado interpretável** para gerar feedback didático na conversa

Abaixo, os conceitos mais relevantes (em ordem de prioridade prática).

---

### 1. **Rasch Model-based Embeddings (AKT original)**

- **Por que é chave**: Rasch/IRT já é pensado para “dificuldade de item” e “habilidade do aluno”, o que encaixa bem com faixas CEFR.
- **Uso no MVP**:
  - Tratar cada `skill_id` (ex: `verb_conjugation_past`, `vocabulary_basic`) como um “item/conceito” com dificuldade.
  - Ancorar essas skills em **intervalos de CEFR** (A1…C2) via `SKILL_CEFR_MAP`.
  - Usar a probabilidade de acerto/mestre de cada skill como “micro-indicadores” que, agregados, dão o nível CEFR estimado (o que você já começou a fazer).

---

### 2. **Context-aware Representations (AKT)**

- **Por que é chave**: Speech-to-speech tem muito ruído (ASR, hesitações, correções). Precisa olhar **sequências**, não interações isoladas.
- **Uso no MVP**:
  - Garantir que o AKT considere **histórico de turnos** (não só o último) ao atualizar domínio de skill.
  - No seu `diagnostic_module`, sempre mandar um pequeno **janela de contexto** (últimos N turnos + cenário) para a análise de skills/erros.

---

### 3. **Monotonic Attention + Context-aware Relative Distance (AKT)**

- **Por que é chave**: CEFR + AKT só fazem sentido se o modelo respeitar **esquecimento ao longo do tempo** (intervalos entre práticas).
- **Uso no MVP**:
  - Usar (ou manter) a versão de AKT que aplica **decay exponencial** para interações antigas.
  - Aproveitar o `last_practiced` de cada skill e o tempo entre sessões para ajustar o **peso** das interações antigas na atualização de mastery.

*(FoLiBi melhora isso, mas dá pra deixar como **fase 2**; MVP funciona com o monotonic attention básico.)*

---

### 4. **Feature-rich / Sub-skill Knowledge State (Vocabulary KT 2017)**

- **Por que é chave**: CEFR é amplo (fluência, gramática, léxico). No MVP, você precisa de um estado que reflita **sub‑skills linguísticas**:
  - fonologia / prosódia
  - morfologia (conjugação, gênero, número)
  - sintaxe básica vs complexa
  - tipos de vocabulário (cotidiano, acadêmico, coloquial)
- **Uso no MVP**:
  - Refinar seu `skill_registry` para ter **subskills mais linguísticas**, não só “gramática genérica”.
  - Fazer o `diagnostic_module` taggear erros/acertos em nível de subskill (como o paper faz com features de sufixo, tempo, pessoa).
  - Agregar essas subskills por CEFR para ter **“CEFR por dimensão”** (ex: A2 em gramática, B1 em vocabulário oral).

---

### 5. **Interpretable Knowledge State for Feedback (Vocabulary KT + AKT)**

- **Por que é chave**: No speech-to-speech, o tutor tem que conseguir dizer:
  - “Você já está bem em X, mas ainda precisa praticar Y no nível B1.”
- **Uso no MVP**:
  - Expor na `student_model` uma visão simples: para cada CEFR e cada dimensão (lexical, gramatical, pronúncia), uma **probabilidade de domínio** + exemplos de skills fortes/fracas.
  - Usar isso no `pedagogical_policy` para:
    - ajustar a complexidade da fala (CEFR),
    - escolher foco (ex: mais perguntas que forcem o uso de passado, ou de conectores B1),
    - formular feedback explícito (“seus verbos no passado estão perto de B1, vamos consolidar mais um pouco”).

---

### 6. **Inductive Knowledge Tracing + LLM-based Semantic Encoding (SINKT)**

- **Por que é relevante**: Em um tutor de fala, você terá **novos prompts/cenários/frases** o tempo todo, não só um banco fixo de exercícios.
- **Uso no MVP (versão leve)**:
  - Em vez de IDs fixos, gerar embeddings semânticos para “tipo de tarefa” ou “scenario + prompt alvo” com o LLM.
  - Mapear esses embeddings para skills existentes (por ex., se o cenário exige `verb_conjugation_past` + `vocabulary_travel`).
  - Não precisa implementar SINKT completo; basta a **ideia**: usar o LLM para classificar / mapear novos enunciados em skills + CEFR.

---

### 7. **AKT como Ambiente para o RL (ALPN)**

- **Por que é futuro próximo**: Para o MVP, você pode só **logar dados**; RL vem depois.
- **Uso no MVP**:
  - Tratar o AKT + CEFR como “estado” já agora.
  - Estruturar seu `learning_path` de modo que, no futuro, seja trivial plugar um bandit ou PPO simples — mas sem precisar implementar já.

---

## Recomendações práticas para o seu MVP

- **Foque primeiro em**:
  1. `Rasch Model-based Embeddings` + `SKILL_CEFR_MAP` (CEFR confiável).
  2. `Context-aware Representations` + `Monotonic Attention` (sequência + esquecimento).
  3. `Feature-rich / Sub-skill State` + `Interpretable Knowledge State` (subskills linguísticos visíveis).

- **Depois**, em uma segunda iteração:
  - Adicionar ideias de **FoLiBi** (esquecimento mais fino).
  - Começar a usar a ideia de **SINKT light** (LLM para mapear novos prompts em skills/CEFR).

Se você quiser, posso desenhar o fluxo exato “voz → ASR → diagnóstico → AKT+CEFR → policy → TTS” marcando onde cada um desses conceitos entra no código que você já tem.