## Objetivo do modelo

- **Meta**: Treinar um LLM especializado para **detecção de skills linguísticas** e **features linguísticas** a partir do texto do aluno, alinhado ao nosso conjunto de ~189 skills em `SKILL_CEFR_MAP`.
- **Uso**: Substituir/aperfeiçoar o fluxo atual de:
  - `analyze_grammar` (identificação de `errors`, `correct_skills`, `linguistic_features`)
  - `extract_skills` (lista de skills usadas + confidence)

---

## Espaço de skills (rótulos)

- Fonte: `src/services/student_model/skill_registry.py` → `SKILL_CEFR_MAP`
- Níveis: `A1` a `C2`
- Tipos de skills:
  - Gramática (verbos, tempos, modos, artigos, preposições, estruturas complexas)
  - Vocabulário (domínios semânticos, registro, complexidade)
  - Pronúncia (apenas parcialmente observável via texto)
  - Interação (fala, gestão de conversa, troca de informações)
  - Produção (descrição, narrativa, argumentação)
  - Mediação (de textos, conceitos, comunicação)

- **Tarefa principal**: classificação **multi-label**:
  - Entrada: texto (1 turno ou trecho de conversa)
  - Saída: subconjunto das 189 skills que estão sendo utilizadas naquele texto

---

## Formato de dados de treino

### 1. Representação recomendada (JSONL)

Cada linha = 1 instância (tipicamente 1 turno do aluno, opcionalmente com contexto).

```json
{
  "id": "session123_turn05",
  "user_text": "Ontem eu fui ao cinema com meus amigos.",
  "ai_text": "Que legal! O que você assistiu?",
  "cefr_level_estimated": "A2",
  "skills_true": [
    "verb_conjugation_past",
    "vocabulary_daily_routine",
    "interaction_simple_questions"
  ],
  "linguistic_features_true": {
    "tense": "past",
    "person": "1st",
    "number": "singular",
    "register": "informal",
    "domain": "daily_life"
  },
  "context": {
    "previous_user_utterances": [
      "Eu moro em São Paulo.",
      "Eu gosto de sair com meus amigos."
    ],
    "previous_ai_utterances": [
      "Onde você mora?",
      "O que você gosta de fazer no tempo livre?"
    ]
  }
}
```

### 2. Variantes úteis

- **Turno isolado**: usar apenas `user_text` (cenário mais simples).
- **Janela de contexto**: incluir últimas 2–4 falas do aluno + AI para skills de interação/mediação.

---

## Diretrizes de anotação de skills

### 1. Regra geral

- Anotar **apenas** skills que:
  - São **observáveis diretamente** no texto
  - Têm evidência concreta (palavras, estruturas, função discursiva)
- Não anotar:
  - Capacidades globais (ex: “pode participar de qualquer conversa”) se não há evidência suficiente naquele trecho.

### 2. Exemplos de mapeamento

- Texto: `"Ontem eu fui ao banco sacar dinheiro."`
  - `verb_conjugation_past`
  - `vocabulary_daily_routine` ou `vocabulary_finance` (se existir)
  - `linguistic_features_true.tense = "past"`

- Texto: `"Eu acho que viajar é muito importante para aprender línguas."`
  - `vocabulary_abstract_concepts`
  - `vocabulary_travel`
  - Possivelmente `production_express_opinions` (B1+)

- Texto: `"Descreva a sua rotina diária de manhã até a noite."` (do professor)
  - Não anotar skills de aluno (é fala do tutor), pode ser usada apenas para contexto.

### 3. Níveis CEFR

- `cefr_level_estimated` pode vir:
  - Do próprio sistema (AKT + CEFR)
  - De anotadores humanos
  - De um modelo separado de nível global

Não precisa ser perfeito, mas ajuda a calibrar quais skills esperar por nível.

---

## Tarefa de modelagem

### 1. Objetivos

1. **Detecção de skills (multi-label)**  
   - Para cada skill `s` na lista de 189:
     - Predizer `p(s | texto)`  
   - Limite de decisão típico: `p > 0.5` ou calibrado por validação.

2. **Predição de features linguísticas estruturadas**
   - `tense`, `person`, `number`, `mood`, `register`, `domain`, etc.
   - Pode ser tratada como:
     - Multi-classe (ex: `tense ∈ {present, past, future, ...}`)
     - Ou extração estruturada via JSON + pós-processamento.

### 2. Arquiteturas possíveis

- **Opção 1 – Fine-tune encoder/decoder com cabeças específicas**:
  - Base: LLM encoder (ex: modelo encoder-only ou encoder de um LLM)
  - Cabeça 1: vetor de 189 logits → sigmoid (multi-label skills)
  - Cabeça 2+: classificadores de features (tense, person, register, etc.)

- **Opção 2 – Instruction-tuning LLM**:
  - Treinar o modelo via prompt + saída JSON padronizada.
  - Perda baseada em:
    - Cross-entropy token-level
    - + métricas extrínsecas (F1 de skills) para avaliação.

---

## Loss e métricas

### 1. Loss

- `L_total = L_skills + λ * L_features`
  - `L_skills`: Binary cross-entropy multi-label sobre 189 skills
  - `L_features`: cross-entropy para cada feature (tense, person, etc.)
  - `λ`: peso (ex: 0.5)

### 2. Métricas de avaliação

- **Por skill**:
  - Precision, Recall, F1
- **Macro/micro**:
  - F1 micro (todas as skills juntas)
  - F1 macro (média por skill → importante para skills raras)
- **Por categoria**:
  - Gramática vs. Vocabulário vs. Interação vs. Produção
- **Por nível CEFR**:
  - Desempenho em A1 vs. C2

---

## Pipeline de treinamento sugerido

1. **Coletar dados**
   - Exportar transcrições reais do sistema (com consentimento e anonimização).
   - Gerar subset inicial anotado por humanos (1000–5000 turnos).

2. **Definir guidelines de anotação**
   - Documento com:
     - Definição de cada skill
     - Exemplos positivos e negativos
     - Casos ambíguos e como tratar

3. **Anotar manualmente**
   - Ferramenta simples (ex: interface web) para marcar:
     - `skills_true`
     - `linguistic_features_true`

4. **Pré-treino / Fine-tune**
   - Usar modelo base (ex: LLaMA, Qwen, etc.)
   - Treinar em batches com:
     - Input = prompt estruturado + texto
     - Output = JSON de skills + features

5. **Validação**
   - Separar conjunto de teste por:
     - Nível CEFR
     - Tipo de tarefa (diálogo livre, exercício focado, etc.)

6. **Calibração**
   - Ajustar thresholds de decisão para cada skill:
     - Ex: `verb_conjugation_past` pode ter threshold 0.6
     - `interaction_fluent_conversation` pode precisar >0.8

7. **Integração no sistema**
   - Substituir/estender:
     - `DiagnosticLLMClient.analyze_grammar`
     - `DiagnosticLLMClient.extract_skills`
   - Garantir compatibilidade de formato com:
     - `AnalyzeTurnResponse`
     - `ExtractSkillsResponse`

---

## Estratégia incremental (MVP → avançado)

### Fase 1 – Foco em skills concretas

- Treinar apenas para:
  - Gramática (verbos, artigos, preposições)
  - Vocabulário (básico, rotinas, família, viagens)
- Ignorar (por enquanto):
  - Interação, mediação, compreensão global
- Objetivo: **F1 > 0.90** para ~50–70 skills principais A1–B1.

### Fase 2 – Expandir para B2–C1

- Incluir:
  - Estruturas complexas (subjuntivo, discurso indireto, conectores avançados)
  - Vocabulário abstrato e acadêmico

### Fase 3 – Skills abstratas

- Interação, mediação, “fluência”, nuances de registro
- Necessário:
  - Dados multi-turnos
  - Janelas de contexto maiores

---

## Integração com AKT e CEFR

- As probabilidades por skill saídas pelo modelo treinado:
  - Podem alimentar diretamente o AKT como observações (correct / incorrect / used)
  - Podem ser agregadas por nível CEFR para:
    - Atualizar `cefr_progress`
    - Enriquecer o `interpretable_knowledge_state`

- As `linguistic_features_true` aprendidas:
  - Alimentam o mecanismo de FoLiBi (esquecimento por feature)
  - Melhoram a análise de padrões de erro por feature (já implementada).

---

## Próximos passos práticos

1. Definir lista “core” de ~60–80 skills prioritárias (A1–B1) para o primeiro modelo.
2. Criar um pequeno dataset anotado (ex: 500–1000 exemplos) com essas skills.
3. Escolher modelo base e biblioteca (ex: Hugging Face Transformers, vLLM, etc.).
4. Implementar script de treino + avaliação.
5. Integrar no `DiagnosticLLMClient` como modo alternativo (flag de experimento).


