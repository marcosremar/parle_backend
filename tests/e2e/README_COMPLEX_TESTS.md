# Testes E2E Complexos - Multi-Turn Conversations

## 📋 Visão Geral

Este arquivo contém testes E2E complexos que testam interações com múltiplos turnos de conversação, mantendo contexto e testando cenários diversos.

## 🧪 Testes Disponíveis

### 1. `test_contextual_conversation_name_remembering`
**Objetivo**: Testa se o sistema mantém contexto sobre informações do usuário (nome) através de múltiplos turnos.

**Cenário**:
- Turn 1: Usuário se apresenta ("Meu nome é Maria Silva")
- Turn 2: Usuário pergunta seu próprio nome
- Turn 3: Continuação da conversa

**Validação**: Sistema deve manter contexto sobre o nome do usuário.

---

### 2. `test_topic_switching_conversation`
**Objetivo**: Testa mudanças de tópico durante a conversa.

**Cenário**:
- Turn 1: Fala sobre tempo
- Turn 2: Muda para comida
- Turn 3: Muda para viagens
- Turn 4: Volta para tempo

**Validação**: Sistema deve lidar bem com mudanças de tópico.

---

### 3. `test_question_answer_chain`
**Objetivo**: Testa cadeia de perguntas e respostas que se constroem umas sobre as outras.

**Cenário**:
- Perguntas sobre sistema solar que se relacionam
- Cada pergunta referencia a anterior

**Validação**: Sistema deve manter contexto entre perguntas relacionadas.

---

### 4. `test_long_conversation_10_turns`
**Objetivo**: Testa conversa longa com 10+ turnos mantendo contexto.

**Cenário**:
- 10 turnos de conversa sobre diversos tópicos
- Testa se o sistema mantém contexto ao longo de toda a conversa

**Validação**: Sistema deve processar todos os turnos e manter contexto.

---

### 5. `test_conversation_with_scenario`
**Objetivo**: Testa conversa com cenário específico (ex: restaurante).

**Cenário**:
- Cria cenário de restaurante
- Múltiplos turnos simulando pedido em restaurante

**Validação**: Sistema deve seguir o contexto do cenário.

---

### 6. `test_conversation_history_retention`
**Objetivo**: Testa se histórico de conversa é mantido.

**Cenário**:
- Estabelece contexto inicial
- Pergunta posterior que requer contexto anterior

**Validação**: Sistema deve referenciar informações anteriores.

---

### 7. `test_mixed_text_and_context`
**Objetivo**: Testa mistura de diferentes tipos de interações.

**Cenário**:
- Mistura de declarações, perguntas e pedidos
- Testa diferentes tipos de input do usuário

**Validação**: Sistema deve lidar com todos os tipos de interação.

---

### 8. `test_conversation_with_error_recovery`
**Objetivo**: Testa recuperação de erros durante conversa.

**Cenário**:
- Turnos normais
- Continuação após possíveis problemas

**Validação**: Sistema deve continuar funcionando após erros.

---

### 9. `test_concurrent_sessions`
**Objetivo**: Testa múltiplas sessões de conversa simultâneas.

**Cenário**:
- Cria 3 sessões diferentes
- Processa turnos em paralelo em cada sessão

**Validação**: Sistema deve manter sessões isoladas e funcionar em paralelo.

---

### 10. `test_conversation_with_voice_preferences`
**Objetivo**: Testa manutenção de preferências de voz através dos turnos.

**Cenário**:
- Turnos com diferentes vozes especificadas
- Testa se preferências são mantidas

**Validação**: Sistema deve respeitar preferências de voz.

---

### 11. `test_deep_context_conversation`
**Objetivo**: Testa conversa profunda requerendo múltiplas camadas de contexto.

**Cenário**:
- 6 turnos construindo contexto complexo
- 5 perguntas que requerem contexto profundo

**Validação**: Sistema deve manter e usar contexto complexo.

---

## 🚀 Como Executar

### Executar todos os testes complexos
```bash
pytest tests/e2e/test_complex_multi_turn_conversations.py -v
```

### Executar teste específico
```bash
pytest tests/e2e/test_complex_multi_turn_conversations.py::TestComplexMultiTurnConversations::test_long_conversation_10_turns -v
```

### Executar testes por categoria
```bash
# Testes de contexto
pytest tests/e2e/test_complex_multi_turn_conversations.py -k "contextual or history or deep" -v

# Testes de múltiplos turnos
pytest tests/e2e/test_complex_multi_turn_conversations.py -k "long or chain" -v

# Testes de cenários
pytest tests/e2e/test_complex_multi_turn_conversations.py -k "scenario or topic" -v
```

## 📊 Estatísticas

- **Total de testes**: 11
- **Turnos testados**: 50+ turnos de conversa
- **Cenários cobertos**: Contexto, mudança de tópico, cadeias de perguntas, sessões paralelas
- **Tempo estimado**: 5-10 minutos para todos os testes

## ✅ Validações

Cada teste valida:
- ✅ Respostas são geradas para cada turno
- ✅ Contexto é mantido entre turnos
- ✅ Sessões são isoladas corretamente
- ✅ Sistema funciona com diferentes tipos de input
- ✅ Recuperação de erros funciona

## 🔍 Notas

- Testes usam módulos diretamente (não requerem servidor HTTP)
- Testes são assíncronos e podem ser executados em paralelo
- Alguns testes podem ser lentos devido a chamadas de API externas (LLM)
- Testes pulam automaticamente se dependências não estiverem disponíveis
