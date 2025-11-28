# E2E Module Tests

Testes E2E completos e organizados para cada módulo do sistema.

## Estrutura de Testes

### Testes por Módulo

1. **`test_modules_speech.py`** - Módulos de Speech
   - STT (Speech-to-Text)
   - TTS (Text-to-Speech)
   - Testes de transcrição e síntese
   - Testes de providers e modelos

2. **`test_modules_auth.py`** - Módulo de Autenticação
   - Criação de usuários
   - Login e autenticação
   - Busca de usuários
   - Validação de credenciais

3. **`test_modules_storage.py`** - Módulos de Storage
   - Conversation Store (armazenamento de conversas)
   - File Storage (armazenamento de arquivos)
   - Database (armazenamento de dados)
   - Testes de CRUD completo

4. **`test_modules_conversation.py`** - Módulos de Conversação
   - Session (gerenciamento de sessões)
   - Scenarios (gerenciamento de cenários)
   - Orchestrator (orquestração de conversas)
   - Testes de fluxo completo

5. **`test_modules_llm.py`** - Módulo LLM
   - Geração de texto
   - Chat completion
   - Listagem de modelos

6. **`test_modules_integration.py`** - Testes de Integração
   - Integração entre múltiplos módulos
   - Fluxos completos end-to-end
   - Persistência de dados
   - Padrão singleton

## Como Executar

### Executar todos os testes

```bash
pytest tests/e2e/test_modules_*.py -v
```

### Executar testes específicos

```bash
# Apenas speech
pytest tests/e2e/test_modules_speech.py -v

# Apenas auth
pytest tests/e2e/test_modules_auth.py -v

# Apenas storage
pytest tests/e2e/test_modules_storage.py -v

# Apenas conversation
pytest tests/e2e/test_modules_conversation.py -v

# Apenas LLM
pytest tests/e2e/test_modules_llm.py -v

# Apenas integração
pytest tests/e2e/test_modules_integration.py -v
```

### Executar teste específico

```bash
pytest tests/e2e/test_modules_auth.py::TestUserModule::test_create_user -v
```

## Cobertura de Testes

### Speech Modules
- ✅ Inicialização
- ✅ Listagem de modelos/vozes
- ✅ Transcrição de áudio
- ✅ Síntese de texto para áudio
- ✅ Diferentes providers

### Auth Module
- ✅ Criação de usuários
- ✅ Login
- ✅ Validação de credenciais
- ✅ Busca por ID e email
- ✅ Prevenção de duplicatas

### Storage Modules
- ✅ Conversation Store: save/get/list
- ✅ File Storage: upload/download/metadata/list
- ✅ Database: set/get com diferentes tipos

### Conversation Modules
- ✅ Session: create/get/list
- ✅ Scenarios: create/get/list
- ✅ Orchestrator: process_text_conversation, stats

### LLM Module
- ✅ Inicialização
- ✅ Listagem de modelos
- ✅ Geração de texto
- ✅ Chat completion

### Integration Tests
- ✅ Fluxo completo de conversação
- ✅ Integração entre módulos
- ✅ Persistência de dados
- ✅ Padrão singleton

## Notas

- Testes que requerem APIs externas (LLM, STT, TTS) podem ser pulados se não configurados
- Testes usam `pytest.skip()` quando dependências não estão disponíveis
- Todos os testes definem `MONOLITH_MODE=true` automaticamente
- Cache de módulos é limpo entre testes

## Melhorias Futuras

- [ ] Testes de performance
- [ ] Testes de carga
- [ ] Testes de erro e recuperação
- [ ] Testes de concorrência
- [ ] Testes de tutoriais (quando habilitados)
