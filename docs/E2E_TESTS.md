# Testes End-to-End (E2E) - Parle Backend

Este documento descreve como executar e criar testes E2E para o Parle Backend.

## 📋 Visão Geral

Testes E2E verificam o sistema completo de ponta a ponta, simulando fluxos reais de usuário.

## 🎯 Cenários Críticos

### 1. Fluxo Completo Speech-to-Speech

**Descrição**: Usuário envia áudio, sistema transcreve, gera resposta com LLM, e sintetiza áudio de resposta.

**Passos**:
1. Usuário faz login
2. Cria sessão de conversação
3. Envia áudio (gravação ou arquivo)
4. Sistema processa: STT → LLM → TTS
5. Usuário recebe áudio de resposta
6. Verifica histórico da conversação

### 2. Fluxo de Conversação por Texto

**Descrição**: Conversação completa via texto.

**Passos**:
1. Usuário faz login
2. Envia mensagem de texto
3. Sistema gera resposta com LLM
4. Usuário recebe resposta
5. Continua conversação

### 3. Autenticação e Autorização

**Descrição**: Verifica fluxo completo de autenticação.

**Passos**:
1. Usuário se registra
2. Faz login
3. Acessa endpoints protegidos
4. Token expira
5. Renova token

### 4. Fallback de Provedores

**Descrição**: Verifica fallback quando provedor primário falha.

**Passos**:
1. Simula falha do provedor STT primário
2. Sistema usa provedor secundário
3. Requisição completa com sucesso

## 🚀 Executar Testes E2E

### Localmente

```bash
# Executar todos os testes E2E
pytest tests/e2e/ -v

# Executar teste específico
pytest tests/e2e/test_speech_to_speech.py -v

# Com cobertura
pytest tests/e2e/ --cov=src --cov-report=html
```

### Com Docker

```bash
# Iniciar ambiente de teste
docker-compose -f docker/docker-compose.test.yml up -d

# Executar testes
pytest tests/e2e/ -v

# Parar ambiente
docker-compose -f docker/docker-compose.test.yml down
```

## 📝 Criar Novos Testes E2E

### Estrutura de Teste

```python
import pytest
from fastapi.testclient import TestClient

@pytest.mark.e2e
def test_complete_flow():
    """Test complete user flow"""
    client = TestClient(app)
    
    # 1. Register user
    response = client.post("/api/v1/auth/register", json={
        "username": "testuser",
        "email": "test@example.com",
        "password": "testpass123"
    })
    assert response.status_code == 200
    
    # 2. Login
    response = client.post("/api/v1/auth/login", json={
        "email": "test@example.com",
        "password": "testpass123"
    })
    assert response.status_code == 200
    token = response.json()["token"]
    
    # 3. Create session
    headers = {"Authorization": f"Bearer {token}"}
    response = client.post(
        "/api/v1/conversation/session/create",
        headers=headers,
        data={"user_id": "testuser"}
    )
    assert response.status_code == 200
    session_id = response.json()["session_id"]
    
    # 4. Send message
    response = client.post(
        "/api/v1/conversation",
        headers=headers,
        data={
            "message": "Hello",
            "session_id": session_id
        }
    )
    assert response.status_code == 200
    assert "text" in response.json()
```

## 🔧 Fixtures para E2E

```python
# tests/e2e/conftest.py
import pytest
from fastapi.testclient import TestClient

@pytest.fixture
def authenticated_client():
    """Client with authenticated user"""
    client = TestClient(app)
    # Register and login
    # Return client with auth headers
    return client

@pytest.fixture
def test_audio():
    """Test audio file"""
    # Return test audio data
    pass
```

## 🎭 Dados de Teste Realistas

### Áudio de Teste

```python
@pytest.fixture
def sample_audio_base64():
    """Realistic test audio in base64"""
    # Use actual audio file converted to base64
    with open("tests/fixtures/test_audio_real_speech.wav", "rb") as f:
        return base64.b64encode(f.read()).decode()
```

### Conversações de Teste

```python
@pytest.fixture
def test_conversation_scenarios():
    """Realistic conversation scenarios"""
    return [
        {
            "user": "Olá, como você está?",
            "expected_topics": ["saudação", "pergunta"]
        },
        {
            "user": "Explique o que é Python",
            "expected_topics": ["educação", "tecnologia"]
        }
    ]
```

## 🔄 CI/CD

### GitHub Actions

```yaml
# .github/workflows/e2e.yml
name: E2E Tests

on:
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM
  workflow_dispatch:

jobs:
  e2e:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Start services
        run: docker-compose -f docker/docker-compose.yml up -d
      - name: Run E2E tests
        run: pytest tests/e2e/ -v
      - name: Cleanup
        run: docker-compose -f docker/docker-compose.yml down
```

## 📊 Relatórios

### Gerar Relatório HTML

```bash
pytest tests/e2e/ --html=reports/e2e_report.html --self-contained-html
```

### Screenshots/Videos

Para testes com interface web, capture screenshots:

```python
@pytest.mark.e2e
def test_ui_flow():
    # Take screenshot on failure
    try:
        # Test code
        pass
    except Exception:
        driver.save_screenshot("failure.png")
        raise
```

## 🐛 Debugging

### Executar com Logs Detalhados

```bash
pytest tests/e2e/ -v -s --log-cli-level=DEBUG
```

### Pausar em Falhas

```bash
pytest tests/e2e/ --pdb
```

## 📚 Referências

- [pytest Documentation](https://docs.pytest.org/)
- [FastAPI Testing](https://fastapi.tiangolo.com/tutorial/testing/)
- [E2E Testing Best Practices](https://kentcdodds.com/blog/common-mistakes-with-react-testing-library)
