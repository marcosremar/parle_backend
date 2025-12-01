# Circuit Breakers - Parle Backend

Este documento descreve o sistema de circuit breakers implementado no Parle Backend.

## 📋 Visão Geral

Circuit breakers protegem o sistema contra falhas em cascata, automaticamente desabilitando serviços que estão falhando e usando fallbacks.

## 🔌 Circuit Breaker Implementado

### Localização

- **Arquivo**: `src/modules/conversation/orchestrator/utils/pipeline/circuit_breaker.py`
- **Classe**: `CircuitBreaker`

### Estados

1. **CLOSED** (Fechado)
   - Estado normal
   - Usa provedor primário
   - Monitora falhas

2. **OPEN** (Aberto)
   - Muitas falhas detectadas
   - Usa apenas provedor fallback
   - Não tenta provedor primário

3. **HALF_OPEN** (Meio Aberto)
   - Testando recuperação
   - Tenta provedor primário novamente
   - Se sucesso: volta para CLOSED
   - Se falha: volta para OPEN

### Fluxo de Estados

```
CLOSED --(3 falhas)--> OPEN --(30s timeout)--> HALF_OPEN --(sucesso)--> CLOSED
                                                          |
                                                    (falha)
                                                          |
                                                          v
                                                       OPEN
```

## ⚙️ Configuração

### Configuração Padrão

```python
config = CircuitBreakerConfig(
    failure_threshold=3,        # 3 falhas antes de abrir
    recovery_timeout=30,         # 30 segundos antes de tentar novamente
    half_open_max_calls=1,      # 1 chamada de teste em half-open
    primary_timeout=10,          # 10s timeout para primário
    fallback_timeout=15          # 15s timeout para fallback
)
```

### Personalizar Configuração

```python
from src.modules.conversation.orchestrator.utils.pipeline.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig
)

# Configuração customizada
config = CircuitBreakerConfig(
    failure_threshold=5,        # Mais tolerante
    recovery_timeout=60,         # Espera mais tempo
    primary_timeout=15,
    fallback_timeout=20
)

circuit_breaker = CircuitBreaker(config)
```

## 🔄 Uso

### Exemplo Básico

```python
async def call_with_fallback():
    result, provider = await circuit_breaker.call_with_fallback(
        primary_fn=primary_llm_call,
        fallback_fn=fallback_llm_call,
        context={"prompt": "Hello"}
    )
    
    if provider == "fallback":
        logger.warning("Using fallback provider")
    
    return result
```

### Integração com LLM

O circuit breaker já está integrado no módulo LLM para fallback automático entre provedores.

## 📊 Monitoramento

### Métricas

O circuit breaker expõe métricas que podem ser monitoradas:

- Estado atual do circuit breaker
- Número de falhas
- Última falha
- Tentativas de recuperação

### Logs

O circuit breaker registra eventos importantes:

```
🔌 Circuit breaker initialized: threshold=3, recovery_timeout=30s
⚠️  Circuit OPEN - using fallback LLM (retry in 25.3s)
🔄 Circuit breaker HALF_OPEN - testing primary LLM recovery
✅ Circuit breaker CLOSED - primary LLM recovered
```

## 🧪 Testes

### Testar Circuit Breaker

```python
import pytest
from src.modules.conversation.orchestrator.utils.pipeline.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState
)

@pytest.mark.asyncio
async def test_circuit_breaker_opens_after_failures():
    """Test that circuit opens after threshold failures"""
    config = CircuitBreakerConfig(failure_threshold=2, recovery_timeout=5)
    cb = CircuitBreaker(config)
    
    # Simulate failures
    async def failing_fn(ctx):
        raise Exception("Failure")
    
    async def fallback_fn(ctx):
        return "fallback_result"
    
    # First failure
    try:
        await cb.call_with_fallback(failing_fn, fallback_fn, {})
    except:
        pass
    
    # Second failure should open circuit
    result, provider = await cb.call_with_fallback(failing_fn, fallback_fn, {})
    assert provider == "fallback"
    assert cb.state == CircuitState.OPEN
```

## 🔧 Melhorias Futuras

### Adicionar Circuit Breakers em Outros Módulos

1. **STT Module**:
   ```python
   # Fallback: OpenAI Whisper -> Groq -> Local
   ```

2. **TTS Module**:
   ```python
   # Fallback: ElevenLabs -> HuggingFace -> gTTS
   ```

3. **Database Calls**:
   ```python
   # Fallback: Primary DB -> Read Replica -> Cache
   ```

### Configuração Dinâmica

Permitir ajustar thresholds em runtime baseado em métricas:

```python
# Ajustar threshold baseado em taxa de erro
if error_rate > 0.1:
    circuit_breaker.config.failure_threshold = 2  # Mais sensível
else:
    circuit_breaker.config.failure_threshold = 5  # Mais tolerante
```

## 📚 Referências

- [Circuit Breaker Pattern](https://martinfowler.com/bliki/CircuitBreaker.html)
- [Resilience4j Documentation](https://resilience4j.readme.io/docs/circuitbreaker)
