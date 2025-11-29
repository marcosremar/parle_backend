# Simplificação de Chamadas HTTP

## Resumo

Simplificamos o sistema para usar **chamadas diretas aos módulos** ao invés de HTTP para módulos internos, eliminando camadas desnecessárias e melhorando performance.

## Mudanças Principais

### 1. BaseServiceClient Simplificado

**Antes:**
- Módulos internos usavam HTTP mesmo sendo módulos locais
- Necessário `MONOLITH_MODE=true` para usar chamadas diretas
- Sempre criava HTTP session mesmo para módulos internos

**Depois:**
- Módulos marcados com `is_module_service=True` **sempre** usam chamadas diretas
- Não precisa mais de `MONOLITH_MODE`
- HTTP session só é criado para serviços externos reais

### 2. Clients Atualizados

Todos os clients de módulos internos agora usam chamadas diretas:

- ✅ **LLM Client** (`ExternalLLMClient`) - usa `LLMModule` diretamente
- ✅ **STT Client** (`ExternalSTTClient`) - usa `STTModule` diretamente  
- ✅ **TTS Client** (`ExternalTTSClient`) - usa `TTSModule` diretamente
- ✅ **Session Client** - usa `SessionModule` diretamente
- ✅ **Scenarios Client** - usa `ScenariosModule` diretamente
- ✅ **Database Client** - usa `DatabaseModule` diretamente
- ✅ **File Storage Client** - usa `FileStorageModule` diretamente

### 3. Serviços que Ainda Usam HTTP

Apenas serviços externos reais continuam usando HTTP:

- `conversation_store` - pode ser externo
- `conversation_history` - pode ser externo
- `user` - pode ser externo
- `websocket` - comunicação real-time externa
- `webrtc` - comunicação real-time externa
- `rest_polling` - comunicação real-time externa
- `api_gateway` - gateway externo

## Benefícios

1. **Performance**: Elimina overhead de serialização/deserialização HTTP
2. **Simplicidade**: Menos código, menos complexidade
3. **Latência**: Chamadas diretas são muito mais rápidas
4. **Menos Recursos**: Não precisa manter HTTP sessions para módulos internos
5. **Debugging**: Mais fácil debugar chamadas diretas

## Como Funciona Agora

### Inicialização

```python
# Engine cria clients
clients = create_service_clients()

# Inicializa cada client
for name, client in clients.items():
    if client.is_module_service:
        # Módulos internos: sem HTTP session
        await client.initialize(None)
    else:
        # Serviços externos: com HTTP session
        await client.initialize(http_session)
```

### Uso

```python
# LLM Client - chamada direta
result = await llm_client.generate("Hello")

# Internamente:
# 1. Verifica se direct_module existe
# 2. Inicializa módulo se necessário
# 3. Chama método diretamente: await direct_module.generate(...)
# 4. Fallback para HTTP apenas se módulo não disponível
```

## Migração

### Para Novos Módulos

Ao criar um novo módulo interno:

1. Marque como `is_module_service=True` no client
2. O sistema automaticamente usará chamadas diretas
3. Não precisa configurar URLs ou HTTP

### Para Módulos Existentes

Se um módulo precisa continuar usando HTTP:

1. Não marque como `is_module_service=True`
2. Configure URL do serviço
3. Sistema usará HTTP normalmente

## Compatibilidade

- ✅ **Backward Compatible**: Código existente continua funcionando
- ✅ **Fallback Automático**: Se módulo não disponível, usa HTTP
- ✅ **Gradual**: Pode migrar módulo por módulo

## Exemplo de Código

### Antes (HTTP)

```python
# Client fazia HTTP request
result = await self._post("/api/generate", json_data={...})
```

### Depois (Direto)

```python
# Client chama módulo diretamente
if self.direct_module:
    result = await self.direct_module.generate(...)
else:
    # Fallback HTTP se módulo não disponível
    result = await self._post("/api/generate", ...)
```

## Métricas Esperadas

- **Latência**: Redução de 50-90% (dependendo do módulo)
- **Throughput**: Aumento de 20-50%
- **Uso de Memória**: Redução de ~10-20% (menos HTTP sessions)
- **CPU**: Redução de ~5-15% (menos serialização JSON)

## Notas

- HTTP ainda é usado como fallback se módulo não disponível
- Serviços externos continuam usando HTTP normalmente
- Não há breaking changes - tudo é backward compatible
