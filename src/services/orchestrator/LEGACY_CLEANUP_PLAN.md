# Plano de Limpeza de Código Legado

## Resumo

Foram identificados **3 arquivos** que precisam ser atualizados para remover o código legado:

1. `orchestrator_engine.py` - Remover `self.stats`, migrar para `stats_tracker`
2. `process_turn_with_talker.py` - Migrar para usar `stats_tracker`
3. `service.py` - Migrar endpoint `/stats` para usar `stats_tracker`

---

## Dependências Externas Encontradas

### 1. process_turn_with_talker.py (6 ocorrências)
- Linha 53: `orchestrator.stats["total_turns"] += 1`
- Linha 117: `orchestrator.stats["failed_turns"] += 1`
- Linha 167: `orchestrator.stats["successful_turns"] += 1`
- Linha 168: `orchestrator.stats["total_processing_time"] += total_time`
- Linha 172: `orchestrator.stats["in_process_count"] += 1`
- Linha 174: `orchestrator.stats["fallback_llm_count"] += 1`
- Linha 198: `orchestrator.stats["failed_turns"] += 1`

### 2. service.py (1 ocorrência)
- Linha 217: `"stats": self.orchestrator.stats if self.orchestrator else {}`

---

## Plano de Execução

### Fase 1: Migrar process_turn_with_talker.py
**Arquivo**: `src/services/orchestrator/process_turn_with_talker.py`

**Mudanças**:
- Substituir `orchestrator.stats["key"] += 1` por `orchestrator.stats_tracker.increment_*()`
- Mapeamento:
  - `stats["total_turns"]` → `stats_tracker.increment_total_turns()`
  - `stats["failed_turns"]` → `stats_tracker.increment_failed_turns()`
  - `stats["successful_turns"]` → `stats_tracker.increment_successful_turns()`
  - `stats["total_processing_time"]` → `stats_tracker.add_processing_time()`
  - `stats["in_process_count"]` → `stats_tracker.increment_in_process_count()`
  - `stats["fallback_llm_count"]` → `stats_tracker.increment_fallback_llm_count()`

### Fase 2: Migrar service.py
**Arquivo**: `src/services/orchestrator/service.py`

**Mudanças**:
- Linha 217: Substituir `self.orchestrator.stats` por `self.orchestrator.get_stats()`
- `get_stats()` já usa `stats_tracker` internamente (após migração)

### Fase 3: Migrar orchestrator_engine.py
**Arquivo**: `src/services/orchestrator/orchestrator_engine.py`

**Mudanças**:

#### 3.1 Remover definição de self.stats (linha 145-155)
```python
# REMOVER:
# Legacy stats (for backward compatibility, will be replaced by stats_tracker)
self.stats: StatsDict = {
    StatsKey.TOTAL_TURNS: 0,
    # ...
}
```

#### 3.2 Migrar get_stats() (linha 524-585)
```python
# ANTES:
def get_stats(self) -> StatsDict:
    avg_time = (self.stats[StatsKey.TOTAL_PROCESSING_TIME] / self.stats[StatsKey.TOTAL_TURNS]
               if self.stats[StatsKey.TOTAL_TURNS] > 0 else 0)
    return {
        **self.stats,
        # ...
    }

# DEPOIS:
def get_stats(self) -> StatsDict:
    if not self.stats_tracker:
        return {}
    stats = self.stats_tracker.get_stats()
    avg_time = stats.get("average_processing_time_ms", 0) / 1000
    return {
        **stats,
        # ... (manter campos adicionais como controller_integration, etc)
    }
```

#### 3.3 Migrar reset_stats() (linha 587-599)
```python
# ANTES:
def reset_stats(self) -> None:
    self.stats = {
        StatsKey.TOTAL_TURNS: 0,
        # ...
    }

# DEPOIS:
def reset_stats(self) -> None:
    if self.stats_tracker:
        self.stats_tracker.reset()
    logger.info("📊 Statistics reset")
```

#### 3.4 Remover sincronização (linha 445-448)
```python
# REMOVER:
# Sync stats from tracker to legacy stats dict for backward compatibility
if self.stats_tracker:
    tracker_stats = self.stats_tracker.get_stats()
    self.stats.update(tracker_stats)
```

#### 3.5 Migrar process_text_conversation() (linha 601-910)
- Linha 631: `self.stats[StatsKey.TOTAL_TURNS] += 1` → `self.stats_tracker.increment_total_turns()`
- Linha 779, 790, 803: `self.stats["primary_llm_count"] += 1` → `self.stats_tracker.increment_primary_llm_count()`
- Linha 807: `self.stats["failed_turns"] += 1` → `self.stats_tracker.increment_failed_turns()`
- Linha 882: `self.stats[StatsKey.SUCCESSFUL_TURNS] += 1` → `self.stats_tracker.increment_successful_turns()`
- Linha 883: `self.stats[StatsKey.TOTAL_PROCESSING_TIME] += processing_time` → `self.stats_tracker.add_processing_time(processing_time)`
- Linha 910: `self.stats[StatsKey.FAILED_TURNS] += 1` → `self.stats_tracker.increment_failed_turns()`

#### 3.6 Migrar process_turn_structured() (linha 918-1173)
- Linha 956: `self.stats[StatsKey.TOTAL_TURNS] += 1` → `self.stats_tracker.increment_total_turns()`
- Linha 974: `self.stats[StatsKey.FAILED_TURNS] += 1` → `self.stats_tracker.increment_failed_turns()`

#### 3.7 Migrar process_turn_with_talker() (linha 1175-1205)
- Linha 1059: `self.stats[StatsKey.FAILED_TURNS] += 1` → `self.stats_tracker.increment_failed_turns()`
- Linha 1133: `self.stats[StatsKey.SUCCESSFUL_TURNS] += 1` → `self.stats_tracker.increment_successful_turns()`
- Linha 1134: `self.stats[StatsKey.TOTAL_PROCESSING_TIME] += processing_time` → `self.stats_tracker.add_processing_time(processing_time)`
- Linha 1168: `self.stats[StatsKey.FAILED_TURNS] += 1` → `self.stats_tracker.increment_failed_turns()`

#### 3.8 Remover comentário solto (linha 455)
```python
# REMOVER:
    # Step 1a: Load session first (required for scenario_id and conversation_id)
```

### Fase 4: Limpeza Final
1. Remover import de `StatsDict` se não for mais usado
2. Verificar que não há mais referências a `self.stats`
3. Executar testes
4. Verificar que `get_stats()` retorna dados corretos

---

## Verificação Pós-Migração

### Comandos de Verificação
```bash
# Verificar que não há mais referências a .stats[
grep -r "\.stats\[" src/services/orchestrator/

# Verificar que stats_tracker está sendo usado
grep -r "stats_tracker\." src/services/orchestrator/

# Executar testes
pytest src/services/orchestrator/tests/ -v
```

### Testes a Verificar
1. ✅ Testes unitários dos engines
2. ✅ Testes de integração (se existirem)
3. ✅ Verificar endpoint `/stats` retorna dados corretos
4. ✅ Verificar que `reset_stats()` funciona

---

## Riscos e Mitigações

### Risco 1: Quebra de API Externa
**Risco**: Se algum código externo acessa `orchestrator.stats` diretamente
**Mitigação**: 
- Verificar todas as referências antes de remover
- Manter `get_stats()` como interface pública
- Se necessário, criar propriedade `@property` temporária para compatibilidade

### Risco 2: Inconsistência de Dados
**Risco**: Se `stats_tracker` não estiver inicializado
**Mitigação**:
- Adicionar verificações `if self.stats_tracker:` antes de usar
- Garantir que `stats_tracker` é sempre inicializado em `initialize()`

### Risco 3: Perda de Métricas Durante Migração
**Risco**: Métricas podem ser perdidas durante a migração
**Mitigação**:
- Migração é atômica (todos os métodos migrados de uma vez)
- `stats_tracker` já está funcionando, apenas removendo duplicação

---

## Estimativa

- **Tempo**: 2-3 horas
- **Arquivos afetados**: 3 arquivos
- **Linhas modificadas**: ~80-100 linhas
- **Linhas removidas**: ~30-40 linhas
- **Risco**: BAIXO (stats_tracker já está funcionando)

---

## Checklist de Execução

- [ ] Fase 1: Migrar `process_turn_with_talker.py`
- [ ] Fase 2: Migrar `service.py`
- [ ] Fase 3.1: Remover definição de `self.stats`
- [ ] Fase 3.2: Migrar `get_stats()`
- [ ] Fase 3.3: Migrar `reset_stats()`
- [ ] Fase 3.4: Remover sincronização
- [ ] Fase 3.5: Migrar `process_text_conversation()`
- [ ] Fase 3.6: Migrar `process_turn_structured()`
- [ ] Fase 3.7: Migrar `process_turn_with_talker()` (método interno)
- [ ] Fase 3.8: Remover comentário solto
- [ ] Fase 4: Limpeza final e verificação
- [ ] Executar testes
- [ ] Verificar que não há mais referências a `self.stats`
