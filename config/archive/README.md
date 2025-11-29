# Configurações Arquivadas

Este diretório contém configurações deprecated que foram arquivadas.

## Arquivos Arquivados

### `settings.py` e `settings_service.py`

**Status**: ⚠️ **DEPRECATED** desde v5.2

**Razão**: Substituídos por `src.core.config`

**Migração**:
```python
# OLD (deprecated):
from config.settings import get_settings
settings = get_settings()

# NEW (recomendado):
from src.core.config import get_config
config = get_config()
```

**Data de Arquivamento**: 2025-01-XX

**Nota**: Estes arquivos foram mantidos para referência histórica. Não devem ser usados em novo código.
