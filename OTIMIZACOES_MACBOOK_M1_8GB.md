# Otimizações para MacBook M1 8GB

Guia completo de otimizações para melhorar performance do MacBook M1 com 8GB de RAM, focando no desenvolvimento do Parle Backend.

## 🎯 Otimizações do Projeto

### 1. Configuração de Workers e Concorrência

**Problema**: Com 8GB de RAM, múltiplos workers podem causar swap e lentidão.

**Solução**: Reduzir workers e otimizar pools de conexão.

#### Ajustar `src/core/config.py`:

```python
# ServerConfig - Reduzir workers para 1 no M1 8GB
workers: int = Field(default=1, ge=1, le=2)  # Máximo 2 workers

# DatabaseConfig - Reduzir pool size
pool_size: int = Field(default=5, ge=1, le=10)  # Reduzido de 10 para 5
max_overflow: int = Field(default=5, ge=0)  # Reduzido de 20 para 5
```

#### Variáveis de ambiente (`.env`):

```bash
# Workers e concorrência
SERVER_WORKERS=1
DB_POOL_SIZE=5
DB_MAX_OVERFLOW=5

# Limitar threads do PyTorch
OMP_NUM_THREADS=2
MKL_NUM_THREADS=2
NUMEXPR_NUM_THREADS=2
TORCH_NUM_THREADS=2
```

### 2. Otimização de Modelos ML

**Problema**: PyTorch, transformers e sentence-transformers consomem muita RAM.

**Soluções**:

#### a) Usar modelos menores e quantizados:

```python
# Em vez de modelos grandes, usar versões quantizadas ou menores
# Exemplo para sentence-transformers:
from sentence_transformers import SentenceTransformer

# Usar modelo menor e mais eficiente
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')  # ~420MB vs 1.5GB
# Ou ainda menor:
# model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')  # ~420MB
```

#### b) Lazy loading de modelos:

```python
# Carregar modelos apenas quando necessário
from functools import lru_cache

@lru_cache(maxsize=1)
def get_model():
    """Carrega modelo apenas uma vez, com cache"""
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
```

#### c) Limpar cache do PyTorch:

```python
import torch
import gc

# Após usar modelos pesados
torch.cuda.empty_cache() if torch.cuda.is_available() else None
gc.collect()
```

### 3. Configuração do PyTorch para M1

**Problema**: PyTorch pode não usar Metal Performance Shaders (MPS) corretamente.

**Solução**: Instalar PyTorch otimizado para M1:

```bash
# Desinstalar versão atual
pip uninstall torch torchaudio

# Instalar versão otimizada para M1
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
# Ou usar conda:
conda install pytorch torchvision torchaudio -c pytorch
```

**Configurar para usar MPS (Metal)**:

```python
import torch

# Verificar se MPS está disponível
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ Usando Metal Performance Shaders (GPU M1)")
else:
    device = torch.device("cpu")
    print("⚠️ MPS não disponível, usando CPU")
```

### 4. Otimização de spaCy

**Problema**: spaCy carrega modelos grandes na memória.

**Solução**:

```bash
# Usar modelo menor do spaCy
python -m spacy download pt_core_news_sm  # Small model (~40MB) vs lg (~500MB)
```

```python
# Carregar modelo menor
import spacy
nlp = spacy.load("pt_core_news_sm")  # Small model
```

### 5. Redis e Cache

**Problema**: Redis pode consumir muita memória.

**Solução**: Limitar memória do Redis:

```bash
# No arquivo de configuração do Redis ou .env
REDIS_MAXMEMORY=512mb
REDIS_MAXMEMORY_POLICY=allkeys-lru
```

### 6. Processamento de Áudio

**Problema**: librosa e soundfile carregam arquivos inteiros na memória.

**Solução**: Processar em chunks:

```python
import librosa
import soundfile as sf

# Processar áudio em chunks ao invés de carregar tudo
def process_audio_chunks(file_path, chunk_size=1024*1024):  # 1MB chunks
    with sf.SoundFile(file_path) as f:
        for chunk in sf.blocks(file_path, blocksize=chunk_size):
            yield chunk
```

## 🖥️ Otimizações do Sistema macOS

### 1. Gerenciar Memória

#### a) Monitorar uso de memória:

```bash
# Terminal: monitorar memória em tempo real
watch -n 1 vm_stat

# Ou usar Activity Monitor (GUI)
# Applications > Utilities > Activity Monitor
```

#### b) Fechar aplicações desnecessárias:

- Fechar navegadores com muitas abas
- Fechar IDEs não utilizados
- Desativar extensões do VS Code/Cursor não essenciais
- Fechar aplicações em background (Spotify, Slack, etc.)

### 2. Swap e Memória Virtual

#### a) Limpar swap periodicamente:

```bash
# Ver uso de swap
sysctl vm.swapusage

# Reiniciar ajuda a limpar swap (não há comando direto no macOS)
```

#### b) Reduzir uso de swap:

```bash
# Desativar apps que usam muita memória
# Verificar com:
top -o mem
```

### 3. Configurações do Terminal/Shell

#### a) Limitar histórico do shell:

```bash
# No ~/.zshrc
export HISTSIZE=1000
export SAVEHIST=1000
```

#### b) Desativar plugins pesados do zsh:

```bash
# Comentar plugins não essenciais no ~/.zshrc
# plugins=(git docker kubectl)  # Manter apenas essenciais
```

### 4. Docker (se usado)

**Problema**: Docker Desktop consome muita RAM.

**Soluções**:

```bash
# Limitar memória do Docker
# Docker Desktop > Settings > Resources > Advanced
# Memory: 2GB (ao invés de 4GB+)
# CPUs: 2 (ao invés de todos)
```

### 5. Python Environment

#### a) Usar conda/mamba ao invés de venv (mais eficiente):

```bash
# Mamba é mais rápido que conda
conda install mamba -n base -c conda-forge
mamba env create -f environment.yml
```

#### b) Limpar cache do pip:

```bash
pip cache purge
```

#### c) Limpar cache do Python:

```bash
# Limpar __pycache__
find . -type d -name __pycache__ -exec rm -r {} +
find . -name "*.pyc" -delete
```

### 6. Desativar Recursos Desnecessários

#### a) Spotlight Indexing (se não usar muito):

```bash
# Desativar indexação de pastas grandes
sudo mdutil -i off /path/to/large/folder
```

#### b) Time Machine (se não usar):

```bash
# Desativar backups automáticos temporariamente
sudo tmutil disable
```

#### c) iCloud Drive (se não usar):

```bash
# System Settings > Apple ID > iCloud
# Desativar sincronização de pastas não essenciais
```

### 7. Configurações de Desenvolvimento

#### a) VS Code/Cursor:

```json
// settings.json
{
  "files.watcherExclude": {
    "**/.git/objects/**": true,
    "**/.git/subtree-cache/**": true,
    "**/node_modules/**": true,
    "**/__pycache__/**": true,
    "**/.pytest_cache/**": true,
    "**/reports/**": true
  },
  "search.exclude": {
    "**/node_modules": true,
    "**/__pycache__": true,
    "**/.pytest_cache": true,
    "**/reports": true
  },
  "python.analysis.autoImportCompletions": false,  // Reduz uso de memória
  "python.analysis.typeCheckingMode": "off"  // Mais rápido
}
```

#### b) Desativar extensões pesadas:

- Desativar extensões de IA não essenciais
- Desativar linters pesados quando não necessário
- Usar apenas extensões essenciais

### 8. Limpar Sistema

#### a) Limpar cache do sistema:

```bash
# Limpar cache do usuário
rm -rf ~/Library/Caches/*

# Limpar logs antigos
sudo rm -rf /private/var/log/*.log
```

#### b) Limpar espaço em disco:

```bash
# Ver uso de disco
df -h

# Limpar arquivos grandes
du -sh ~/* | sort -h
```

## 🚀 Script de Otimização Automática

Crie um script para aplicar otimizações:

```bash
#!/bin/bash
# optimize_m1.sh

echo "🔧 Otimizando MacBook M1 8GB..."

# Limpar cache Python
echo "🧹 Limpando cache Python..."
find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null

# Limpar cache pip
echo "🧹 Limpando cache pip..."
pip cache purge 2>/dev/null

# Limpar cache conda
echo "🧹 Limpando cache conda..."
conda clean --all -y 2>/dev/null

# Verificar processos pesados
echo "📊 Top 5 processos usando mais memória:"
ps aux | sort -nrk 4 | head -6

# Verificar swap
echo "💾 Uso de swap:"
sysctl vm.swapusage

echo "✅ Otimização concluída!"
```

## 📊 Monitoramento

### Script de monitoramento de memória:

```python
# monitor_memory.py
import psutil
import os

def get_memory_info():
    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()
    
    print(f"RAM Total: {mem.total / (1024**3):.2f} GB")
    print(f"RAM Usada: {mem.used / (1024**3):.2f} GB ({mem.percent}%)")
    print(f"RAM Disponível: {mem.available / (1024**3):.2f} GB")
    print(f"Swap Total: {swap.total / (1024**3):.2f} GB")
    print(f"Swap Usada: {swap.used / (1024**3):.2f} GB ({swap.percent}%)")
    
    # Top processos
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
        try:
            processes.append({
                'pid': proc.info['pid'],
                'name': proc.info['name'],
                'memory': proc.info['memory_info'].rss / (1024**2)  # MB
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    processes.sort(key=lambda x: x['memory'], reverse=True)
    print("\n🔝 Top 5 processos por memória:")
    for p in processes[:5]:
        print(f"  {p['name']}: {p['memory']:.2f} MB")

if __name__ == "__main__":
    get_memory_info()
```

## ⚡ Quick Wins (Aplicar Imediatamente)

1. **Reduzir workers para 1** no `.env`:
   ```bash
   SERVER_WORKERS=1
   ```

2. **Limitar threads do PyTorch**:
   ```bash
   export OMP_NUM_THREADS=2
   export TORCH_NUM_THREADS=2
   ```

3. **Fechar aplicações desnecessárias** (navegador com muitas abas, etc.)

4. **Usar modelos ML menores** (MiniLM ao invés de modelos grandes)

5. **Limpar cache Python**:
   ```bash
   find . -type d -name __pycache__ -exec rm -r {} +
   ```

6. **Desativar reload do uvicorn em produção**:
   ```bash
   SERVER_RELOAD=false
   ```

## 📝 Checklist de Otimização

- [ ] Reduzir `SERVER_WORKERS` para 1
- [ ] Reduzir `DB_POOL_SIZE` para 5
- [ ] Configurar variáveis de ambiente de threads (OMP_NUM_THREADS, etc.)
- [ ] Usar modelos ML menores (MiniLM)
- [ ] Configurar PyTorch para MPS (Metal)
- [ ] Usar spaCy small model
- [ ] Limitar memória do Redis
- [ ] Fechar aplicações desnecessárias
- [ ] Limpar cache Python/pip/conda
- [ ] Configurar VS Code/Cursor para excluir pastas grandes
- [ ] Desativar extensões pesadas do editor
- [ ] Limitar memória do Docker (se usado)
- [ ] Criar script de monitoramento de memória

## 🔍 Diagnóstico

Se ainda estiver lento, verifique:

1. **Qual processo está usando mais memória?**
   ```bash
   top -o mem
   ```

2. **Há swap sendo usado?**
   ```bash
   sysctl vm.swapusage
   ```

3. **Quanto espaço em disco está disponível?** (menos de 10GB pode causar lentidão)
   ```bash
   df -h
   ```

4. **Há processos Python órfãos?**
   ```bash
   ps aux | grep python
   ```

## 📚 Referências

- [PyTorch M1 Optimization](https://pytorch.org/get-started/locally/)
- [macOS Memory Management](https://support.apple.com/guide/activity-monitor/view-memory-usage-actmntr1001/mac)
- [Python Memory Profiling](https://pypi.org/project/memory-profiler/)

---

**Última atualização**: 2025-01-22
