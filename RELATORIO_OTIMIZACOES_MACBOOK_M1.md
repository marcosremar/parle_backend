# Relatório de Otimizações para MacBook M1
## Pesquisa Baseada em Fontes da Internet (2024-2025)

**Data do Relatório**: 2025-01-22  
**Foco**: Otimizações específicas para MacBook M1 (todas as variantes)  
**Status**: Apenas documentação - Nenhuma alteração aplicada

---

## 📋 Sumário Executivo

Este relatório compila otimizações específicas para MacBook M1 encontradas em fontes especializadas da internet. As otimizações são organizadas por categoria e incluem comandos específicos, configurações do sistema e práticas recomendadas para melhorar performance, especialmente em modelos com 8GB de RAM.

---

## 1. 🖥️ Otimizações do Sistema macOS

### 1.1 Gerenciamento de Processos em Background

**Problema**: Aplicações em background consomem recursos do sistema.

**Soluções**:

1. **Monitorar Uso de Recursos**
   - Usar Activity Monitor para identificar aplicações que consomem muitos recursos
   - Localização: Applications > Utilities > Activity Monitor

2. **Desativar Itens de Inicialização Desnecessários**
   - Navegar para: System Settings > General > Login Items
   - (Versões antigas: System Preferences > Users & Groups > Login Items)
   - Desativar programas que não precisam iniciar automaticamente

3. **Pausar Sincronizações de Nuvem Durante Tarefas Intensivas**
   - Pausar temporariamente serviços como Dropbox ou iCloud quando trabalhar com arquivos grandes
   - Reduz atividade de disco e uso de memória

**Fonte**: [Kate's Tech Blog](https://kates-tech-blog.gitbook.io/web-design-blog-2024/tips-for-optimizing-performance-when-using-heavy-creative-software-on-a-macbook)

### 1.2 Otimização de Armazenamento e Arquivos

**Problema**: SSD quase cheio degrada performance do sistema.

**Soluções**:

1. **Manter Espaço Livre**
   - Garantir pelo menos 15-20% do SSD livre para operação eficiente
   - Espaço insuficiente causa lentidão significativa

2. **Limpar Arquivos Temporários Regularmente**
   - Deletar caches e renders antigos de projetos
   - Organizar arquivos eficientemente

3. **Organizar Arquivos**
   - Armazenar projetos completos em drives externos ou cloud storage
   - Manter armazenamento interno organizado

**Fonte**: [Kate's Tech Blog](https://kates-tech-blog.gitbook.io/web-design-blog-2024/tips-for-optimizing-performance-when-using-heavy-creative-software-on-a-macbook)

### 1.3 Ajustes de Efeitos Visuais

**Problema**: Efeitos visuais consomem recursos do sistema.

**Soluções**:

1. **Desativar Animações**
   - System Settings > Accessibility > Display
   - Habilitar "Reduce Motion" e "Reduce Transparency"
   - Minimiza efeitos visuais e libera recursos

**Fonte**: [MacPaw](https://macpaw.com/how-to/optimize-macos-sequoia)

### 1.4 Gerenciamento de Temperatura

**Problema**: Superaquecimento leva a throttling de performance.

**Soluções**:

1. **Garantir Ventilação Adequada**
   - Usar MacBook em superfícies duras e planas
   - Permitir fluxo de ar adequado

2. **Limpar Vents e Fans**
   - Remover poeira regularmente dos vents e fans
   - Previne superaquecimento

**Fonte**: [Advice Scout](https://www.advicescout.com/mac-performance-optimization/)

### 1.5 Reinicialização Regular

**Solução**:
- Reiniciar MacBook regularmente limpa arquivos temporários e reseta memória do sistema
- Melhora performance geral

**Fonte**: [MacPaw](https://macpaw.com/how-to/optimize-macos-sequoia)

---

## 2. 💾 Otimizações de Memória (Especialmente para 8GB RAM)

### 2.1 Ambiente de Desenvolvimento

**Soluções**:

1. **Escolher Ferramentas Leves**
   - Usar editores eficientes como Visual Studio Code ou Sublime Text
   - Consomem menos memória que IDEs pesados

2. **Gerenciar Extensões do IDE**
   - Desativar plugins e extensões desnecessários no IDE
   - Em VS Code: ajustar configurações para excluir diretórios do file watching
   - Conserva recursos significativamente

**Fonte**: [StarMorph Blog](https://blog.starmorph.com/blog/mac-speed-optimization-guide)

### 2.2 Uso Eficiente de Docker e Virtualização

**Soluções**:

1. **Definir Limites de Recursos**
   - Alocar limites específicos de memória e CPU para containers
   - Exemplo:
     ```bash
     docker run --memory=2g --cpus=2 [image]
     ```

2. **Pausar Containers Não Utilizados**
   - Parar ou pausar containers Docker que não estão em uso ativo
   - Libera memória imediatamente

**Fonte**: [StarMorph Blog](https://blog.starmorph.com/blog/mac-speed-optimization-guide)

### 2.3 Monitoramento e Gerenciamento de Recursos

**Soluções**:

1. **Activity Monitor**
   - Usar Activity Monitor do macOS para identificar e fechar aplicações/processos que consomem muita memória
   - Monitoramento regular ajuda a manter performance ótima

2. **Fechar Aplicações Desnecessárias**
   - Manter número mínimo de aplicações abertas
   - Reduzir abas do navegador (navegadores com múltiplas abas são particularmente intensivos em memória)

**Fonte**: 
- [MacInfinity](https://www.macinfinity.sg/post/maximizing-mac-ram-performance)
- [XDA Developers](https://www.xda-developers.com/ways-free-up-macos-memory-apple-silicon/)

### 2.4 Ambientes de Desenvolvimento Baseados em Nuvem

**Solução**:
- Considerar usar IDEs baseados em nuvem como AWS Cloud9 ou GitHub Codespaces
- Essas plataformas descarregam tarefas de processamento para servidores remotos
- Reduz tensão na máquina local

**Fonte**: [Budget PC Upgrade Repair](https://www.budgetpcupgraderepair.com/can-you-code-websites-with-8gb-ram-macbook/)

### 2.5 Manutenção Regular do Sistema

**Soluções**:

1. **Reiniciar Periodicamente**
   - Reiniciar MacBook a cada poucos dias limpa caches de memória
   - Reduz swaps de memória virtual
   - Mantém responsividade do sistema

2. **Manter Software Atualizado**
   - Garantir que macOS e todas as ferramentas de desenvolvimento estão atualizadas
   - Updates frequentemente incluem melhorias de performance e melhor gerenciamento de memória

**Fonte**: [Apple Discussions](https://discussions.apple.com/thread/253915399)

### 2.6 Otimização do Finder e Desktop

**Soluções**:

1. **Organizar Desktop**
   - Desktop desorganizado pode consumir memória adicional
   - Organizar arquivos em pastas e manter desktop limpo
   - Melhora performance do sistema

2. **Limitar Janelas do Finder**
   - Evitar manter múltiplas janelas do Finder abertas
   - Especialmente aquelas exibindo diretórios grandes
   - Podem usar memória significativa

**Fonte**: 
- [MacInfinity](https://www.macinfinity.sg/post/maximizing-mac-ram-performance)
- [XDA Developers](https://www.xda-developers.com/ways-free-up-macos-memory-apple-silicon/)

---

## 3. 🐍 Otimizações para Python e PyTorch no M1

### 3.1 Requisitos do Sistema

**Requisitos**:
- Hardware: Mac com Apple silicon (M1, M1 Pro, M1 Max, etc.)
- Sistema Operacional: macOS 12.3 ou posterior
- Python: Versão 3.7 ou posterior
- Xcode Command-Line Tools: Instalar executando:
  ```bash
  xcode-select --install
  ```

**Fonte**: [Medium - PyTorch M1 Setup](https://medium.com/@ialwayslikedgrime/why-pytorch-is-better-than-tensorflow-successfully-setting-up-your-macbook-for-transformers-8e222bb21b22)

### 3.2 Instalação do Python

**Importante**: Usar versão Python compatível com arquitetura do Mac.

**Para Apple silicon, garantir uso da versão arm64 do Python**:

```bash
# Usando pyenv
brew install pyenv
pyenv install 3.12.11
pyenv global 3.12.11
python --version  # Deve exibir Python 3.12.11
```

Este setup garante compatibilidade com a maioria das bibliotecas de machine learning.

**Fonte**: [Medium - PyTorch M1 Setup](https://medium.com/@ialwayslikedgrime/why-pytorch-is-better-than-tensorflow-successfully-setting-up-your-macbook-for-transformers-8e222bb21b22)

### 3.3 Configuração de Ambiente Virtual

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3.4 Instalação do PyTorch com Suporte MPS

**Para aproveitar o backend MPS (Metal Performance Shaders)**:

```bash
pip install torch torchvision torchaudio
```

Este comando instala PyTorch junto com torchvision e torchaudio. Garantir que a versão instalada suporta MPS.

**Fonte**: [Apple Developer - Metal PyTorch](https://developer.apple.com/metal/pytorch/)

### 3.5 Verificar Disponibilidade do MPS

**Script de verificação**:

```python
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")

# Testar criação de tensor no MPS
if torch.backends.mps.is_available():
    device = torch.device("mps")
    x = torch.randn(1000, 1000, device=device)
    print(f"Successfully created tensor on MPS: {x.device}")
else:
    print("MPS device not found.")
```

**Fonte**: [Medium - PyTorch M1 Setup](https://medium.com/@ialwayslikedgrime/why-pytorch-is-better-than-tensorflow-successfully-setting-up-your-macbook-for-transformers-8e222bb21b22)

### 3.6 Utilizar MPS no Código PyTorch

**Para executar modelos na GPU, mover modelo e tensors para o dispositivo MPS**:

```python
device = torch.device("mps")

# Mover modelo para MPS
model = YourModel().to(device)

# Mover dados para MPS
inputs, labels = inputs.to(device), labels.to(device)

# Forward pass
outputs = model(inputs)
```

Isso garante que computações sejam realizadas na GPU, aproveitando o backend MPS.

**Fonte**: [Apple Developer - Metal PyTorch](https://developer.apple.com/metal/pytorch/)

### 3.7 Considerações de Performance

**Nota Importante**:
- Embora MPS forneça aceleração GPU, ganhos de performance podem variar dependendo da carga de trabalho específica
- Alguns usuários reportaram speedups significativos
- Outros notaram que certas operações podem não estar totalmente otimizadas ainda
- É aconselhável fazer profiling de tarefas específicas para avaliar melhorias de performance

**Fonte**: [PyTorch Blog](https://pytorch.org/blog/introducing-accelerated-pytorch-training-on-mac/)

---

## 4. ⌨️ Comandos do Terminal para Otimização

### 4.1 Reduzir Overhead de System Logging

**Problema**: macOS gera dados de logging extensivos que podem consumir recursos do sistema.

**Solução**: Desativar subsistemas de logging desnecessários:

```bash
# Exemplo: desativar logging excessivo do CoreSuggestions
sudo log config --subsystem com.apple.CoreSuggestions --mode "level:off"
sudo log config --subsystem com.apple.CoreSuggestions --mode "persist:off"
sudo log config --subsystem com.apple.CoreSuggestions --mode "stream:default"
```

Após aplicar essas mudanças, deve-se notar redução no uso de CPU e disco.

**Fonte**: [Under Code Testing](https://undercodetesting.com/drastically-free-up-resources-on-your-macbook-osx-in-terminal/)

### 4.2 Limpar Memória Inativa

**Problema**: Com o tempo, memória inativa pode acumular, potencialmente desacelerando o sistema.

**Solução**:

```bash
sudo purge
```

Este comando força macOS a liberar memória inativa, tornando mais RAM disponível para processos ativos.

**Fonte**: [Tick Tech Told](https://www.ticktechtold.com/clean-mac-terminal-commands/)

### 4.3 Limpar Cache DNS

**Solução**: Limpar cache DNS pode resolver problemas de conectividade:

```bash
sudo dscacheutil -flushcache; sudo killall -HUP mDNSResponder
```

Este comando limpa o cache DNS e reinicia o resolvedor DNS.

**Fonte**: [Tick Tech Told](https://www.ticktechtold.com/clean-mac-terminal-commands/)

### 4.4 Desativar Animações de Janela

**Solução**: Reduzir ou desativar animações de janela pode tornar o sistema mais responsivo:

```bash
defaults write com.apple.dock launchanim -bool false; killall Dock
```

Este comando desativa a animação ao abrir aplicações do Dock.

**Fonte**: [Yahoo Lifestyle](https://www.yahoo.com/lifestyle/articles/fast-track-macs-performance-terminal-140000298.html)

### 4.5 Otimizar Performance do Finder

**Ajustar configurações do Finder pode melhorar sua responsividade**:

1. **Definir visualização padrão do Finder como List View**:
   ```bash
   defaults write com.apple.finder FXPreferredViewStyle -string "Nlsv"
   ```

2. **Desativar thumbnails de preview de arquivos**:
   ```bash
   defaults write com.apple.finder QLInlinePreview -bool false
   ```

3. **Reiniciar Finder para aplicar mudanças**:
   ```bash
   killall Finder
   ```

Esses ajustes podem acelerar tempos de carregamento de pastas.

**Fonte**: [iBoySoft](https://iboysoft.com/tips/terminal-commands-to-speed-up-mac.html)

### 4.6 Limpar Arquivos de Log do Sistema e Usuário

**Solução**: Com o tempo, arquivos de log podem acumular e consumir espaço em disco:

1. **Limpar arquivos de log do usuário**:
   ```bash
   sudo rm -rf ~/Library/Logs/*
   ```

2. **Limpar arquivos de log do sistema**:
   ```bash
   sudo rm -rf /private/var/log/*
   ```

Limpar esses logs regularmente pode liberar espaço em disco e melhorar performance.

**Fonte**: [Tick Tech Told](https://www.ticktechtold.com/clean-mac-terminal-commands/)

### 4.7 Ajustar Taxas de Repetição de Teclas

**Solução**: Aumentar taxa de repetição de teclas pode melhorar responsividade de digitação:

```bash
defaults write -g KeyRepeat -int 1
defaults write -g InitialKeyRepeat -int 10
```

Esses comandos definem taxa de repetição de teclas para configuração mais rápida que disponível através de System Settings.

**Fonte**: [AppleVis](https://applevis.com/guides/ten-advanced-hidden-tips-macos-hidden-applications-terminal-commands)

### 4.8 Desativar Dashboard

**Solução**: Se não usar o Dashboard, desativá-lo pode liberar recursos do sistema:

```bash
# Desativar
defaults write com.apple.dashboard mcx-disabled -boolean YES; killall Dock

# Reativar (se necessário)
defaults write com.apple.dashboard mcx-disabled -boolean NO; killall Dock
```

**Fonte**: [Hercules Technical Support](https://ts.hercules.com/faqs/eng/her_eng_00251.pdf)

### 4.9 Reset do System Management Controller (SMC)

**Nota Importante**: Para Macs com Apple Silicon (M1 e posteriores), o SMC reseta automaticamente ao reiniciar. Reiniciar Mac regularmente pode ajudar a manter performance ótima.

**Fonte**: [Simply Mac](https://www.simplymac.com/macbooks/how-to-speed-up-slow-macbook)

### ⚠️ Aviso sobre Comandos do Terminal

**Cuidado**: Embora esses comandos possam melhorar performance, usá-los com cuidado. Uso incorreto pode levar a comportamento não intencional do sistema. Sempre garantir backups de dados importantes antes de fazer mudanças significativas no sistema.

---

## 5. 🐳 Otimizações do Docker no M1

### 5.1 Atualizar Docker Desktop

**Solução**: Garantir uso da versão mais recente do Docker Desktop.

- Docker Desktop 4.22 inclui modo Resource Saver que automaticamente reduz utilização de CPU e memória quando containers não estão rodando
- Melhora eficiência significativamente

**Fonte**: [Docker Blog](https://www.docker.com/blog/unleash-docker-desktop-4-22-rapid-development/)

### 5.2 Ajustar Alocação de Recursos

**Soluções**:

1. **Configurações de Memória e CPU**:
   - Ajustar alocação de recursos do Docker baseado nas especificações do MacBook
   - **M1 MacBook Air 8GB RAM**: Alocar 2-3GB para Docker
   - **M1 Max 32GB RAM**: Pode alocar 8-12GB
   - Ajustar alocação de cores de CPU para balancear performance e responsividade do sistema

2. **Otimização de File Sharing**:
   - Limitar diretórios compartilhados com Docker apenas aos necessários para projetos
   - Esta prática reduz overhead e melhora performance

**Fonte**: 
- [ToolsTac](https://toolstac.com/howto/install-docker-mac-m1/mac-m1-installation-guide)
- [Life in Tech](https://www.lifeintech.com/2021/11/03/docker-performance-on-m1/)

### 5.3 Habilitar VirtioFS para File Sharing

**Solução**: Se usando macOS 12.5 ou posterior, habilitar VirtioFS nas configurações do Docker Desktop.

- Esta feature oferece ganhos substanciais de performance ao compartilhar arquivos entre host e containers
- Leva a tempos de build mais rápidos

**Fonte**: [Docker Blog](https://www.docker.com/blog/unleash-docker-desktop-4-22-rapid-development/)

### 5.4 Considerar Soluções Alternativas

**Solução**: Ferramentas como OrbStack oferecem gerenciamento dinâmico de memória.

- Aloca apenas memória que containers precisam
- Libera memória não utilizada de volta ao sistema
- Esta abordagem pode melhorar performance geral do sistema e prevenir lentidões

**Fonte**: [OrbStack Blog](https://orbstack.dev/blog/dynamic-memory)

### 5.5 Monitorar e Ajustar Configurações

**Solução**: Monitorar regularmente uso de recursos do Docker e ajustar configurações conforme necessário.

- Reduzir alocação de CPU
- Habilitar implementações eficientes de file sharing
- Ajustar alocação de memória
- Pode ajudar a manter performance ótima

**Fonte**: [Medium - Docker M1 Performance](https://medium.com/@sohail_saifi/why-docker-compose-is-actually-killing-your-m1-mac-the-performance-truth-no-one-talks-about-4357678c8584)

---

## 6. 📊 Resumo de Recomendações por Categoria

### 6.1 Prioridade Alta (Impacto Imediato)

1. ✅ Reduzir workers do servidor para 1-2
2. ✅ Fechar aplicações desnecessárias (especialmente navegador com muitas abas)
3. ✅ Limitar memória do Docker para 2-3GB (em M1 8GB)
4. ✅ Usar modelos ML menores (MiniLM ao invés de modelos grandes)
5. ✅ Habilitar MPS no PyTorch para usar GPU M1
6. ✅ Limpar cache Python/pip/conda regularmente

### 6.2 Prioridade Média (Melhorias Graduais)

1. ⚠️ Desativar animações e efeitos visuais
2. ⚠️ Organizar desktop e limitar janelas do Finder
3. ⚠️ Desativar itens de inicialização desnecessários
4. ⚠️ Usar VirtioFS no Docker (macOS 12.5+)
5. ⚠️ Limpar logs do sistema regularmente
6. ⚠️ Manter pelo menos 15-20% do SSD livre

### 6.3 Prioridade Baixa (Otimizações Finais)

1. ℹ️ Ajustar configurações do Finder
2. ℹ️ Desativar subsistemas de logging desnecessários
3. ℹ️ Considerar IDEs baseados em nuvem
4. ℹ️ Usar OrbStack ao invés de Docker Desktop (alternativa)
5. ℹ️ Ajustar taxas de repetição de teclas

---

## 7. 📚 Fontes e Referências

### Fontes Principais

1. **Kate's Tech Blog** - Otimização de performance para software pesado
   - https://kates-tech-blog.gitbook.io/web-design-blog-2024/tips-for-optimizing-performance-when-using-heavy-creative-software-on-a-macbook

2. **MacPaw** - Otimização do macOS Sequoia
   - https://macpaw.com/how-to/optimize-macos-sequoia

3. **Advice Scout** - Otimização de performance do Mac
   - https://www.advicescout.com/mac-performance-optimization/

4. **StarMorph Blog** - Guia de otimização de velocidade do Mac
   - https://blog.starmorph.com/blog/mac-speed-optimization-guide

5. **MacInfinity** - Maximizando performance de RAM do Mac
   - https://www.macinfinity.sg/post/maximizing-mac-ram-performance

6. **XDA Developers** - Formas de liberar memória macOS Apple Silicon
   - https://www.xda-developers.com/ways-free-up-macos-memory-apple-silicon/

7. **Medium - PyTorch M1 Setup** - Configuração de PyTorch no MacBook M1
   - https://medium.com/@ialwayslikedgrime/why-pytorch-is-better-than-tensorflow-successfully-setting-up-your-macbook-for-transformers-8e222bb21b22

8. **Apple Developer** - Metal PyTorch
   - https://developer.apple.com/metal/pytorch/

9. **PyTorch Blog** - Treinamento acelerado do PyTorch no Mac
   - https://pytorch.org/blog/introducing-accelerated-pytorch-training-on-mac/

10. **Under Code Testing** - Liberar recursos drasticamente no MacBook
    - https://undercodetesting.com/drastically-free-up-resources-on-your-macbook-osx-in-terminal/

11. **Tick Tech Told** - Comandos de limpeza do Mac no Terminal
    - https://www.ticktechtold.com/clean-mac-terminal-commands/

12. **Docker Blog** - Docker Desktop 4.22
    - https://www.docker.com/blog/unleash-docker-desktop-4-22-rapid-development/

13. **OrbStack Blog** - Gerenciamento dinâmico de memória
    - https://orbstack.dev/blog/dynamic-memory

14. **Medium - Docker M1 Performance** - Verdade sobre performance do Docker no M1
    - https://medium.com/@sohail_saifi/why-docker-compose-is-actually-killing-your-m1-mac-the-performance-truth-no-one-talks-about-4357678c8584

---

## 8. ⚠️ Avisos e Considerações Importantes

### 8.1 Comandos do Terminal

- **Sempre fazer backup** antes de executar comandos que modificam configurações do sistema
- **Testar em ambiente de desenvolvimento** antes de aplicar em produção
- **Documentar mudanças** para poder reverter se necessário

### 8.2 Otimizações de Memória

- Algumas otimizações podem afetar funcionalidade de certas aplicações
- Monitorar impacto após aplicar mudanças
- Reverter se causar problemas

### 8.3 Docker

- Limites muito baixos de memória podem causar problemas em containers
- Ajustar gradualmente e monitorar performance
- Considerar alternativas como OrbStack se Docker Desktop continuar problemático

### 8.4 PyTorch MPS

- Nem todas as operações PyTorch estão otimizadas para MPS ainda
- Algumas operações podem ser mais lentas no MPS que na CPU
- Fazer profiling de código específico para determinar melhor dispositivo

---

## 9. 📝 Checklist de Implementação

### Fase 1: Otimizações Rápidas (Sem Risco)
- [ ] Fechar aplicações desnecessárias
- [ ] Reduzir abas do navegador
- [ ] Limpar cache Python: `find . -type d -name __pycache__ -exec rm -r {} +`
- [ ] Verificar uso de memória: `python scripts/monitor_memory.py`
- [ ] Limpar cache pip: `pip cache purge`

### Fase 2: Configurações do Sistema (Baixo Risco)
- [ ] Desativar animações: System Settings > Accessibility > Display
- [ ] Desativar itens de inicialização desnecessários
- [ ] Organizar desktop e arquivos
- [ ] Verificar espaço livre no disco (manter 15-20% livre)

### Fase 3: Configurações do Projeto (Médio Risco)
- [ ] Reduzir `SERVER_WORKERS` para 1 no `.env`
- [ ] Reduzir `DB_POOL_SIZE` para 5 no `.env`
- [ ] Configurar variáveis de ambiente de threads (OMP_NUM_THREADS, etc.)
- [ ] Limitar memória do Docker para 2-3GB

### Fase 4: Otimizações Avançadas (Requer Testes)
- [ ] Configurar PyTorch para usar MPS
- [ ] Usar modelos ML menores
- [ ] Habilitar VirtioFS no Docker
- [ ] Aplicar comandos do terminal (com backup)

---

## 10. 🔄 Próximos Passos Recomendados

1. **Revisar este relatório** e identificar otimizações relevantes para seu caso
2. **Começar com otimizações de baixo risco** (Fase 1 e 2)
3. **Monitorar impacto** usando scripts de monitoramento
4. **Aplicar otimizações de projeto** gradualmente (Fase 3)
5. **Testar otimizações avançadas** em ambiente de desenvolvimento (Fase 4)
6. **Documentar resultados** para referência futura

---

**Nota Final**: Este relatório é baseado em pesquisas da internet e não substitui testes específicos no seu ambiente. Sempre testar mudanças em ambiente de desenvolvimento antes de aplicar em produção.

**Última Atualização**: 2025-01-22
