# Guia de Contribuição - Parle Backend

Obrigado por considerar contribuir para o Parle Backend! Este documento fornece diretrizes para contribuir com o projeto.

## 📋 Índice

- [Código de Conduta](#código-de-conduta)
- [Como Contribuir](#como-contribuir)
- [Configuração do Ambiente](#configuração-do-ambiente)
- [Padrões de Código](#padrões-de-código)
- [Processo de Pull Request](#processo-de-pull-request)
- [Testes](#testes)
- [Documentação](#documentação)

## Código de Conduta

Este projeto adere a um Código de Conduta. Ao participar, você concorda em manter este código.

## Como Contribuir

### Reportar Bugs

1. Verifique se o bug já não foi reportado nas [Issues](https://github.com/parle/parle_backend/issues)
2. Se não existir, crie uma nova issue com:
   - Descrição clara do problema
   - Passos para reproduzir
   - Comportamento esperado vs. atual
   - Ambiente (OS, Python version, etc.)
   - Logs relevantes (se aplicável)

### Sugerir Melhorias

1. Verifique se a sugestão já não existe
2. Crie uma issue descrevendo:
   - O problema que a melhoria resolve
   - Proposta de solução
   - Benefícios esperados

### Contribuir com Código

1. Fork o repositório
2. Crie uma branch para sua feature (`git checkout -b feature/nova-feature`)
3. Faça suas alterações seguindo os padrões do projeto
4. Adicione testes para suas alterações
5. Certifique-se de que todos os testes passam
6. Commit suas alterações com mensagens descritivas
7. Push para sua branch (`git push origin feature/nova-feature`)
8. Abra um Pull Request

## Configuração do Ambiente

### Pré-requisitos

- Python 3.10+ (3.11 recomendado)
- Git
- Conda (opcional, mas recomendado)

### Setup Inicial

```bash
# 1. Clone o repositório
git clone https://github.com/parle/parle_backend.git
cd parle_backend

# 2. Configure o ambiente
./main.sh setup

# 3. Instale pre-commit hooks
pip install pre-commit
pre-commit install

# 4. Instale dependências de desenvolvimento
pip install -r requirements-dev.txt
```

### Variáveis de Ambiente

Copie `.env.example` para `.env` e configure as variáveis necessárias:

```bash
cp .env.example .env
# Edite .env com suas configurações
```

## Padrões de Código

### Formatação

O projeto usa:
- **Black** para formatação automática (line length: 100)
- **Ruff** para linting
- **mypy** para type checking

```bash
# Formatar código
black src/ tests/

# Verificar linting
ruff check src/

# Verificar tipos
mypy src/
```

### Estrutura de Commits

Use mensagens de commit descritivas seguindo o padrão:

```
tipo(escopo): descrição curta

Descrição mais detalhada (opcional)

Fixes #123
```

Tipos:
- `feat`: Nova feature
- `fix`: Correção de bug
- `docs`: Documentação
- `style`: Formatação
- `refactor`: Refatoração
- `test`: Testes
- `chore`: Manutenção

### Docstrings

Use Google-style docstrings para todas as funções públicas:

```python
def minha_funcao(param1: str, param2: int) -> dict:
    """
    Descrição curta da função.
    
    Args:
        param1: Descrição do parâmetro 1
        param2: Descrição do parâmetro 2
        
    Returns:
        Descrição do retorno
        
    Raises:
        ValueError: Quando algo dá errado
        
    Example:
        >>> resultado = minha_funcao("teste", 42)
        >>> print(resultado)
        {'status': 'ok'}
    """
    pass
```

## Processo de Pull Request

1. **Atualize sua branch**: `git pull origin main`
2. **Execute testes**: `pytest`
3. **Verifique qualidade**: `ruff check src/ && black --check src/`
4. **Atualize documentação** se necessário
5. **Crie o PR** com descrição clara do que foi alterado
6. **Aguarde review** e responda aos comentários

### Checklist do PR

- [ ] Código segue os padrões do projeto
- [ ] Testes foram adicionados/atualizados
- [ ] Todos os testes passam
- [ ] Documentação foi atualizada
- [ ] Pre-commit hooks passaram
- [ ] CI/CD passou

## Testes

### Executar Testes

```bash
# Todos os testes
pytest

# Testes específicos
pytest tests/unit/
pytest tests/integration/

# Com cobertura
pytest --cov=src --cov-report=html
```

### Escrever Testes

- Use `pytest` e `pytest-asyncio` para testes assíncronos
- Mantenha cobertura acima de 80%
- Teste casos de sucesso e erro
- Use fixtures do `conftest.py` quando possível

## Documentação

- Atualize docstrings ao modificar funções
- Atualize README.md se necessário
- Adicione exemplos de uso quando relevante
- Documente decisões arquiteturais em `docs/ADRs/`

## Perguntas?

Se tiver dúvidas, abra uma issue ou entre em contato com a equipe.

Obrigado por contribuir! 🎉
