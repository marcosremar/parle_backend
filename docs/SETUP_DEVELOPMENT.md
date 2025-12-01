# Setup do Ambiente de Desenvolvimento

Este guia descreve como configurar o ambiente de desenvolvimento completo do Parle Backend.

## 📋 Pré-requisitos

- Python 3.10+ (3.11 recomendado)
- pip
- Git
- (Opcional) Conda

## 🚀 Setup Automático

Execute o script de setup:

```bash
./scripts/setup_dev_environment.sh
```

Este script irá:
1. Verificar versão do Python
2. Instalar dependências de produção
3. Instalar dependências de desenvolvimento
4. Instalar e configurar pre-commit hooks
5. Executar auditoria de dependências

## 🔧 Setup Manual

Se preferir fazer manualmente:

### 1. Instalar Dependências

```bash
# Dependências de produção
pip install -r requirements.txt

# Dependências de desenvolvimento
pip install -r requirements-dev.txt

# Dependências de testes
pip install -r requirements-test.txt
```

### 2. Configurar Pre-commit Hooks

```bash
# Instalar pre-commit
pip install pre-commit

# Instalar hooks
pre-commit install

# (Opcional) Executar em todos os arquivos
pre-commit run --all-files
```

### 3. Configurar Variáveis de Ambiente

```bash
# Copiar arquivo de exemplo
cp .env.example .env

# Editar com suas configurações
nano .env  # ou use seu editor preferido
```

### 4. Verificar Instalação

```bash
# Executar testes
pytest tests/unit/ -v

# Verificar linting
ruff check src/

# Verificar formatação
black --check src/

# Verificar tipos
mypy src/
```

## 🔍 Auditoria de Dependências

Execute auditoria regularmente:

```bash
# Auditoria de dependências de produção
pip-audit -r requirements.txt

# Auditoria de dependências de desenvolvimento
pip-audit -r requirements-dev.txt
```

Ou use o workflow do GitHub Actions que executa automaticamente semanalmente.

## ✅ Verificação Final

Após o setup, verifique:

- [ ] Python 3.10+ instalado
- [ ] Todas as dependências instaladas
- [ ] Pre-commit hooks funcionando
- [ ] Variáveis de ambiente configuradas
- [ ] Testes passando
- [ ] Linting sem erros

## 🐛 Problemas Comuns

### Pre-commit não funciona

```bash
# Reinstalar hooks
pre-commit uninstall
pre-commit install
```

### Erros de importação

```bash
# Verificar PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Ou executar com -m
python -m src.api.main
```

### Dependências conflitantes

```bash
# Criar ambiente virtual limpo
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Reinstalar dependências
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## 📚 Próximos Passos

1. Leia [CONTRIBUTING.md](../CONTRIBUTING.md) para guia de contribuição
2. Explore a [documentação da API](./API.md)
3. Veja [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) se tiver problemas

## 🔗 Recursos Adicionais

- [Python Virtual Environments](https://docs.python.org/3/tutorial/venv.html)
- [Pre-commit Documentation](https://pre-commit.com/)
- [Ruff Documentation](https://docs.astral.sh/ruff/)
