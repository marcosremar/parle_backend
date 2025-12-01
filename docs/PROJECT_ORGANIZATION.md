# Organização do Projeto

Este documento descreve a estrutura organizacional do projeto Parle Backend.

## 📁 Estrutura de Diretórios

### Raiz do Projeto

A raiz contém apenas arquivos essenciais:

```
parle_backend/
├── README.md                    # Documentação principal
├── LICENSE.txt                  # Licença do projeto
├── CONTRIBUTING.md             # Guia de contribuição
├── main.sh                     # Script principal de gerenciamento
├── pyproject.toml              # Configuração Python (ruff, black, mypy)
├── .pre-commit-config.yaml     # Configuração pre-commit
├── .coveragerc                 # Configuração de cobertura
├── .gitignore                  # Arquivos ignorados pelo Git
├── .gitmodules                 # Submódulos Git (se houver)
├── environment.yml             # Ambiente Conda
├── requirements.txt            # Dependências de produção
├── requirements-dev.txt        # Dependências de desenvolvimento
├── requirements-test.txt       # Dependências de testes
├── docker/                     # Arquivos Docker
├── config/                     # Configurações
├── docs/                       # Documentação
├── scripts/                    # Scripts de automação
├── src/                        # Código fonte
└── tests/                      # Testes
```

### Diretórios Principais

#### `docker/`
Todos os arquivos relacionados ao Docker:
- `Dockerfile` - Imagem de produção
- `Dockerfile.dev` - Imagem de desenvolvimento
- `docker-compose.yml` - Orquestração principal
- `docker-compose.logging.yml` - Stack de logging
- `.dockerignore` - Arquivos excluídos do build

#### `docs/`
Documentação completa do projeto:
- `status/` - Relatórios de status e implementação
- `ADRs/` - Architecture Decision Records
- Guias operacionais (deployment, troubleshooting, etc.)
- Documentação técnica (security, performance, etc.)

#### `config/`
Configurações do projeto:
- `settings.yaml` - Configurações principais
- `archive/` - Configurações legadas

#### `scripts/`
Scripts de automação e utilitários:
- Setup e instalação
- Análise e auditoria
- Testes e validação
- Limpeza e manutenção

#### `src/`
Código fonte principal:
- `api/` - API FastAPI
- `core/` - Biblioteca core compartilhada
- `modules/` - Módulos do sistema

#### `tests/`
Testes do projeto:
- `unit/` - Testes unitários
- `integration/` - Testes de integração
- `e2e/` - Testes end-to-end
- `performance/` - Testes de performance
- `fixtures/` - Dados de teste

## 🗑️ Arquivos Não Necessários na Raiz

Os seguintes arquivos foram movidos ou removidos:

- ✅ Relatórios de status → `docs/status/`
- ✅ `.claude.md` → `docs/.claude.md`
- ✅ `QUALITY_CHECKLIST.md` → `docs/status/QUALITY_CHECKLIST.md`

## 📝 Convenções

### Nomes de Arquivos

- **Documentação**: `UPPER_CASE.md` para documentos principais
- **Scripts**: `snake_case.sh` ou `snake_case.py`
- **Configuração**: `kebab-case.yaml` ou `snake_case.toml`

### Organização de Documentação

- **Status/Relatórios**: `docs/status/`
- **Guias Operacionais**: `docs/` (raiz)
- **ADRs**: `docs/ADRs/`
- **Técnicos**: `docs/` (raiz)

## 🔄 Manutenção

Para manter a organização:

1. **Novos documentos de status** → `docs/status/`
2. **Novos guias** → `docs/`
3. **Novos scripts** → `scripts/`
4. **Novos testes** → `tests/` (subdiretório apropriado)

---

*Última atualização: 2025-01-XX*
