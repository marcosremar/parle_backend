# Guia de Execução de Testes

## Executar no GitHub Codespace

### Opção 1: Script Automatizado (Recomendado)

```bash
# Dar permissão de execução
chmod +x scripts/run_tests_codespace.sh

# Executar todos os testes
./scripts/run_tests_codespace.sh
```

### Opção 2: Testes Rápidos (Apenas Críticos)

```bash
chmod +x scripts/run_tests_quick.sh
./scripts/run_tests_quick.sh
```

### Opção 3: Comando Manual

```bash
# Todos os testes
python -m pytest tests/ -v --tb=short -q

# Apenas testes unitários (mais rápidos)
python -m pytest tests/unit/ -v -q

# Apenas testes de integração
python -m pytest tests/integration/ -v -q

# Testes específicos
python -m pytest tests/unit/speech/ -v
```

## Executar via GitHub Actions

Os testes são executados automaticamente via GitHub Actions quando você:
- Faz push para `main`, `develop`, ou `monolito-modular`
- Abre um Pull Request
- Dispara manualmente via `workflow_dispatch`

Veja o workflow em: `.github/workflows/test.yml`

## Estrutura de Testes

```
tests/
├── unit/              # Testes unitários (rápidos)
│   ├── speech/       # STT e TTS
│   └── orchestrator/ # Talkers e process_turn
├── integration/       # Testes de integração
├── performance/      # Testes de performance
├── quality/          # Testes de qualidade
├── robustness/       # Testes de robustez
└── regression/       # Testes de regressão
```

## Dicas para Execução Rápida

1. **Testes mais rápidos primeiro**: Comece com `tests/unit/`
2. **Limitar falhas**: Use `--maxfail=5` para parar após 5 falhas
3. **Modo quieto**: Use `-q` para menos output
4. **Apenas falhas**: Use `--tb=no` para ver apenas resumo

## Troubleshooting

### Erro: "Module not found"
```bash
# Certifique-se de estar no diretório raiz
cd /workspaces/parle_backend
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Erro: "Fixture not found"
```bash
# Limpar cache do pytest
pytest --cache-clear
```

### Testes muito lentos
```bash
# Executar apenas testes unitários
pytest tests/unit/ -q

# Ou usar paralelização (se disponível)
pytest -n auto tests/unit/ -q
```
