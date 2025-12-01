# Security Scanning - Parle Backend

Este documento descreve como executar scans de vulnerabilidades e pentests.

## 🔒 Scans de Vulnerabilidades

### Ferramentas Configuradas

1. **Bandit** - Análise estática de segurança Python
2. **pip-audit** - Auditoria de dependências
3. **CodeQL** - Análise de segurança (GitHub Actions)
4. **Dependabot** - Atualizações automáticas de segurança

### Executar Scans

#### Bandit (Vulnerabilidades Python)

```bash
# Instalar
pip install bandit

# Executar scan
bandit -r src/ -f json -o bandit-report.json

# Ver relatório
cat bandit-report.json
```

#### pip-audit (Vulnerabilidades de Dependências)

```bash
# Instalar
pip install pip-audit

# Auditar dependências
pip-audit -r requirements.txt --format json --output pip-audit-report.json

# Ver relatório
cat pip-audit-report.json
```

#### CodeQL (GitHub Actions)

CodeQL é executado automaticamente via GitHub Actions:
- `.github/workflows/codeql.yml`
- Executa em push, PR, e semanalmente

### Resultados

Os relatórios são gerados em:
- `bandit-report.json` - Vulnerabilidades Python
- `pip-audit-report.json` - Vulnerabilidades de dependências
- GitHub Security tab - CodeQL findings

## 🎯 Pentest

### Quando Realizar Pentest

- Antes de deploy em produção
- Após mudanças significativas
- Anualmente ou conforme política de segurança

### Escopo de Pentest

1. **Autenticação e Autorização**
   - Testar força bruta
   - Testar bypass de autenticação
   - Testar escalação de privilégios

2. **Input Validation**
   - SQL Injection
   - XSS
   - Command Injection
   - Path Traversal

3. **API Security**
   - Rate limiting bypass
   - CSRF
   - IDOR (Insecure Direct Object Reference)

4. **Infrastructure**
   - Exposição de informações
   - Configurações inseguras
   - Secrets em logs/código

### Ferramentas Recomendadas

1. **OWASP ZAP** - Automated security testing
2. **Burp Suite** - Manual testing
3. **Nmap** - Network scanning
4. **SQLMap** - SQL injection testing

### Checklist de Pentest

- [ ] Autenticação e autorização
- [ ] Input validation
- [ ] API security
- [ ] Infrastructure security
- [ ] Data protection
- [ ] Error handling
- [ ] Logging e monitoramento

## 📋 Processo de Remediação

1. **Priorizar**: Críticas > Altas > Médias > Baixas
2. **Corrigir**: Implementar correções
3. **Verificar**: Re-executar scans
4. **Documentar**: Registrar correções

## 🔗 Recursos

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [Bandit Documentation](https://bandit.readthedocs.io/)
- [pip-audit Documentation](https://pypi.org/project/pip-audit/)

---

*Documento criado em: 2025-01-XX*
