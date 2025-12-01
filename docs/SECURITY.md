# Segurança - Parle Backend

Este documento descreve as práticas de segurança implementadas no Parle Backend.

## 🔒 Modelo de Segurança

### Autenticação

O Parle Backend usa **JWT (JSON Web Tokens)** para autenticação:

- Tokens são assinados com `JWT_SECRET_KEY`
- Tokens expiram após período configurado
- Tokens são validados em cada requisição autenticada

**⚠️ IMPORTANTE**: Em produção, `JWT_SECRET_KEY` deve ser:
- Forte (mínimo 32 caracteres aleatórios)
- Único por ambiente
- Armazenado de forma segura (secrets management)
- Nunca commitado no repositório

### Autorização

- Endpoints protegidos requerem token JWT válido
- Rate limiting por IP e por usuário
- Validação de permissões por endpoint

### Proteção de Dados

#### SQL Injection

**Proteção**: Uso de SQLAlchemy ORM previne SQL injection através de:
- Parameterized queries
- Type-safe queries
- Escaping automático

**Exemplo seguro**:
```python
# ✅ Seguro
user = session.query(User).filter(User.email == email).first()

# ❌ Nunca faça isso
query = f"SELECT * FROM users WHERE email = '{email}'"
```

#### XSS (Cross-Site Scripting)

**Proteção**:
- FastAPI escapa automaticamente respostas JSON
- Headers de segurança configurados
- Validação de inputs

#### CSRF (Cross-Site Request Forgery)

**Proteção**:
- Tokens JWT em headers (não cookies)
- CORS configurado adequadamente
- Validação de origem quando necessário

### Criptografia

#### Dados em Trânsito

- **HTTPS obrigatório em produção**
- TLS 1.2+ configurado
- Certificados SSL válidos (Let's Encrypt recomendado)

#### Dados em Repouso

- Senhas: Hash bcrypt (nunca armazenadas em texto plano)
- Tokens: Assinados, não criptografados (JWT)
- Dados sensíveis: Criptografados se necessário (usar biblioteca de criptografia)

### Rate Limiting

Proteção contra abuso através de rate limiting:

- **Por IP**: Limites gerais por endereço IP
- **Por Usuário**: Limites mais altos para usuários autenticados
- **Por Endpoint**: Limites específicos por endpoint crítico

**Configuração**:
```python
# src/core/constants.py
CONVERSATION_RATE_LIMIT = "10/minute"
AUTH_LOGIN_RATE_LIMIT = "5/minute"
```

## 🛡️ Headers de Segurança

O Parle Backend inclui os seguintes headers de segurança:

- `X-Content-Type-Options: nosniff` - Previne MIME sniffing
- `X-Frame-Options: DENY` - Previne clickjacking
- `X-XSS-Protection: 1; mode=block` - Proteção XSS
- `Referrer-Policy: strict-origin-when-cross-origin` - Controle de referrer
- `Strict-Transport-Security` - HSTS (apenas em produção com HTTPS)

## 🔐 Gestão de Credenciais

### Variáveis de Ambiente

**Nunca commite**:
- API keys
- Secrets
- Credenciais de banco de dados
- Tokens de acesso

**Use**:
- `.env` para desenvolvimento local (não commitado)
- Secrets management em produção (AWS Secrets Manager, HashiCorp Vault, etc.)
- Variáveis de ambiente do sistema/container

### Rotação de Chaves

- Rotacione `JWT_SECRET_KEY` periodicamente
- Rotacione API keys quando comprometidas
- Mantenha histórico de chaves antigas para tokens ainda válidos

## 🚨 Modelo de Ameaças

### Ameaças Identificadas

1. **Ataques de Força Bruta**
   - **Mitigação**: Rate limiting em endpoints de autenticação

2. **Token Theft**
   - **Mitigação**: Tokens com expiração, HTTPS obrigatório

3. **SQL Injection**
   - **Mitigação**: SQLAlchemy ORM, validação de inputs

4. **XSS**
   - **Mitigação**: Escaping automático, headers de segurança

5. **DDoS**
   - **Mitigação**: Rate limiting, load balancer, CDN

6. **Credential Stuffing**
   - **Mitigação**: Rate limiting, monitoramento de tentativas falhas

## 📋 Checklist de Segurança

### Desenvolvimento

- [ ] Nenhuma credencial commitada
- [ ] Variáveis de ambiente configuradas
- [ ] Inputs validados
- [ ] Erros não expõem informações sensíveis
- [ ] Logs não contêm dados sensíveis

### Produção

- [ ] HTTPS configurado e funcionando
- [ ] `JWT_SECRET_KEY` forte e único
- [ ] Rate limiting ativo
- [ ] Firewall configurado
- [ ] Logs de segurança monitorados
- [ ] Backup de dados configurado
- [ ] Plano de resposta a incidentes

## 🚨 Resposta a Incidentes

### Plano de Resposta

1. **Identificação**: Detectar incidente através de logs/monitoramento
2. **Contenção**: Isolar sistema afetado se necessário
3. **Eradicação**: Remover ameaça (revogar tokens, bloquear IPs, etc.)
4. **Recuperação**: Restaurar serviços
5. **Pós-Incidente**: Documentar, analisar, melhorar

### Contatos

- **Equipe de Segurança**: security@parle.ai
- **On-Call**: Ver runbook de incidentes

## 📚 Referências

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [FastAPI Security](https://fastapi.tiangolo.com/tutorial/security/)
- [JWT Best Practices](https://datatracker.ietf.org/doc/html/rfc8725)

## 🔍 Auditoria

### Logs de Segurança

O sistema registra:
- Tentativas de login (sucesso e falha)
- Ações administrativas
- Acessos a dados sensíveis
- Violações de rate limit

### Revisão Periódica

- Revisar logs de segurança mensalmente
- Auditar permissões trimestralmente
- Atualizar dependências regularmente
- Executar scans de vulnerabilidade
