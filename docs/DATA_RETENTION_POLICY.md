# Política de Retenção de Dados - Parle Backend

Este documento define a política de retenção de dados do Parle Backend.

## 📋 Visão Geral

A política de retenção define por quanto tempo diferentes tipos de dados são mantidos antes de serem removidos ou arquivados.

## ⏱️ Períodos de Retenção

### Dados de Usuário

| Tipo de Dado | Período de Retenção | Ação Após Expiração |
|--------------|---------------------|---------------------|
| Conta de usuário ativa | Indefinido | Manter |
| Conta inativa | 2 anos | Arquivar, depois deletar |
| Dados de autenticação | 1 ano após último login | Deletar logs de autenticação |
| Tokens JWT | 24 horas (padrão) | Expirar automaticamente |

### Dados de Conversação

| Tipo de Dado | Período de Retenção | Ação Após Expiração |
|--------------|---------------------|---------------------|
| Conversações ativas | Indefinido | Manter |
| Conversações inativas | 1 ano | Arquivar |
| Mensagens | 1 ano após última mensagem | Arquivar, depois deletar |
| Áudio de conversações | 6 meses | Deletar arquivos de áudio |
| Transcrições STT | 1 ano | Deletar |

### Dados de Logs

| Tipo de Dado | Período de Retenção | Ação Após Expiração |
|--------------|---------------------|---------------------|
| Logs de aplicação | 30 dias | Deletar |
| Logs de erro | 90 dias | Deletar |
| Audit logs | 2 anos | Arquivar, depois deletar |
| Logs de acesso | 1 ano | Deletar |

### Dados Temporários

| Tipo de Dado | Período de Retenção | Ação Após Expiração |
|--------------|---------------------|---------------------|
| Sessões ativas | 24 horas | Expirar |
| Sessões inativas | 7 dias | Deletar |
| Cache | Variável (TTL) | Expirar automaticamente |
| Arquivos temporários | 24 horas | Deletar |

## 🔧 Implementação

### Script de Limpeza Automática

O script `scripts/cleanup_old_data.py` implementa a política de retenção:

```bash
# Executar limpeza manualmente
python scripts/cleanup_old_data.py

# Ou via cron (diariamente às 2 AM)
0 2 * * * cd /path/to/parle_backend && python scripts/cleanup_old_data.py
```

### Configuração

Períodos de retenção podem ser configurados via variáveis de ambiente:

```bash
# Retenção de conversações (dias)
CONVERSATION_RETENTION_DAYS=365

# Retenção de logs (dias)
LOG_RETENTION_DAYS=30

# Retenção de audit logs (dias)
AUDIT_LOG_RETENTION_DAYS=730
```

## 📊 Processo de Arquivamento

### Antes de Deletar

1. **Arquivar dados importantes**:
   - Exportar para formato estruturado (JSON, CSV)
   - Armazenar em storage de longo prazo (S3, Backblaze B2)
   - Manter metadados mínimos

2. **Notificar usuário** (se aplicável):
   - Email antes de deletar dados do usuário
   - Opção de exportar dados

3. **Backup final**:
   - Criar backup antes de deletar
   - Manter backup por período adicional (30 dias)

## 🗑️ Processo de Deleção

### Dados Permanentes

Alguns dados são deletados permanentemente:
- Logs após período de retenção
- Arquivos temporários
- Cache expirado

### Dados de Usuário

Dados de usuário seguem processo especial:
1. **Aviso**: Notificar usuário 30 dias antes
2. **Exportação**: Oferecer exportação de dados
3. **Anonimização**: Anonimizar dados sensíveis
4. **Deleção**: Deletar após período de retenção

## 🔒 Segurança

### Dados Sensíveis

- **Senhas**: Nunca armazenadas, apenas hashes (bcrypt)
- **Tokens**: Expirar automaticamente
- **Dados pessoais**: Criptografados em repouso (se configurado)

### Conformidade

- **LGPD/GDPR**: Respeitar direitos de usuário
- **Direito ao esquecimento**: Implementar deleção sob demanda
- **Portabilidade**: Permitir exportação de dados

## 📝 Logs de Limpeza

Todas as operações de limpeza são registradas:

```json
{
  "timestamp": "2025-01-XX",
  "action": "cleanup",
  "data_type": "conversations",
  "records_deleted": 1234,
  "retention_period": "365 days"
}
```

## 🎯 Exceções

### Dados Críticos

Alguns dados nunca são deletados automaticamente:
- Configurações do sistema
- Métricas agregadas (sem dados pessoais)
- Logs de segurança críticos

### Requisições Legais

Dados podem ser mantidos além do período normal se:
- Requisição legal pendente
- Investigação de segurança
- Disputa em andamento

## 📚 Referências

- [LGPD - Lei Geral de Proteção de Dados](https://www.planalto.gov.br/ccivil_03/_ato2015-2018/2018/lei/l13709.htm)
- [GDPR - General Data Protection Regulation](https://gdpr.eu/)
