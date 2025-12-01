# Criptografia de Dados - Parle Backend

Este documento descreve o estado atual da criptografia de dados sensíveis em repouso.

## 📋 Visão Geral

Este documento verifica e documenta como dados sensíveis são protegidos no Parle Backend.

## 🔒 Dados Sensíveis Identificados

### 1. Senhas de Usuário

**Status**: ✅ Protegido

**Implementação**:
- Senhas são armazenadas como hash bcrypt
- Nunca armazenadas em texto plano
- Hash inclui salt automático

**Localização**: `src/modules/auth/user_module.py`

**Verificação**:
```python
# Senhas são hasheadas antes de armazenar
from passlib.context import CryptContext
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
hashed = pwd_context.hash(password)
```

### 2. Tokens JWT

**Status**: ✅ Protegido

**Implementação**:
- Tokens são assinados (não criptografados)
- Assinatura com JWT_SECRET_KEY
- Tokens expiram automaticamente
- Não armazenados no servidor (stateless)

**Verificação**: Tokens são gerados e validados, mas não armazenados.

### 3. Dados de Conversação

**Status**: ⚠️ Depende da configuração

**Implementação Atual**:
- Armazenados em banco de dados (SQLite/PostgreSQL)
- Não criptografados por padrão
- Depende de criptografia do banco de dados

**Recomendação**:
- Usar PostgreSQL com TDE (Transparent Data Encryption)
- Ou criptografar campos sensíveis antes de armazenar

### 4. Arquivos de Áudio

**Status**: ⚠️ Depende do storage

**Implementação Atual**:
- Armazenados em Backblaze B2 ou local
- Não criptografados por padrão

**Recomendação**:
- Usar storage com criptografia (S3 com SSE, B2 com encryption)
- Ou criptografar antes de upload

### 5. API Keys e Secrets

**Status**: ✅ Protegido

**Implementação**:
- Armazenados em variáveis de ambiente
- Nunca commitados no repositório
- Usados apenas em runtime

**Verificação**: `.env` está no `.gitignore`

## 🔐 Criptografia em Repouso

### Banco de Dados

**SQLite**:
- Não tem criptografia nativa
- Recomendação: Usar PostgreSQL em produção

**PostgreSQL**:
- Suporta TDE (Transparent Data Encryption)
- Configurar via `postgresql.conf`
- Ou usar serviço gerenciado (RDS, Cloud SQL) com encryption

### Arquivos

**Local Storage**:
- Não criptografado por padrão
- Recomendação: Usar filesystem encryption (LUKS, BitLocker)

**Cloud Storage**:
- Backblaze B2: Suporta encryption
- AWS S3: Server-Side Encryption (SSE)
- Google Cloud Storage: Encryption at rest

## 🛡️ Recomendações de Implementação

### 1. Criptografar Campos Sensíveis

Para dados muito sensíveis, criptografar antes de armazenar:

```python
from cryptography.fernet import Fernet

# Gerar chave (armazenar em variável de ambiente)
key = os.getenv("ENCRYPTION_KEY")
cipher = Fernet(key)

# Criptografar
encrypted_data = cipher.encrypt(sensitive_data.encode())

# Descriptografar
decrypted_data = cipher.decrypt(encrypted_data).decode()
```

### 2. Usar PostgreSQL com Encryption

```bash
# Configurar PostgreSQL com encryption
# postgresql.conf
ssl = on
ssl_cert_file = '/path/to/cert.pem'
ssl_key_file = '/path/to/key.pem'
```

### 3. Criptografar Arquivos

```python
# Antes de upload
from cryptography.fernet import Fernet

cipher = Fernet(encryption_key)
encrypted_file = cipher.encrypt(file_content)

# Upload encrypted file
storage.upload(encrypted_file)
```

## ✅ Checklist de Verificação

- [x] Senhas são hasheadas (bcrypt)
- [x] Tokens JWT são assinados
- [x] Secrets em variáveis de ambiente
- [ ] Dados de conversação criptografados (depende de configuração)
- [ ] Arquivos de áudio criptografados (depende de storage)
- [ ] Banco de dados com encryption (recomendado para produção)

## 📚 Referências

- [OWASP Data Protection](https://owasp.org/www-project-web-security-testing-guide/latest/4-Web_Application_Security_Testing/09-Testing_for_Weak_Cryptography/README)
- [PostgreSQL Encryption](https://www.postgresql.org/docs/current/encryption-options.html)
- [Python Cryptography](https://cryptography.io/)
