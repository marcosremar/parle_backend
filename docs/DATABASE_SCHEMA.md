# Esquema do Banco de Dados - Parle Backend

Este documento descreve o esquema do banco de dados usado pelo Parle Backend.

## 📋 Visão Geral

O Parle Backend usa SQLAlchemy como ORM e suporta múltiplos backends de banco de dados:
- **SQLite**: Padrão para desenvolvimento
- **PostgreSQL**: Recomendado para produção
- **MySQL**: Suportado via SQLAlchemy

## 🗄️ Tabelas Principais

### users

Armazena informações de usuários.

```sql
CREATE TABLE users (
    id VARCHAR PRIMARY KEY,
    username VARCHAR UNIQUE NOT NULL,
    email VARCHAR UNIQUE NOT NULL,
    password_hash VARCHAR NOT NULL,
    full_name VARCHAR,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);
```

**Campos**:
- `id`: UUID do usuário
- `username`: Nome de usuário único
- `email`: Email único
- `password_hash`: Hash bcrypt da senha
- `full_name`: Nome completo (opcional)
- `created_at`: Data de criação
- `updated_at`: Data de última atualização
- `is_active`: Status ativo/inativo

### conversations

Armazena conversações entre usuários e o sistema.

```sql
CREATE TABLE conversations (
    id VARCHAR PRIMARY KEY,
    user_id VARCHAR NOT NULL,
    title VARCHAR,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSON,
    FOREIGN KEY (user_id) REFERENCES users(id)
);
```

**Campos**:
- `id`: UUID da conversação
- `user_id`: ID do usuário (FK para users)
- `title`: Título da conversação (opcional)
- `created_at`: Data de criação
- `updated_at`: Data de última atualização
- `metadata`: Dados adicionais em JSON

### messages

Armazena mensagens individuais dentro de conversações.

```sql
CREATE TABLE messages (
    id VARCHAR PRIMARY KEY,
    conversation_id VARCHAR NOT NULL,
    role VARCHAR NOT NULL,  -- 'user' ou 'assistant'
    content TEXT NOT NULL,
    audio_url VARCHAR,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSON,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
);
```

**Campos**:
- `id`: UUID da mensagem
- `conversation_id`: ID da conversação (FK)
- `role`: Papel da mensagem ('user' ou 'assistant')
- `content`: Conteúdo textual da mensagem
- `audio_url`: URL do arquivo de áudio (opcional)
- `created_at`: Timestamp da mensagem
- `metadata`: Dados adicionais em JSON

### sessions

Armazena sessões de conversação ativas.

```sql
CREATE TABLE sessions (
    id VARCHAR PRIMARY KEY,
    user_id VARCHAR NOT NULL,
    conversation_id VARCHAR,
    scenario_id VARCHAR,
    state JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
);
```

**Campos**:
- `id`: UUID da sessão
- `user_id`: ID do usuário (FK)
- `conversation_id`: ID da conversação associada (FK, opcional)
- `scenario_id`: ID do cenário (opcional)
- `state`: Estado da sessão em JSON
- `created_at`: Data de criação
- `updated_at`: Data de última atualização
- `expires_at`: Data de expiração (opcional)

### files

Armazena metadados de arquivos armazenados.

```sql
CREATE TABLE files (
    id VARCHAR PRIMARY KEY,
    user_id VARCHAR,
    filename VARCHAR NOT NULL,
    content_type VARCHAR,
    size INTEGER,
    storage_url VARCHAR,
    tags TEXT[],
    metadata JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);
```

**Campos**:
- `id`: UUID do arquivo
- `user_id`: ID do usuário proprietário (FK, opcional)
- `filename`: Nome do arquivo
- `content_type`: Tipo MIME do arquivo
- `size`: Tamanho em bytes
- `storage_url`: URL no storage (Backblaze B2, S3, etc.)
- `tags`: Array de tags
- `metadata`: Dados adicionais em JSON
- `created_at`: Data de upload

### student_profiles

Armazena perfis de estudantes (módulo de tutoring).

```sql
CREATE TABLE student_profiles (
    id VARCHAR PRIMARY KEY,
    user_id VARCHAR UNIQUE NOT NULL,
    cefr_level VARCHAR,  -- 'A1', 'A2', 'B1', 'B2', 'C1', 'C2'
    skills JSON,
    learning_path JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);
```

**Campos**:
- `id`: UUID do perfil
- `user_id`: ID do usuário (FK, único)
- `cefr_level`: Nível CEFR atual
- `skills`: Habilidades avaliadas em JSON
- `learning_path`: Caminho de aprendizado em JSON
- `created_at`: Data de criação
- `updated_at`: Data de última atualização

## 🔗 Relacionamentos

```
users (1) ──< (N) conversations
conversations (1) ──< (N) messages
users (1) ──< (N) sessions
users (1) ──< (N) files
users (1) ──< (1) student_profiles
```

## 📊 Índices

Índices recomendados para performance:

```sql
-- Índices para users
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_username ON users(username);

-- Índices para conversations
CREATE INDEX idx_conversations_user_id ON conversations(user_id);
CREATE INDEX idx_conversations_created_at ON conversations(created_at);

-- Índices para messages
CREATE INDEX idx_messages_conversation_id ON messages(conversation_id);
CREATE INDEX idx_messages_created_at ON messages(created_at);

-- Índices para sessions
CREATE INDEX idx_sessions_user_id ON sessions(user_id);
CREATE INDEX idx_sessions_expires_at ON sessions(expires_at);

-- Índices para files
CREATE INDEX idx_files_user_id ON files(user_id);
CREATE INDEX idx_files_tags ON files USING GIN(tags);
```

## 🔄 Migrações

O projeto usa SQLAlchemy para gerenciar migrações. Para criar uma nova migração:

```bash
# Usando Alembic (se configurado)
alembic revision --autogenerate -m "descrição da migração"
alembic upgrade head
```

## 🔒 Segurança

- **Senhas**: Sempre armazenadas como hash bcrypt
- **SQL Injection**: Prevenido pelo uso de SQLAlchemy ORM
- **Dados Sensíveis**: Criptografados em repouso (se configurado)

## 📈 Performance

### Otimizações

1. **Paginação**: Sempre use paginação para listagens
2. **Índices**: Mantenha índices atualizados
3. **Connection Pooling**: Configure pool de conexões
4. **Caching**: Use Redis para cache de queries frequentes

### Queries Comuns

```python
# Listar conversações de um usuário (paginado)
SELECT * FROM conversations 
WHERE user_id = ? 
ORDER BY updated_at DESC 
LIMIT ? OFFSET ?;

# Buscar mensagens de uma conversação
SELECT * FROM messages 
WHERE conversation_id = ? 
ORDER BY created_at ASC;

# Buscar sessão ativa
SELECT * FROM sessions 
WHERE user_id = ? 
AND expires_at > NOW();
```

## 🔧 Manutenção

### Backup

```bash
# SQLite
cp parle.db parle.db.backup

# PostgreSQL
pg_dump parle_db > backup.sql

# Restaurar
psql parle_db < backup.sql
```

### Limpeza

```sql
-- Limpar sessões expiradas
DELETE FROM sessions WHERE expires_at < NOW();

-- Limpar arquivos antigos (exemplo: > 1 ano)
DELETE FROM files 
WHERE created_at < NOW() - INTERVAL '1 year';
```

## 📚 Referências

- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [Database Design Best Practices](./DATABASE_BEST_PRACTICES.md)
