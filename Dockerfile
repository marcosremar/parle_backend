FROM python:3.12-slim

WORKDIR /app

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements
COPY user/requirements.txt /app/requirements.txt

# Instalar dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código do serviço e core
COPY user/ /app/services/user/
COPY core/ /app/src/core/

# Criar estrutura de diretórios necessária
RUN mkdir -p /app/src && \
    ln -s /app/core /app/src/core && \
    ln -s /app/services/user /app/src/services/user

# Variáveis de ambiente
ENV PYTHONPATH=/app:/app/src
ENV PORT=8200

# Expor porta
EXPOSE 8200

# Comando para iniciar o serviço
WORKDIR /app/services/user
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8200"]

