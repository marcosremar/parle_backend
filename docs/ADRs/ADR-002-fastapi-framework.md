# ADR-002: Uso de FastAPI como Framework Web

## Status
Aceita

## Contexto

Precisávamos escolher um framework web Python moderno que suportasse:
- APIs assíncronas (async/await)
- Documentação automática (OpenAPI/Swagger)
- Validação de dados
- Type hints e type checking
- Performance alta

## Decisão

Adotar FastAPI como framework web principal porque:
- Suporte nativo a async/await
- Geração automática de documentação OpenAPI
- Validação baseada em Pydantic
- Type hints integrados
- Performance comparável a Node.js e Go
- Ecossistema maduro e comunidade ativa

## Consequências

### Positivas
- **Documentação**: OpenAPI gerado automaticamente
- **Validação**: Validação automática de requests/responses
- **Type Safety**: Type hints melhoram qualidade do código
- **Performance**: Uma das frameworks Python mais rápidas
- **Developer Experience**: Excelente DX com autocomplete e validação

### Negativas
- **Dependência**: Nova dependência no projeto
- **Curva de Aprendizado**: Equipe precisa aprender FastAPI
- **Ecossistema**: Menor que Django/Flask (mas crescente)

## Alternativas Consideradas

### Django REST Framework
- **Rejeitada**: Muito pesado e síncrono por padrão
- **Motivo**: Overhead desnecessário para uma API

### Flask
- **Rejeitada**: Não tem suporte nativo a async
- **Motivo**: Performance inferior para I/O intensivo

### Tornado
- **Rejeitada**: API menos intuitiva e documentação automática limitada
- **Motivo**: FastAPI oferece melhor DX

## Referências

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [FastAPI Performance](https://www.techempower.com/benchmarks/)
