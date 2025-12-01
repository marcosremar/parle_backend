# Architecture Decision Records (ADRs)

Este diretório contém Architecture Decision Records (ADRs) para o projeto Parle Backend.

## O que são ADRs?

ADRs são documentos que capturam decisões arquiteturais importantes, incluindo:
- Contexto da decisão
- Decisão tomada
- Consequências (positivas e negativas)
- Alternativas consideradas

## Estrutura de um ADR

Cada ADR segue o formato:

```markdown
# ADR-000: Título da Decisão

## Status
[Proposta | Aceita | Rejeitada | Depreciada | Substituída]

## Contexto
Por que esta decisão precisa ser tomada?

## Decisão
O que foi decidido?

## Consequências
- Positivas: Benefícios da decisão
- Negativas: Custos e trade-offs

## Alternativas Consideradas
- Alternativa 1: Descrição e por que foi rejeitada
- Alternativa 2: Descrição e por que foi rejeitada
```

## ADRs Existentes

- [ADR-001: Arquitetura Monolítica Modular](./ADR-001-monolithic-modular-architecture.md)
- [ADR-002: Uso de FastAPI como Framework Web](./ADR-002-fastapi-framework.md)
- [ADR-003: Sistema de Módulos Internos](./ADR-003-internal-modules.md)

## Como Criar um Novo ADR

1. Crie um novo arquivo: `ADR-XXX-titulo-kebab-case.md`
2. Use o template acima
3. Adicione referência neste README
4. Commit com mensagem: `docs: add ADR-XXX for [decisão]`

## Referências

- [Documenting Architecture Decisions](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions)
- [ADR Template](https://github.com/joelparkerhenderson/architecture-decision-record)
