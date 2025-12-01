# ADR-001: Arquitetura Monolítica Modular

## Status
Aceita

## Contexto

O Parle Backend inicialmente foi projetado como uma arquitetura de microserviços, com cada serviço rodando como processo HTTP independente. Isso trouxe complexidade:
- Múltiplos processos para gerenciar
- Overhead de comunicação HTTP entre serviços
- Complexidade de deploy e orquestração
- Dificuldade de debugging distribuído

## Decisão

Adotar uma arquitetura monolítica modular onde:
- Todos os serviços rodam em um único processo Python
- Comunicação entre módulos via chamadas diretas Python (sem HTTP)
- Módulos são carregados dinamicamente via factory pattern
- Mantém separação de responsabilidades através de módulos bem definidos

## Consequências

### Positivas
- **Performance**: Chamadas diretas são muito mais rápidas que HTTP
- **Simplicidade**: Um único processo para gerenciar
- **Debugging**: Mais fácil debugar em um único processo
- **Deploy**: Deploy simplificado (um único container/processo)
- **Desenvolvimento**: Mais rápido desenvolver e testar localmente

### Negativas
- **Escalabilidade**: Dificulta escalar componentes individuais
- **Isolamento**: Falhas podem afetar todo o sistema
- **Tecnologia**: Todos os módulos devem usar Python
- **Acoplamento**: Módulos compartilham o mesmo processo e memória

## Alternativas Consideradas

### Microserviços Puros
- **Rejeitada**: Complexidade desnecessária para o tamanho atual do projeto
- **Motivo**: Overhead de comunicação e orquestração não justificados

### Monolito Tradicional
- **Rejeitada**: Falta de separação de responsabilidades
- **Motivo**: Dificulta manutenção e testes isolados

### Híbrida (Alguns serviços separados)
- **Considerada**: Manter WebSocket como serviço separado
- **Decisão**: WebSocket permanece separado por necessidades específicas de tempo real

## Referências

- [Monolithic vs Microservices Architecture](https://martinfowler.com/articles/microservices.html)
- [Modular Monolith](https://www.kamilgrzybek.com/blog/modular-monolith-primer)
