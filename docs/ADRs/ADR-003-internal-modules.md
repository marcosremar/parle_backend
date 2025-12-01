# ADR-003: Sistema de Módulos Internos

## Status
Aceita

## Contexto

Com a arquitetura monolítica modular, precisávamos de um sistema para:
- Carregar módulos dinamicamente
- Gerenciar dependências entre módulos
- Permitir inicialização lazy
- Facilitar testes isolados
- Manter separação de responsabilidades

## Decisão

Implementar um sistema de módulos baseado em Factory Pattern:
- Cada módulo é uma classe Python independente
- Factory Pattern para criação e cache de instâncias
- Lazy initialization (módulos criados sob demanda)
- Interface comum para todos os módulos
- Suporte a inicialização assíncrona

## Consequências

### Positivas
- **Modularidade**: Código bem organizado e separado
- **Testabilidade**: Módulos podem ser testados isoladamente
- **Flexibilidade**: Fácil adicionar/remover módulos
- **Performance**: Lazy loading evita inicialização desnecessária
- **Manutenibilidade**: Código mais fácil de manter

### Negativas
- **Complexidade**: Sistema adicional para gerenciar
- **Acoplamento**: Módulos ainda compartilham processo
- **Debugging**: Pode ser mais difícil rastrear dependências

## Alternativas Consideradas

### Import Direto
- **Rejeitada**: Dificulta testes e substituição de implementações
- **Motivo**: Factory Pattern oferece mais flexibilidade

### Dependency Injection Container
- **Considerada**: Usar biblioteca como dependency-injector
- **Rejeitada**: Complexidade adicional desnecessária para o tamanho do projeto

### Plugin System
- **Considerada**: Sistema de plugins mais complexo
- **Rejeitada**: Overhead desnecessário, Factory Pattern é suficiente

## Implementação

```python
# src/modules/module_factory.py
class ModuleFactory:
    _modules = {}
    
    @classmethod
    def create(cls, module_name: str):
        if module_name not in cls._modules:
            # Lazy creation
            module = cls._create_module(module_name)
            cls._modules[module_name] = module
        return cls._modules[module_name]
```

## Referências

- [Factory Pattern](https://refactoring.guru/design-patterns/factory-method)
- [Dependency Injection in Python](https://python-dependency-injector.ets-labs.org/)
