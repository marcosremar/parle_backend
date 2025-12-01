# Template de Docstrings - Parle Backend

Este documento define o padrão de docstrings usado no projeto.

## 📋 Formato: Google Style

O projeto usa formato Google-style para docstrings.

## 📝 Template para Funções

```python
def function_name(param1: str, param2: int, optional_param: Optional[str] = None) -> dict:
    """
    Descrição curta e clara da função em uma linha.
    
    Descrição mais detalhada se necessário. Pode incluir múltiplas
    linhas explicando o comportamento, contexto, ou uso da função.
    
    Args:
        param1: Descrição do parâmetro 1. Incluir tipo esperado e
            comportamento se não for óbvio.
        param2: Descrição do parâmetro 2. Explicar validações ou
            restrições se houver.
        optional_param: Descrição do parâmetro opcional. Explicar
            valor padrão e quando usar.
    
    Returns:
        Descrição do retorno. Incluir estrutura se for dict/list.
        Exemplo:
            {
                "status": "success",
                "data": {...}
            }
    
    Raises:
        ValueError: Quando o parâmetro é inválido.
        HTTPException: Quando a requisição falha (status_code, detail).
        Exception: Para outros erros inesperados.
    
    Example:
        >>> result = function_name("test", 42)
        >>> print(result["status"])
        'success'
    
    Note:
        Notas adicionais sobre comportamento, limitações, ou
        considerações importantes.
    """
    pass
```

## 📝 Template para Classes

```python
class ClassName:
    """
    Descrição curta da classe.
    
    Descrição mais detalhada explicando propósito, uso, e
    características principais da classe.
    
    Attributes:
        attribute1: Descrição do atributo 1.
        attribute2: Descrição do atributo 2.
    
    Example:
        >>> instance = ClassName(param1="value")
        >>> instance.method()
        result
    """
    
    def __init__(self, param1: str):
        """
        Inicializa a classe.
        
        Args:
            param1: Descrição do parâmetro de inicialização.
        """
        self.attribute1 = param1
```

## 📝 Template para Métodos Assíncronos

```python
async def async_function(param: str) -> dict:
    """
    Descrição da função assíncrona.
    
    Args:
        param: Descrição do parâmetro.
        
    Returns:
        Descrição do retorno.
        
    Raises:
        Exception: Descrição do erro.
        
    Note:
        Esta é uma função assíncrona e deve ser chamada com await.
    """
    pass
```

## 📝 Template para Endpoints FastAPI

```python
@router.post("/endpoint/path")
async def endpoint_function(request: RequestModel):
    """
    Descrição do endpoint.
    
    Descrição detalhada do que o endpoint faz, quando usar,
    e comportamento esperado.
    
    Args:
        request: Modelo de requisição com campos documentados.
        
    Returns:
        Modelo de resposta ou dict com estrutura documentada.
        
    Raises:
        HTTPException: 
            - 400: Quando validação falha.
            - 401: Quando não autenticado.
            - 500: Quando erro interno ocorre.
    
    Example:
        ```json
        {
            "field1": "value1",
            "field2": 42
        }
        ```
    """
    pass
```

## ✅ Checklist de Docstring

Para cada função/classe pública:

- [ ] Descrição curta na primeira linha
- [ ] Descrição detalhada se necessário
- [ ] Todos os parâmetros documentados (Args)
- [ ] Tipo de retorno documentado (Returns)
- [ ] Exceções documentadas (Raises)
- [ ] Exemplo de uso (Example) se útil
- [ ] Notas adicionais (Note) se necessário

## 🎯 Prioridades

### Alta Prioridade (Fazer Primeiro)

1. Endpoints da API (`src/api/routers/`)
2. Funções públicas em `src/core/`
3. Métodos principais de módulos (`src/modules/`)

### Média Prioridade

1. Classes principais
2. Funções utilitárias públicas
3. Helpers e utilities

### Baixa Prioridade

1. Funções privadas (começam com `_`)
2. Métodos internos
3. Código legado

## 📚 Referências

- [Google Python Style Guide - Docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- [PEP 257 - Docstring Conventions](https://peps.python.org/pep-0257/)
