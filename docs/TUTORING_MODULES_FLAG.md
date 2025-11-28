# Tutoring Modules - Feature Flag

## Visão Geral

Os módulos de tutoring estão desativados por padrão e podem ser habilitados através de uma variável de ambiente.

## Módulos Afetados

Os seguintes módulos são controlados pela flag:

- `student_model` - Modelo de estudante
- `diagnostic_module` - Análise diagnóstica
- `pedagogical_policy` - Política pedagógica
- `learning_path` - Caminho de aprendizado

## Como Habilitar

Para habilitar os módulos de tutoring, defina a variável de ambiente:

```bash
export ENABLE_TUTORING_MODULES=true
```

Ou no arquivo `.env`:

```env
ENABLE_TUTORING_MODULES=true
```

## Comportamento

### Quando Desativado (padrão):

- Módulos retornam um `DisabledTutoringWrapper`
- Chamadas a métodos retornam erro `RuntimeError` informando que o módulo está desativado
- Endpoints da API retornam `503 Service Unavailable` com mensagem explicativa
- Logs mostram aviso quando módulo é criado

### Quando Ativado:

- Módulos funcionam normalmente
- Todos os métodos estão disponíveis
- Endpoints da API funcionam normalmente

## Exemplo de Uso

### Python

```python
from src.modules import module_factory

# Tentar criar módulo (desativado por padrão)
student = module_factory.create("student_model")

# Verificar se está desativado
if hasattr(student, 'disabled') and student.disabled:
    print("Tutoring modules are disabled")
else:
    profile = await student.get_profile("user123")
```

### API

```bash
# Tentar usar endpoint (retorna 503 se desativado)
curl http://localhost:8000/api/tutoring/student/user123/profile

# Resposta quando desativado:
# {
#   "detail": "Tutoring modules are disabled. Set ENABLE_TUTORING_MODULES=true to enable."
# }
```

## Endpoints Afetados

- `POST /api/tutoring/diagnostic/analyze`
- `POST /api/tutoring/policy/compose`
- `GET /api/tutoring/path/{user_id}/next`
- `GET /api/tutoring/student/{user_id}/profile`

Todos retornam `503 Service Unavailable` quando tutoring está desativado.

## Motivo da Desativação

Os módulos de tutoring são considerados "future features" e estão sendo desativados temporariamente para:

- Reduzir complexidade inicial
- Focar em funcionalidades core primeiro
- Facilitar testes e desenvolvimento
- Reduzir dependências opcionais

## Ativação Futura

Quando os módulos de tutoring estiverem prontos para produção:

1. Definir `ENABLE_TUTORING_MODULES=true` no ambiente
2. Testar todos os módulos
3. Validar integração completa
4. Considerar remover a flag e ativar por padrão

---

**Última atualização:** 28/11/2025
