# Testes E2E - Scenario Role Playing

## 📋 Visão Geral

Este arquivo contém testes E2E que verificam se a IA consegue manter o papel/role atribuído através de scenarios em múltiplos turnos de conversação.

## 🎭 Objetivo dos Testes

Verificar que:
1. A IA mantém o papel definido no scenario através de múltiplos turnos
2. A IA não "quebra personagem" durante a conversa
3. A IA responde de forma consistente com o papel atribuído
4. Diferentes scenarios produzem comportamentos diferentes
5. Mudanças de scenario resultam em mudanças de comportamento

## 🧪 Testes Disponíveis

### 1. `test_restaurant_waiter_role_consistency`
**Papel**: Garçom de restaurante  
**Cenário**: Cliente fazendo pedido em restaurante  
**Turnos**: 4 turnos testando comportamento de garçom  
**Validação**: Respostas devem ser profissionais, usar linguagem de restaurante, não quebrar personagem

---

### 2. `test_teacher_role_consistency`
**Papel**: Professor de português  
**Cenário**: Aula de português com aluno  
**Turnos**: 3 turnos testando comportamento de professor  
**Validação**: Respostas devem ser educacionais, explicativas, encorajadoras

---

### 3. `test_doctor_role_consistency`
**Papel**: Médico  
**Cenário**: Consulta médica  
**Turnos**: 3 turnos testando comportamento de médico  
**Validação**: Respostas devem ser profissionais, empáticas, usar terminologia médica

---

### 4. `test_travel_agent_role_consistency`
**Papel**: Agente de viagens  
**Cenário**: Planejamento de viagem  
**Turnos**: 3 turnos testando comportamento de agente  
**Validação**: Respostas devem ser entusiasmadas sobre viagens, recomendar destinos

---

### 5. `test_role_consistency_10_turns`
**Papel**: Chef e instrutor de culinária  
**Cenário**: Aula de culinária  
**Turnos**: 10+ turnos testando consistência de papel  
**Validação**: Papel mantido através de toda a conversa longa

---

### 6. `test_role_switching_between_scenarios`
**Papéis**: Personal Trainer → Nutritionist  
**Cenário**: Mudança de scenario durante conversa  
**Turnos**: 2 turnos em scenarios diferentes  
**Validação**: Diferentes scenarios produzem comportamentos diferentes

---

### 7. `test_strict_role_enforcement`
**Papel**: Cavaleiro medieval (papel muito específico)  
**Cenário**: Role-playing medieval  
**Turnos**: 5 turnos tentando quebrar o personagem  
**Validação**: IA não quebra personagem mesmo quando questionada diretamente

---

### 8. `test_role_consistency_with_topic_changes`
**Papel**: Jornalista  
**Cenário**: Entrevista com mudanças de tópico  
**Turnos**: 5 turnos com tópicos diferentes  
**Validação**: Papel mantido mesmo com mudanças de tópico

---

### 9. `test_complex_role_with_multiple_attributes`
**Papel**: Consultor de negócios (amigável mas profissional)  
**Cenário**: Consultoria empresarial  
**Turnos**: 5 turnos testando múltiplos atributos  
**Validação**: IA mantém equilíbrio entre amigabilidade e profissionalismo

---

### 10. `test_role_consistency_across_long_conversation`
**Papel**: Terapeuta  
**Cenário**: Sessão de terapia  
**Turnos**: 15+ turnos testando consistência  
**Validação**: Papel mantido através de conversa muito longa

---

### 11. `test_role_consistency_with_interruptions`
**Papel**: Atendente de suporte ao cliente  
**Cenário**: Atendimento com interrupções  
**Turnos**: 7 turnos com interrupções e mudanças de tópico  
**Validação**: Papel mantido mesmo com interrupções

---

## 🚀 Como Executar

### Executar todos os testes de role-playing
```bash
pytest tests/e2e/test_scenario_role_playing.py -v
```

### Executar teste específico
```bash
pytest tests/e2e/test_scenario_role_playing.py::TestScenarioRolePlaying::test_restaurant_waiter_role_consistency -v
```

### Executar por tipo de role
```bash
# Testes de roles profissionais
pytest tests/e2e/test_scenario_role_playing.py -k "waiter or doctor or teacher" -v

# Testes de consistência longa
pytest tests/e2e/test_scenario_role_playing.py -k "long or 10_turns" -v

# Testes de enforcement
pytest tests/e2e/test_scenario_role_playing.py -k "strict or enforcement" -v
```

## 📊 Estatísticas

- **Total de testes**: 11
- **Turnos testados**: 70+ turnos de conversa
- **Papéis testados**: 10+ papéis diferentes
- **Cenários cobertos**: Restaurante, educação, saúde, viagens, negócios, terapia, suporte
- **Tempo estimado**: 10-15 minutos para todos os testes (com API keys configuradas)

## ✅ Validações

Cada teste valida:
- ✅ Respostas são geradas para cada turno
- ✅ Respostas mantêm o papel atribuído
- ✅ IA não quebra personagem
- ✅ Linguagem e tom são apropriados para o papel
- ✅ Consistência através de múltiplos turnos
- ✅ Diferentes scenarios produzem comportamentos diferentes

## 🔍 Indicadores de Papel

Cada teste verifica indicadores específicos do papel:

- **Garçom**: "mesa", "reserva", "pedido", "prato", "recomendo"
- **Professor**: "exemplo", "diferença", "correto", "prática"
- **Médico**: "sintoma", "dor", "consulta", "tratamento"
- **Agente de viagens**: "destino", "viagem", "recomendo", "lugar"
- **Chef**: "ingrediente", "receita", "cozinhar", "temperatura"
- **Terapeuta**: "entendo", "sentimento", "apoio", "técnica"

## ⚠️ Notas Importantes

- **API Keys**: Testes requerem API keys configuradas (LLM_API_KEY, etc.)
- **Testes podem ser pulados**: Se API keys não estiverem configuradas, testes serão pulados automaticamente
- **Dependências externas**: Testes fazem chamadas reais para APIs de LLM
- **Tempo de execução**: Testes podem ser lentos devido a chamadas de API
- **Consistência**: Resultados podem variar dependendo da qualidade do LLM usado

## 🎯 Casos de Uso

Estes testes são úteis para:
- Validar que scenarios funcionam corretamente
- Garantir que a IA mantém personagens consistentes
- Testar diferentes tipos de interações
- Verificar robustez do sistema de scenarios
- Validar que mudanças de scenario funcionam

## 📝 Estrutura dos Scenarios

Cada scenario deve ter:
- `name`: Nome do scenario
- `description`: Descrição do scenario
- `system_prompt`: Prompt do sistema definindo o papel da IA

Exemplo:
```python
scenario = await scenarios.create_scenario(
    name="Restaurant Waiter",
    description="Fine dining restaurant waiter scenario",
    system_prompt="You are a professional restaurant waiter..."
)
```
