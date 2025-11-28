# Implementação do AKT (Attentive Knowledge Tracing)

## ✅ AKT é o Padrão

O sistema agora usa **AKT (Attentive Knowledge Tracing)** como algoritmo padrão para rastreamento de conhecimento. O **BKT (Bayesian Knowledge Tracing)** foi mantido como fallback.

## Características do AKT Implementado

### 1. Attention Mechanisms
- **Histórico de Interações**: Considera toda a sequência de interações, não apenas a última
- **Pesos Temporais**: Interações mais recentes recebem mais peso (decay exponencial)
- **Contextualização**: Adapta predições baseado em padrões históricos

### 2. Parâmetros Adaptativos
- **p_T (Aprendizado)**: Ajusta-se baseado em streaks de acertos/erros
  - Streak de acertos → aumenta p_T (aprende mais rápido)
  - Streak de erros → diminui p_T (aprende mais devagar)
  
- **p_F (Esquecimento)**: Ajusta-se baseado em tempo desde última prática
  - Muito tempo sem praticar → aumenta p_F
  - Prática frequente → diminui p_F

### 3. Features Contextuais
- **Dificuldade**: Considera dificuldade da questão
- **Complexidade**: Considera complexidade linguística
- **Tempo**: Considera tempo desde última interação
- **Severidade**: Considera severidade do erro

### 4. Predição Melhorada
- Combina predição base (BKT) com modulação por atenção
- Ajusta predição baseado em performance recente
- Considera padrões de acertos/erros no histórico

## Comparação: AKT vs BKT

| Característica | BKT | AKT |
|---------------|-----|-----|
| Considera histórico | ❌ Apenas última interação | ✅ Toda sequência |
| Parâmetros adaptativos | ❌ Fixos | ✅ Adaptam-se ao estudante |
| Contexto | ❌ Não considera | ✅ Considera dificuldade, tempo, etc. |
| Attention | ❌ Não tem | ✅ Mecanismos de atenção |
| Predição | Básica | Melhorada com atenção |

## Implementação Técnica

### Classe Principal
```python
from src.services.student_model.knowledge_tracer.akt_tracer import AttentiveKnowledgeTracer

tracer = AttentiveKnowledgeTracer(skill_params)
```

### Uso no Student Model Service
```python
# AKT é usado automaticamente (padrão)
akt_params = get_akt_params(skill_id)
tracer = AttentiveKnowledgeTracer(akt_params)

# BKT é usado apenas como fallback
try:
    tracer = AttentiveKnowledgeTracer(akt_params)
except Exception:
    tracer = BayesianKnowledgeTracer(bkt_params)  # Fallback
```

### Parâmetros AKT
```python
{
    'p_L0': 0.2,              # Probabilidade inicial
    'p_T': 0.15,              # Probabilidade de aprender
    'p_F': 0.05,              # Probabilidade de esquecer
    'p_G': 0.85,              # Probabilidade de acertar sabendo
    'p_S': 0.3,               # Probabilidade de errar sabendo
    'temporal_decay': 0.95,   # Decay para interações antigas
    'attention_window': 10,   # Número de interações para considerar
    'adaptation_rate': 0.1    # Taxa de adaptação de parâmetros
}
```

## Fluxo de Atualização

1. **Recebe interação** (correct, context)
2. **Extrai features contextuais** (dificuldade, complexidade, tempo)
3. **Adapta parâmetros** baseado em histórico (streaks, tempo)
4. **Calcula pesos de atenção** para interações históricas
5. **Combina atualização base (BKT) com modulação por atenção**
6. **Atualiza mastery probability**
7. **Adiciona à história** (mantém últimas 50 interações)

## Vantagens do AKT

1. **Mais Preciso**: Considera toda a sequência, não apenas última interação
2. **Adaptativo**: Parâmetros ajustam-se ao estudante individual
3. **Contextual**: Considera dificuldade, tempo, complexidade
4. **Melhor Predição**: Attention melhora predições de performance

## Fallback para BKT

O sistema automaticamente faz fallback para BKT se:
- AKT falha na inicialização
- Erro ao processar interação
- Parâmetros inválidos

Isso garante que o sistema sempre funcione, mesmo em casos extremos.

## Evolução Futura

A implementação atual é funcional e usa attention mechanisms. No futuro, quando houver dados suficientes (~10.000+ interações), pode-se:

1. Treinar modelo AKT mais sofisticado usando PyKT
2. Substituir implementação atual por modelo treinado
3. Migração transparente (mesma interface)

Mas a implementação atual já oferece vantagens significativas sobre BKT!

