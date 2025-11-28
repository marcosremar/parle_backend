# Bayesian Knowledge Tracing (BKT): Explicação Detalhada

## O Que é BKT?

Bayesian Knowledge Tracing é um **modelo probabilístico** usado em Sistemas Tutores Inteligentes para estimar o nível de conhecimento de um estudante sobre uma habilidade específica. Em vez de pensar em "o aluno sabe ou não sabe", o BKT representa o conhecimento como **probabilidades que evoluem ao longo do tempo**.

## Conceitos Fundamentais

### Estados Ocultos
O BKT assume que cada habilidade tem **dois estados ocultos**:
- **Não Aprendido (L₀)**: O aluno ainda não domina a habilidade
- **Aprendido (L₁)**: O aluno já domina a habilidade

### Quatro Parâmetros do BKT

1. **Probabilidade Inicial (p₀)**: Chance do aluno já começar sabendo a habilidade
   - Exemplo: p₀ = 0.3 significa que 30% dos alunos já sabem verbo no passado ao começar

2. **Probabilidade de Aprendizado (p_T)**: Chance de aprender quando exposto ao conceito
   - Exemplo: p_T = 0.2 significa que há 20% de chance de aprender cada vez que a habilidade é praticada

3. **Probabilidade de Esquecer (p_F)**: Chance de esquecer o que já sabia
   - Exemplo: p_F = 0.1 significa que há 10% de chance de esquecer entre sessões

4. **Probabilidade de Acertar (p_G)**: Chance de acertar uma questão quando sabe a resposta
   - Exemplo: p_G = 0.9 significa que alunos que sabem têm 90% de chance de acertar

5. **Probabilidade de Errar (p_S)**: Chance de errar mesmo sabendo (slip)
   - Exemplo: p_S = 0.1 significa que alunos que sabem ainda erram 10% das vezes

## Como Funciona o Algoritmo

### 1. Estado Inicial
Começamos com a probabilidade inicial:
```
P(L₁) = p₀
P(L₀) = 1 - p₀
```

### 2. Observação de uma Resposta
Quando o aluno responde uma questão:

**Se ACERTOU:**
- Probabilidade de ter acertado sabendo: P(correto|L₁) = p_G
- Probabilidade de ter acertado não sabendo: P(correto|L₀) = p_S (chute)

**Se ERROU:**
- Probabilidade de ter errado sabendo: P(errado|L₁) = p_S
- Probabilidade de ter errado não sabendo: P(errado|L₀) = 1 - p_S

### 3. Atualização Bayesiana
Usamos o **Teorema de Bayes** para atualizar as crenças:

```
P(L₁|observação) = [P(observação|L₁) × P(L₁)] / P(observação)
```

### 4. Transição de Estado
Após cada tentativa, consideramos aprendizado e esquecimento:

```
P(L₁|t+1) = P(L₁|t) × (1 - p_F) + P(L₀|t) × p_T
```

## Exemplo Prático: Verbos no Passado

### Cenário
Um aluno está aprendendo conjugação de verbos no passado em português.

**Parâmetros iniciais:**
- p₀ = 0.2 (20% de chance de já saber)
- p_T = 0.15 (15% de chance de aprender por tentativa)
- p_F = 0.05 (5% de chance de esquecer)
- p_G = 0.85 (85% de chance de acertar sabendo)
- p_S = 0.3 (30% de chance de errar mesmo sabendo - slip)

### Sequência de Interações

**Tentativa 1: "Eu *fui* na praia ontem" (CERTO)**
```
Antes: P(L₁) = 0.2
Observação correta: P(correto|L₁) = 0.85, P(correto|L₀) = 0.3
Após Bayes: P(L₁) = 0.41 (41% de chance de saber)
Após transição: P(L₁) = 0.41 × 0.95 + 0.59 × 0.15 = 0.48
```

**Tentativa 2: "Ontem eu *comer* pizza" (ERRADO - deveria ser "comi")**
```
Antes: P(L₁) = 0.48
Observação errada: P(errado|L₁) = 0.3, P(errado|L₀) = 0.7
Após Bayes: P(L₁) = 0.48 × 0.3 / (0.48 × 0.3 + 0.52 × 0.7) = 0.21
Após transição: P(L₁) = 0.21 × 0.95 + 0.79 × 0.15 = 0.30
```

**Tentativa 3: "Eu *estudei* matemática" (CERTO)**
```
Antes: P(L₁) = 0.30
Após: P(L₁) aumenta para ~0.55
```

## Interpretação das Probabilidades

- **0.0 - 0.3**: Aluno provavelmente não sabe
- **0.3 - 0.7**: Aluno está aprendendo/incerto
- **0.7 - 1.0**: Aluno provavelmente sabe

## Vantagens do BKT

1. **Incerteza Explícita**: Representa que o conhecimento não é binário
2. **Adaptação**: Sistema pode ajustar dificuldade baseado na probabilidade
3. **Previsão**: Pode prever performance futura
4. **Diagnóstico**: Identifica quando intervenção é necessária

## Limitações

1. **Assunções Simplistas**: Não considera dificuldade da questão
2. **Parâmetros Fixos**: Mesmo parâmetros para todos os alunos
3. **Habilidades Independentes**: Não modela dependências entre habilidades
4. **Sem Contexto**: Não considera ordem ou contexto das questões

## Evoluções: BKT → AKT (Attentive Knowledge Tracing)

O AKT (do paper de Chen et al., 2023) resolve algumas limitações usando **attention mechanisms**:

- **Contextualização**: Considera sequência de interações
- **Parâmetros Adaptativos**: Diferentes parâmetros por aluno
- **Relações entre Habilidades**: Modela como habilidades se relacionam

## Implementação no Nosso Sistema

### Arquitetura Sugerida

```python
class BayesianKnowledgeTracer:
    def __init__(self, skill_id, params):
        self.skill_id = skill_id
        self.p_L = params['p_L0']  # Probabilidade inicial de saber
        self.p_T = params['p_T']   # Probabilidade de aprender
        self.p_F = params['p_F']   # Probabilidade de esquecer
        self.p_G = params['p_G']   # Probabilidade de acertar sabendo
        self.p_S = params['p_S']   # Probabilidade de errar sabendo

    def update_belief(self, correct: bool):
        """Atualiza crença baseado em observação"""
        if correct:
            p_correct_given_L1 = self.p_G
            p_correct_given_L0 = self.p_S
        else:
            p_correct_given_L1 = 1 - self.p_G
            p_correct_given_L0 = 1 - self.p_S

        # Teorema de Bayes
        p_L1_given_obs = (p_correct_given_L1 * self.p_L) / \
                        (p_correct_given_L1 * self.p_L + p_correct_given_L0 * (1 - self.p_L))

        # Transição de estado (aprendizado + esquecimento)
        self.p_L = p_L1_given_obs * (1 - self.p_F) + (1 - p_L1_given_obs) * self.p_T

        return self.p_L
```

### Integração com StudentModel

```python
@dataclass
class SkillMastery:
    skill_id: str
    mastery_probability: float  # P(L₁) do BKT
    attempts: int
    successes: int
    last_updated: datetime

class StudentModel:
    def __init__(self, user_id):
        self.user_id = user_id
        self.skill_trackers = {}  # skill_id -> BayesianKnowledgeTracer

    def assess_response(self, skill_id: str, correct: bool):
        """Avalia resposta e atualiza conhecimento"""
        if skill_id not in self.skill_trackers:
            # Inicializar com parâmetros padrão
            params = self.get_default_params(skill_id)
            self.skill_trackers[skill_id] = BayesianKnowledgeTracer(skill_id, params)

        tracker = self.skill_trackers[skill_id]
        new_probability = tracker.update_belief(correct)

        # Atualizar banco de dados
        self.save_skill_mastery(skill_id, new_probability)

        return new_probability
```

### Aplicações Pedagógicas

1. **Adaptação de Dificuldade**: Se P(L₁) > 0.7, apresentar questões mais difíceis
2. **Revisão Espaçada**: Reapresentar habilidades com P(L₁) < 0.5
3. **Feedback Personalizado**: "Você está progredindo bem nessa habilidade!" se P(L₁) aumentou
4. **Intervenção**: Focar em habilidades com baixa probabilidade de domínio

## Parâmetros Sugeridos para Língua Portuguesa

### Habilidades Básicas (A1-A2)
- Verbos no Presente: p₀=0.4, p_T=0.2, p_F=0.05, p_G=0.8, p_S=0.2
- Vocabulário Básico: p₀=0.6, p_T=0.25, p_F=0.03, p_G=0.85, p_S=0.15

### Habilidades Intermediárias (B1-B2)
- Verbos no Passado: p₀=0.2, p_T=0.15, p_F=0.08, p_G=0.75, p_S=0.25
- Estruturas Complexas: p₀=0.1, p_T=0.1, p_F=0.1, p_G=0.7, p_S=0.3

## Conclusão

O BKT é uma base sólida para sistemas tutores inteligentes porque:

1. **Representa Incerteza**: Conhecimento não é tudo-ou-nada
2. **Adapta Dinamicamente**: Atualiza crenças com cada interação
3. **É Computacionalmente Eficiente**: Pode rodar em tempo real
4. **Fornece Insights**: Mostra progresso e identifica necessidades de intervenção

Para nosso sistema, começar com BKT clássico e depois evoluir para AKT (com deep learning) seria ideal. Isso nos daria uma base robusta de modelagem de conhecimento que suportaria todas as outras funcionalidades avançadas.
