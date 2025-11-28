# Papers sobre Sistemas Tutores Inteligentes e Aprendizagem Adaptativa

Este diretório contém papers acadêmicos relevantes para o desenvolvimento do sistema de tutor inteligente com memória de erros e adaptação de nível linguístico.

## Papers Baixados

### 1. Adaptive Learning Path Navigation (2023)
- **Arquivo:** `adaptive_learning_path_navigation_2023.pdf`
- **Link:** https://arxiv.org/abs/2305.04475
- **Tópicos:** Knowledge Tracing, Reinforcement Learning, ALPN system
- **Relevância:** Sistema que combina Attentive Knowledge Tracing (AKT) com Reinforcement Learning para otimizar caminhos de aprendizagem

### 2. Empowering Personalized Learning (2024)
- **Arquivo:** `empowering_personalized_learning_2024.pdf`
- **Link:** https://arxiv.org/abs/2403.14071
- **Tópicos:** Student Modeling, Conversational Tutoring, LLM-based instruction
- **Relevância:** Sistema de tutoria personalizado com modelagem do aluno e estratégias instrucionais diversas

### 3. AI Conversational Tutors in Foreign Language Learning (2025)
- **Arquivo:** `ai_conversational_tutors_2025.pdf`
- **Link:** https://arxiv.org/abs/2508.05156
- **Tópicos:** Conversational AI, Language Learning, Evaluation Study
- **Relevância:** Avaliação de tutores conversacionais baseados em IA para ensino de línguas estrangeiras

### 4. TUTORING: Instruction-Grounded Conversational Agent (2023)
- **Arquivo:** `tutoring_instruction_grounded_2023.pdf`
- **Link:** https://arxiv.org/abs/2302.12623
- **Tópicos:** Conversational Agents, Language Learning, Multi-task Learning
- **Relevância:** Chatbot generativo treinado em conversas tutor-aluno com ações de ensino incorporadas

### 5. AI in Intelligent Tutoring Robots (2019)
- **Arquivo:** `ai_intelligent_tutoring_robots_2019.pdf`
- **Link:** https://arxiv.org/abs/1903.03414
- **Tópicos:** Intelligent Tutoring Robots, Design Guidelines, Systematic Review
- **Relevância:** Revisão sistemática sobre design de robôs tutores inteligentes

### 6. Intelligent Tutoring Systems: Comprehensive Survey (2018)
- **Arquivo:** `intelligent_tutoring_systems_survey_2018.pdf`
- **Link:** https://arxiv.org/abs/1808.07241
- **Tópicos:** ITS Survey, Historical Overview, Architecture
- **Relevância:** Survey abrangente sobre sistemas tutores inteligentes

### 7. Context-Aware Attentive Knowledge Tracing (2020)
- **Arquivo:** `attentive_knowledge_tracing_2020.pdf`
- **Link:** https://arxiv.org/abs/2007.12324
- **Tópicos:** AKT, Knowledge Tracing, Attention Mechanisms, Interpretability
- **Relevância:** Paper original do AKT (Attentive Knowledge Tracing) - modelo implementado no sistema atual
- **Arquivo:** `intelligent_tutoring_systems_survey_2018.pdf`
- **Link:** https://arxiv.org/abs/1812.09628
- **Tópicos:** ITS Survey, Historical Development, Architecture
- **Relevância:** Visão abrangente da evolução dos Sistemas Tutores Inteligentes

## Papers Referenciados (Links para Download)

### Papers Brasileiros

1. **Deep Learning em Sistemas Tutores Inteligentes (2023)**
   - Link: https://periodicos.se.df.gov.br/index.php/comcenso/article/view/1688
   - Tópicos: Deep Learning, STIs, EaD
   - Nota: Revisão sistemática sobre uso de deep learning em STIs

2. **Os Sistemas Tutores Inteligentes e a Adaptação do Ensino aos Perfis de Aprendizagem**
   - Link: https://periodicos.sbu.unicamp.br/ojs/index.php/etd/article/view/8663707
   - Tópicos: Adaptação, Perfis de Aprendizagem, Personalização

3. **Sistemas Tutores Inteligentes: um Mapeamento das Produções Brasileiras**
   - Link: https://sol.sbc.org.br/index.php/sbie/article/view/12887
   - Tópicos: Mapeamento Sistemático, Produções Brasileiras

4. **Indicadores da aprendizagem adaptativa em ambientes virtuais (2024)**
   - Link: https://lume.ufrgs.br/handle/10183/283783
   - Tópicos: Aprendizagem Adaptativa, Indicadores, Ambientes Virtuais

### Outros Papers Relevantes

5. **Artificial Intelligence Adaptive Learning Tools: The Teaching of English in Focus**
   - Link: https://doaj.org/article/55371de9f6294a4c8a2a778630b452f0
   - Tópicos: AI Tools, English Teaching, Adaptive Platforms

6. **Avaliação de STIs para Apoio a Tutores Humanos**
   - Link: https://periodicos.ufpe.br/revistas/index.php/gestaoorg/article/view/263224
   - Tópicos: STIs, Tutores Humanos, Cooperação

## Conceitos-Chave Encontrados

### Bayesian Knowledge Tracing (BKT)
- Algoritmo clássico para modelar nível de conhecimento do aluno
- Atualiza probabilidades de domínio baseado em interações
- Wikipedia: https://pt.wikipedia.org/wiki/Rastreamento_de_conhecimento_bayesiano

### Attentive Knowledge Tracing (AKT)
- Evolução do BKT usando attention mechanisms
- Melhor performance em predição de domínio

### Spaced Repetition (SRS)
- Técnica de revisão espaçada para reforçar aprendizado
- Usado em apps como Anki, Duolingo

### Scaffolding Pedagógico
- Suporte adaptativo baseado na Zona de Desenvolvimento Proximal (ZPD)
- Nível de ajuda varia conforme necessidade do aluno

### CEFR (Common European Framework of Reference)
- Padrão internacional de níveis linguísticos (A1-C2)
- Usado para nivelamento e adaptação de conteúdo

## Aplicações no Projeto

### 1. StudentModel Service
- Implementar BKT ou AKT para rastrear habilidades individuais
- Diagnostic Module para análise completa após cada turno
- Skill Mastery tracking com probabilidades

### 2. PromptManager Service
- Pedagogical Policy Engine baseado em Reinforcement Learning
- Estratégias adaptativas (reforço, introdução, desafio)
- Templates por nível CEFR (A1-C2)

### 3. Learning Path Navigator
- Sugestão de próximos tópicos baseado em ZPD
- Spaced Repetition para revisão
- Otimização de caminhos de aprendizagem

## Próximos Passos

1. Implementar Bayesian Knowledge Tracing (BKT) no StudentModel
2. Criar Diagnostic Module para análise completa de cada turno
3. Expandir PromptManager com estratégias pedagógicas
4. Adicionar Learning Path Navigator com SRS

