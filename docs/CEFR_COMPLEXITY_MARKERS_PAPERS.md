# Papers sobre Marcadores de Complexidade CEFR

Este documento lista papers acadêmicos que definem marcadores de complexidade linguística para os níveis CEFR, úteis para melhorar a avaliação automática de complexidade textual.

## Papers Principais

### 1. NILC-Metrix: Avaliação da complexidade da linguagem escrita e falada em português brasileiro
- **Link**: https://arxiv.org/abs/2201.03445
- **Descrição**: Sistema computacional com 200 métricas para avaliar complexidade textual em português brasileiro
- **Métricas**: Coesão, coerência, léxico, sintaxe
- **Relevância**: Específico para português, cobre múltiplos níveis linguísticos
- **Uso**: Base para métricas de complexidade em português

### 2. Automated Classification of Written Proficiency Levels on the CEFR-Scale through Complexity Contours and RNNs
- **Link**: https://aclanthology.org/2021.bea-1.21/
- **Descrição**: Usa "contornos de complexidade" e RNNs para classificar níveis CEFR (A1-C2)
- **Métricas**: Complexidade lexical e sintática
- **Relevância**: Classificação automática de níveis CEFR baseada em complexidade
- **Uso**: Modelo de classificação de níveis baseado em métricas sequenciais

### 3. Predicting CEFRL levels in learner English on the basis of metrics and full texts
- **Link**: https://arxiv.org/abs/1806.11099
- **Descrição**: Analisa métricas linguísticas para classificar aprendizes nos níveis CEFR
- **Métricas**: Lexicais e sintáticas (tokens, tipos de palavras)
- **Relevância**: Foco em métricas que distinguem níveis A1-B1
- **Uso**: Identificação de métricas mais importantes por nível

### 4. Automatic Assessment of Text Complexity Levels in European Portuguese
- **Link**: https://researchportal.ulisboa.pt/en/publications/avalia%C3%A7ao-autom%C3%A1tica-do-n%C3%ADvel-de-complexidade-de-textos-em-portug
- **Descrição**: Avaliação automática de complexidade textual em português europeu usando CEFR
- **Relevância**: Específico para português europeu, usa CEFR como referência
- **Uso**: Classificação de textos por nível de proficiência necessário

### 5. Níveis e descritores de complexidade textual para adultos de baixa literacia: um referencial do projeto iRead4Skills
- **Link**: https://novaresearch.unl.pt/en/publications/n%C3%ADveis-e-descritores-de-complexidade-textual-para-adultos-de-baix/
- **Descrição**: Define níveis de complexidade do português europeu para baixa literacia
- **Aspectos**: Léxico, estruturas sintáticas, coesão textual
- **Relevância**: Traços linguísticos que conferem complexidade aos textos
- **Uso**: Referencial para níveis básicos (A1-A2)

### 6. Automatic Classification of Sentence Difficulty in Arabic
- **Link**: https://arxiv.org/abs/2103.04386
- **Descrição**: Classificador de dificuldade de sentenças usando níveis CEFR
- **Características**: POS tags, árvores de dependência, pontuações de legibilidade, listas de frequência
- **Relevância**: Compara embeddings vs características linguísticas tradicionais
- **Uso**: Metodologia para classificação de dificuldade por nível

### 7. CLaC at SemEval-2016 Task 11: Exploring linguistic and psycholinguistic features for complex word identification
- **Link**: https://arxiv.org/abs/1709.02843
- **Descrição**: Identificação de palavras complexas usando características linguísticas e psicolinguísticas
- **Relevância**: Métricas de complexidade lexical
- **Uso**: Identificação de vocabulário complexo vs simples

## Marcadores de Complexidade Identificados

### Complexidade Sintática
- **Comprimento de sentença**: Média de palavras por sentença
- **Subordinação**: Presença de orações subordinadas
- **Coordenação**: Uso de conectores coordenativos
- **Profundidade sintática**: Profundidade de árvores de dependência
- **Estruturas passivas**: Uso de voz passiva
- **Estruturas complexas**: Subjuntivo, condicionais, relativas

### Complexidade Lexical
- **Frequência de palavras**: Palavras de alta vs baixa frequência
- **Diversidade lexical**: Tipo/token ratio
- **Comprimento de palavras**: Média de caracteres por palavra
- **Vocabulário técnico**: Presença de termos especializados
- **Expressões idiomáticas**: Uso de expressões fixas

### Complexidade Discursiva
- **Coesão**: Uso de conectores e referências
- **Coerência**: Estruturação lógica do texto
- **Marcadores discursivos**: Presença de marcadores de discurso

## Aplicação no Sistema

### Métricas a Implementar

1. **Sintáticas**:
   - Média de palavras por sentença (já implementado)
   - Presença de subordinação
   - Profundidade sintática
   - Uso de voz passiva (já implementado)
   - Uso de subjuntivo (já implementado)
   - Orações relativas (já implementado)

2. **Lexicais**:
   - Tipo/token ratio
   - Frequência média de palavras
   - Comprimento médio de palavras
   - Densidade de vocabulário técnico

3. **Discursivas**:
   - Densidade de conectores
   - Marcadores discursivos
   - Referências anafóricas

### Próximos Passos

1. Baixar e analisar os papers principais
2. Extrair métricas específicas por nível CEFR
3. Implementar métricas adicionais no sistema
4. Validar métricas com corpus anotado por nível CEFR
5. Integrar métricas na análise de complexidade do Sonnet 4.5

## Referências (Formato APA)

1. Leal, S. E., Duran, M. S., Scarton, C. E., Hartmann, N. S., & Aluísio, S. M. (2022). NILC-Metrix: Assessing the complexity of written and spoken language in Brazilian Portuguese. *arXiv preprint arXiv:2201.03445*. https://arxiv.org/abs/2201.03445

2. Vajjala, S., & Rama, T. (2021). Automated classification of written proficiency levels on the CEFR-scale through complexity contours and RNNs. In *Proceedings of the 16th Workshop on Innovative Use of NLP for Building Educational Applications* (pp. 180-190). Association for Computational Linguistics. https://aclanthology.org/2021.bea-1.21/

3. Arnold, T., Ballier, N., Gaillat, T., & Lissòn, P. (2018). Predicting CEFRL levels in learner English on the basis of metrics and full texts. *arXiv preprint arXiv:1806.11099*. https://arxiv.org/abs/1806.11099

4. Ribeiro, E., Mamede, N., & Baptista, J. (2024). Avaliação automática do nível de complexidade de textos em português europeu [Automatic assessment of text complexity levels in European Portuguese]. *Linguamática*, 16(2), 115-139. https://doi.org/10.21814/lm.16.2.449

5. Monteiro, R., Correia, S., Amaro, R., Moutinho, M., Barbosa, S., & Reis, M. L. (2022). Níveis e descritores de complexidade textual para adultos de baixa literacia: um referencial do projeto iRead4Skills [Levels and descriptors of textual complexity for low-literacy adults: a reference framework for the iRead4Skills project]. *Universidade NOVA de Lisboa*. https://novaresearch.unl.pt/en/publications/n%C3%ADveis-e-descritores-de-complexidade-textual-para-adultos-de-baix/

6. Al-Khawaja, N., Habash, N., & Bouamor, H. (2021). Automatic classification of sentence difficulty in Arabic. *arXiv preprint arXiv:2103.04386*. https://arxiv.org/abs/2103.04386

7. Shardlow, M., Cooper, M., Zampieri, M., & Nawaz, R. (2017). CLaC at SemEval-2016 Task 11: Exploring linguistic and psycholinguistic features for complex word identification. *arXiv preprint arXiv:1709.02843*. https://arxiv.org/abs/1709.02843

