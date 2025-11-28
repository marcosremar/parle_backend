# Papers Não Baixados - Informações para Download Manual

Este documento lista os papers que não foram baixados automaticamente porque estão em plataformas que requerem acesso ou download manual.

---

## ❌ Papers Não Baixados (5 papers)

### 1. Fluência Leitora (2023)
- **Título:** "Avaliação de fluência leitora em língua portuguesa: primeira experiência com uso em larga escala de Inteligência Artificial"
- **Fonte:** SBIE 2023 (Simpósio Brasileiro de Informática na Educação)
- **Link:** https://sol.sbc.org.br/index.php/sbie/article/view/31470
- **Acesso:** Pode requerer login ou ser gratuito
- **Download:** Interface web do SBIE

### 2. ASR Leituras (2022)
- **Título:** "Classificação Automática de Áudios de Leituras de Pseudopalavras para Avaliação em Larga Escala de Fluência da Leitura de Crianças em Fase de Alfabetização"
- **Fonte:** SBIE 2022
- **Link:** https://sol.sbc.org.br/index.php/sbie/article/view/22393
- **Acesso:** Pode requerer login ou ser gratuito
- **Download:** Interface web do SBIE

### 3. ASR Qualidade (2022)
- **Título:** "Avaliação de modelos para reconhecimento automático de fala aplicados para identificação da qualidade de leituras em voz alta de narrativas breves"
- **Fonte:** SBIE 2022
- **Link:** https://sol.sbc.org.br/index.php/sbie/article/view/22468
- **Acesso:** Pode requerer login ou ser gratuito
- **Download:** Interface web do SBIE

### 4. CELPE-Bras (2023)
- **Título:** "Um estudo sobre a dimensionalidade das escalas de avaliação da proficiência oral do Certificado de Proficiência em Língua Portuguesa para Estrangeiros"
- **Fonte:** SciELO (Scientific Electronic Library Online)
- **Link:** https://www.scielo.br/j/ep/a/KWYysnwZJK7xFL6NfvkjdND/?lang=en
- **Acesso:** Geralmente gratuito, mas pode requerer acesso institucional
- **Download:** Botão de download PDF na página do artigo

### 5. Fonologia (2023)
- **Título:** "Instrumento de Avaliação Fonológica: Evidências de Fidedignidade"
- **Fonte:** CoDAS (Revista de Fonoaudiologia)
- **Link:** https://www.codas.org.br/article/doi/10.1590/2317-1782/20232022303pt
- **Acesso:** Pode requerer login ou ser gratuito
- **Download:** Interface web do CoDAS

---

## 📥 Como Baixar Manualmente

### Opção 1: Download via Interface Web
1. Acesse o link do paper
2. Procure por botão "Download PDF" ou "Baixar PDF"
3. Salve o arquivo no diretório `v2/` com nome descritivo
4. Execute a conversão para Markdown (veja abaixo)

### Opção 2: Tentativa de Download Automático
Alguns papers podem ter URLs diretas de PDF. Você pode tentar:

```bash
# Exemplo (pode não funcionar para todos):
cd /Users/marcos/Documents/projects/backend/parle_backend/papers/avaliacao-fala/v2

# Tentar baixar (substitua URL pela URL real do PDF)
wget -O "Fluencia_Leitora_2023.pdf" "URL_DO_PDF"
```

### Opção 3: Conversão Após Download Manual
Após baixar manualmente, converta para Markdown:

```python
import pdf4llm

# Converter PDF para Markdown
markdown_text = pdf4llm.to_markdown("paper.pdf")

# Salvar
with open("paper.md", "w", encoding="utf-8") as f:
    f.write(markdown_text)
```

---

## 🔍 Verificação de Disponibilidade

Alguns papers podem estar disponíveis em outras fontes:

1. **ResearchGate:** https://www.researchgate.net/
2. **Academia.edu:** https://www.academia.edu/
3. **Google Scholar:** https://scholar.google.com/
4. **Semantic Scholar:** https://www.semanticscholar.org/

---

## 📝 Notas

- **SBIE:** Geralmente disponível gratuitamente, mas pode requerer registro
- **SciELO:** Geralmente gratuito, acesso aberto
- **CoDAS:** Pode requerer login ou ser gratuito dependendo do artigo

---

## ✅ Após Download

1. Coloque o PDF no diretório `v2/`
2. Use o nome descritivo (ex: `Fluencia_Leitora_2023.pdf`)
3. Execute a conversão para Markdown
4. Atualize o `README.md` com o novo paper

---

**Última atualização:** 2025-11-23
