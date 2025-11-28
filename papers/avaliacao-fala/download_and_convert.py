#!/usr/bin/env python3
"""
Script para baixar PDFs de papers sobre avaliação de fala e converter para Markdown
"""

import os
import sys
import requests
from pathlib import Path
import pdf4llm

# Adicionar projeto root ao path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Diretório de destino
PAPERS_DIR = Path(__file__).parent
PAPERS_DIR.mkdir(parents=True, exist_ok=True)

# Papers para baixar (URLs diretas ou arXiv IDs)
PAPERS = [
    {
        "name": "nilc_metrix_2022",
        "url": "https://arxiv.org/pdf/2201.03445.pdf",
        "title": "NILC-Metrix: Assessing the complexity of written and spoken language in Brazilian Portuguese"
    },
    {
        "name": "arnold_cefr_2018",
        "url": "https://arxiv.org/pdf/1806.11099.pdf",
        "title": "Predicting CEFRL levels in learner English on the basis of metrics and full texts"
    },
    {
        "name": "vajjala_rama_2021",
        "url": "https://aclanthology.org/2021.bea-1.21.pdf",
        "title": "Automated classification of written proficiency levels on the CEFR-scale through complexity contours and RNNs"
    },
    {
        "name": "speech_lm_score_2022",
        "url": "https://arxiv.org/pdf/2212.04559.pdf",
        "title": "SpeechLMScore: Evaluating speech generation using speech language model"
    },
    {
        "name": "llm_eval_2023",
        "url": "https://arxiv.org/pdf/2305.13711.pdf",
        "title": "LLM-Eval: Unified multi-dimensional automatic evaluation for open-domain conversations with large language models"
    }
]


def download_pdf(url: str, output_path: Path) -> bool:
    """Baixa um PDF de uma URL"""
    try:
        print(f"📥 Baixando: {output_path.name}...")
        response = requests.get(url, timeout=30, stream=True)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"✅ Baixado: {output_path.name}")
        return True
    except Exception as e:
        print(f"❌ Erro ao baixar {output_path.name}: {e}")
        return False


def convert_to_markdown(pdf_path: Path) -> bool:
    """Converte PDF para Markdown usando pdf4llm"""
    try:
        md_path = pdf_path.with_suffix('.md')
        
        if md_path.exists():
            print(f"⏭️  Markdown já existe: {md_path.name}")
            return True
        
        print(f"🔄 Convertendo para Markdown: {pdf_path.name}...")
        md_text = pdf4llm.to_markdown(str(pdf_path))
        
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_text)
        
        print(f"✅ Convertido: {md_path.name}")
        return True
    except Exception as e:
        print(f"❌ Erro ao converter {pdf_path.name}: {e}")
        return False


def main():
    print(f"\n{'='*80}")
    print(f"📚 BAIXANDO E CONVERTENDO PAPERS SOBRE AVALIAÇÃO DE FALA")
    print(f"{'='*80}\n")
    
    downloaded = 0
    converted = 0
    
    for paper in PAPERS:
        pdf_path = PAPERS_DIR / f"{paper['name']}.pdf"
        
        # Baixar PDF se não existir
        if not pdf_path.exists():
            if download_pdf(paper['url'], pdf_path):
                downloaded += 1
        else:
            print(f"⏭️  PDF já existe: {pdf_path.name}")
        
        # Converter para Markdown
        if pdf_path.exists():
            if convert_to_markdown(pdf_path):
                converted += 1
    
    print(f"\n{'='*80}")
    print(f"📊 RESUMO:")
    print(f"   PDFs baixados: {downloaded}")
    print(f"   Markdowns criados: {converted}")
    print(f"   Total de papers: {len(PAPERS)}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

