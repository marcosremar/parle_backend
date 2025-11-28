#!/usr/bin/env python3
"""
Script para converter PDFs para Markdown usando pdf4llm
Mantém o mesmo nome do arquivo, apenas muda a extensão
"""

import os
import sys
from pathlib import Path

try:
    import fitz  # PyMuPDF
    from pdf4llm import to_markdown
except ImportError:
    print("Instalando pdf4llm e PyMuPDF...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pdf4llm", "pymupdf", "-q"])
    import fitz  # PyMuPDF
    from pdf4llm import to_markdown

def convert_pdf_to_md(pdf_path: Path):
    """Converte um PDF para Markdown mantendo o mesmo nome"""
    if not pdf_path.exists():
        print(f"❌ Arquivo não encontrado: {pdf_path}")
        return False
    
    # Cria o nome do arquivo Markdown (mesmo nome, extensão .md)
    md_path = pdf_path.with_suffix('.md')
    
    try:
        print(f"📄 Convertendo: {pdf_path.name} -> {md_path.name}")
        
        # Abre o PDF com PyMuPDF
        doc = fitz.open(str(pdf_path))
        
        # Converte PDF para Markdown usando pdf4llm
        markdown_text = to_markdown(doc, filename=str(pdf_path))
        
        # Fecha o documento
        doc.close()
        
        # Salva o Markdown
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(markdown_text)
        
        print(f"✅ Convertido com sucesso: {md_path.name}")
        return True
        
    except Exception as e:
        print(f"❌ Erro ao converter {pdf_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    # Diretório atual (onde está o script)
    current_dir = Path(__file__).parent
    
    # Encontra todos os PDFs no diretório
    pdf_files = list(current_dir.glob("*.pdf"))
    
    if not pdf_files:
        print("❌ Nenhum arquivo PDF encontrado no diretório")
        return
    
    print(f"📚 Encontrados {len(pdf_files)} arquivos PDF\n")
    
    success_count = 0
    for pdf_file in pdf_files:
        if convert_pdf_to_md(pdf_file):
            success_count += 1
        print()  # Linha em branco entre conversões
    
    print(f"\n✅ Conversão concluída: {success_count}/{len(pdf_files)} arquivos convertidos")

if __name__ == "__main__":
    main()

